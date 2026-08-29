"""Multistep (autoregressive-rollout) fine-tuning of DexWM on DexJoCo demos.

`train_dexjoco_ft.py` trains the single-step objective: one forward pass, loss
against the DINOv2 latent of the actual next frame.  The A6 report showed that
the resulting predictor collapses when it is asked to eat its own predictions -
rolling out to h=30 compressed the between-candidate latent spread from 0.266 to
0.120 and pushed the Set-A winrate from 0.628 down to 0.544.  That is textbook
exposure bias: the rollout distribution never appears in training.

This script closes the loop.  It runs the *same* autoregressive `prev_emb` loop
the evaluator runs (`scripts/dexwm/rollout_branch_candidates.rollout`, which in
turn reproduces the public planner) inside the training step, and takes the loss
against the DINOv2 latents of the N actual future frames.  The mechanics mirror
upstream `train_multistep_wm.py:335-364` (per-step `model(..., prev_emb=own
predictions)`, MSE over all N predicted latents), with three deliberate
differences, all of them so that training and our evaluation see the same thing:

  * the context is our 8 *real* frames padded on the right with copies of the
    last one (upstream repeats frame 0 for every slot, i.e. a single-frame
    context - that is not what our rollout evaluator does);
  * the leading `num_context - 1` action slots are zero and the tail slot of
    window n carries the delta into predicted frame n+1, exactly as in
    `rollout_branch_candidates.rollout`;
  * the auxiliary heatmap keypoint loss is OPTIONAL: with `--kp-labels` the
    predictor's otherwise unused `kp_layer` is trained on 2-D keypoint targets
    manufactured from the demo videos (see `scripts/dexwm/label_demo_keypoints.py`
    in the VLS repo), i.e. the geometric supervision the DexJoCo demos do not
    ship; without it the head stays frozen and stubbed exactly as before.

The DINOv2 encoder is frozen and `DexWM.encode_image` already runs under
`torch.no_grad`, so the 8 + N distinct frames of a window are encoded ONCE per
step and the per-rollout-step windows are served from that cache - the same
`_frozen_encoder` trick the evaluator uses.  Without it an N-step window would
re-encode 9 frames N times for nothing.

The checkpoint layout is unchanged, so anything written here loads into
`scripts/dexwm/score_branch_candidates.load_model` with `strict=True`.

Usage (curriculum: N=3 then N=6, both from the single-step checkpoint):
    python train_dexjoco_ms.py --n-future 3 --max-steps 1500 \
        --resume /root/ckpt_local/dexwm/finetune/seed0/best.pth.tar \
        --out /root/ckpt_local/dexwm/finetune/ms/n3
    python train_dexjoco_ms.py --n-future 6 --max-steps 1500 \
        --resume /root/ckpt_local/dexwm/finetune/ms/n3/best.pth.tar \
        --out /root/ckpt_local/dexwm/finetune/ms/n6

Keypoint-supervised (E2: geometric supervision through the heatmap head):
    python train_dexjoco_ms.py --n-future 6 --max-steps 1500 --lr 1e-5 --batch 3 \
        --kp-labels /root/dexwm_ft/kp_labels --kp-weight 100 \
        --resume /root/ckpt_local/dexwm/finetune/ms/e1/best.pth.tar \
        --out /root/ckpt_local/dexwm/finetune/kp

Evaluation-only (M1: score an existing checkpoint under this same protocol):
    python train_dexjoco_ms.py --eval-only --n-future 6 \
        --resume <ckpt> --out /tmp/whatever
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train_dexjoco_ft import build_model, load_dataset_module, load_weights  # noqa: E402


@contextlib.contextmanager
def frozen_encoder(model, latents: torch.Tensor):
    """Serve a precomputed patch-token window instead of re-running DINOv2.

    Byte-for-byte the same idea as
    `scripts/dexwm/rollout_branch_candidates._frozen_encoder`: every rollout
    step of a window looks at a 9-frame slice of the same padded stack, so the
    encoder only ever sees `num_context + n_future` distinct frames per window.
    `DexWM.forward` adds `pos_embed` out-of-place before its only in-place write
    (`x_emb[:, 1:] = ...`), so the cached tensor is never mutated.
    """
    original = model.encode_image
    model.encode_image = lambda _x: latents
    try:
        yield
    finally:
        model.encode_image = original


HEATMAP_GRID = [16, 28]   # DINOv2 patch grid of the 392x224 crop


# The public checkpoint's 12 heatmap channels are, in order,
# [left palm, left thumb, left index, left middle, left ring, left little,
#  right palm, right thumb, right index, right middle, right ring, right little]
# - RoboCasa zeroes the left half (`robocasa_random_movement.py:239-241`) and
# builds the right half from MANO indices [0, 4, 8, 12, 16, 20] with the ring
# finger duplicated as the absent little finger.  Five of our seven channels
# therefore already exist in the pre-trained head.
UPSTREAM_RIGHT_HAND = {"palm": 6, "thumb_tip": 7, "index_tip": 8,
                       "middle_tip": 9, "ring_tip": 10, "little_tip": 11}
CHANNEL_INHERITANCE = {
    "index_tip": 8, "middle_tip": 9, "ring_tip": 10, "thumb_tip": 7,
    "palm": 6,
    # no pre-trained analogue: start both from the spare duplicated little
    # finger, which already emits a sharp blob of the right dynamic range near
    # the hand - which is where the grasped object is
    "object": 11, "bucket": 11,
}


def retarget_kp_head(model, channel_names, verbose: bool = True) -> None:
    """Point the pre-trained heatmap head at OUR channel set.

    Everything in `kp_layer` except the final patch projection is kept.  The
    projection itself is not re-initialised from scratch: each of our channels
    copies the rows of the pre-trained channel it corresponds to
    (`CHANNEL_INHERITANCE`), so the head starts out already able to localise a
    palm and four fingertips in this crop.

    A zero-initialised head was tried first and DIVERGED: held-out kp loss went
    from 1.43e-4 (the target's own energy, i.e. what predicting nothing scores)
    to 2.82e-4 by step 150 while emb degraded 3.7%.  Learning a 256->14*14*C
    spatial code from scratch in 1500 steps is not a thing that happens.

    The head's output is laid out `(p1 p2 c)`, so channel `c` of a `C`-channel
    head owns rows `c::C`.
    """
    if isinstance(channel_names, int):
        channel_names = [f"kp{i}" for i in range(channel_names)]
    channels = len(channel_names)
    decoder = model.kp_layer.vit_decoder
    old = decoder.head[0]
    patch = decoder.patch_size
    old_channels = old.weight.shape[0] // (patch * patch)
    new = torch.nn.Linear(old.in_features, patch * patch * channels,
                          bias=old.bias is not None,
                          device=old.weight.device, dtype=old.weight.dtype)
    torch.nn.init.zeros_(new.weight)
    if new.bias is not None:
        torch.nn.init.zeros_(new.bias)
    with torch.no_grad():
        source = old.weight.view(patch, patch, old_channels, old.in_features)
        target = new.weight.view(patch, patch, channels, old.in_features)
        source_bias = old.bias.view(patch, patch, old_channels)
        target_bias = new.bias.view(patch, patch, channels)
        inherited = []
        for index, name in enumerate(channel_names):
            donor = CHANNEL_INHERITANCE.get(name)
            if donor is None or donor >= old_channels:
                continue
            target[:, :, index] = source[:, :, donor]
            target_bias[:, :, index] = source_bias[:, :, donor]
            inherited.append(f"{name}<-{donor}")
    decoder.head[0] = new
    decoder.output_channels = channels
    if verbose:
        print(f"[kp] head {old_channels}->{channels} channels, inherited "
              f"{' '.join(inherited) if inherited else '(none)'}", flush=True)


@torch.no_grad()
def calibrate_kp_head(model, loader, device, num_context, n_future,
                      batches: int = 16) -> float:
    """Fit the one scalar the inherited head is wrong about: its gain.

    The pre-trained head localises a palm and four fingertips in our frames to
    within ~20 px straight out of the box, but its maps peak at ~0.03 where the
    target peaks at 1.0 - it was trained on RoboCasa, and on DexJoCo it is
    simply unconfident.  Under a squared loss the optimal rescaling is available
    in closed form, `alpha = <pred, target> / <pred, pred>` over the valid
    entries, so it is fit once on TRAINING windows before the first step
    instead of being spent as gradient steps.
    """
    model.eval()
    numerator = denominator = 0.0
    for index, (frames, actions, _rel, heatmaps, valid_kp, *_rest) in enumerate(loader):
        if index >= batches:
            break
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        valid_kp = valid_kp.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, _, maps = multistep_predict(model, frames, actions, num_context,
                                           n_future, with_kp=True)
        maps = maps.float()
        mask = valid_kp[..., None, None]
        numerator += float((maps * heatmaps * mask).sum())
        denominator += float((maps * maps * mask).sum())
    model.train()
    alpha = numerator / max(denominator, 1e-12)
    head = model.kp_layer.vit_decoder.head[0]
    head.weight.mul_(alpha)
    if head.bias is not None:
        head.bias.mul_(alpha)
    return alpha


def freeze_backbone(model, keep_last: int) -> None:
    """Forgetting control: train only the top of the predictor plus the head.

    If the keypoint-supervised run fails, the honest question is whether the
    fine-tune OVERWROTE the pre-trained interaction knowledge or whether that
    knowledge was never there.  Freezing everything below the last `keep_last`
    CDiT blocks (the final layer, the keypoint head and those blocks stay
    trainable) answers it: the frozen variant cannot forget.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for block in model.blocks[-keep_last:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for module in (model.final_layer, model.kp_layer):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.pos_embed.requires_grad_(True)
    for parameter in model.image_embedder.parameters():
        parameter.requires_grad_(False)


def kp_forward_last(model):
    """`DexWM.forward_kp` restricted to the newest predicted frame.

    Upstream decodes all `num_context` slots and then throws away everything but
    `[:, -1:]` (train_multistep_wm.py:344-346).  In an N-step rollout that is 8x
    the decoder work for nothing, so this override decodes the one slot the
    rollout actually supervises.  The loss itself is computed by the caller, so
    the `gt_kps` / `valid_kp` arguments are ignored here.
    """
    def forward_kp(x_emb, cam_pose=None, gt_kps=None, valid_kp=None):
        return model.kp_layer(x_emb[:, -1:], HEATMAP_GRID), None
    return forward_kp


def kponly_loss(model, frames, deltas, num_context, n_future, heatmaps,
                valid_kp, cache_encoder=True, target_latent=False):
    """Head-only objective: train `kp_layer` on the DETACHED predicted latents.

    This asks the question `dexwm_multistep.md` left open - *is the geometry
    already in the predicted latent, and only the readout was wrong?* - without
    letting the auxiliary loss touch the predictor.  Two consequences: the
    embedding objective cannot degrade (K3 holds by construction), and the
    rollout itself needs no graph, so it runs under `inference_mode` and the
    step is several times cheaper than the joint one.
    """
    original = model.forward_kp
    model.forward_kp = lambda *_a, **_k: (None, None)
    try:
        with torch.no_grad():
            predicted, target, _ = multistep_predict(
                model, frames, deltas, num_context, n_future,
                cache_encoder=cache_encoder, with_kp=False)
    finally:
        model.forward_kp = original
    per_step = (predicted.float() - target).square().mean(dim=(0, 2, 3))
    stats = {"emb": per_step.mean().detach(), "emb_per_step": per_step.detach()}
    # `target_latent` trains the same head on the DINOv2 latents of the frames
    # that ACTUALLY happen.  It is the control that separates "the head cannot
    # localise keypoints in a DexWM latent at all" from "the head can, but the
    # PREDICTED latent does not carry the geometry" - and it produces the oracle
    # head that `score_kp_rel_disp.py --source oracle` needs.
    source = (target if target_latent else predicted).detach()
    maps = model.kp_layer(source, HEATMAP_GRID)
    squared = (maps.float() - heatmaps).square().mean(dim=(-2, -1))
    masked = squared * valid_kp
    kp_loss = masked.mean()
    fraction = valid_kp.mean().clamp_min(1e-6)
    stats["kp"] = kp_loss.detach()
    stats["kp_masked"] = (kp_loss / fraction).detach()
    stats["kp_per_step"] = (masked.mean(dim=(0, 2))
                            / valid_kp.mean(dim=(0, 2)).clamp_min(1e-6)).detach()
    stats["kp_valid"] = fraction.detach()
    return kp_loss, stats


def rollout_windows(frames: torch.Tensor, deltas: torch.Tensor,
                    num_context: int, n_future: int):
    """Padded image stack, padded action stack and the N future deltas.

    frames  (B, num_context + N, C, H, W) - N future frames after the context
    deltas  (B, num_context - 1 + N, A)   - consecutive keypoint deltas

    The last N deltas are the transitions into the N future frames; the leading
    `num_context - 1` are context transitions and are replaced by zeros, because
    that is what the evaluator feeds (`padded_actions = [0]*7 ++ deltas`).
    """
    future = deltas[:, num_context - 1:]
    if future.shape[1] != n_future:
        raise ValueError(f"expected {n_future} future deltas, got {future.shape[1]}")
    zeros = torch.zeros(future.shape[0], num_context - 1, future.shape[-1],
                        dtype=future.dtype, device=future.device)
    padded_actions = torch.cat([zeros, future], dim=1)
    context = frames[:, :num_context]
    padded_frames = torch.cat(
        [context, context[:, -1:].expand(-1, n_future, -1, -1, -1)], dim=1)
    return padded_frames, padded_actions


def multistep_predict(model, frames: torch.Tensor, deltas: torch.Tensor,
                      num_context: int, n_future: int,
                      detach_prev: bool = False, cache_encoder: bool = True,
                      with_kp: bool = False):
    """Autoregressive rollout -> (predicted latents, target latents, heatmaps).

    Returns (B, N, P, D) predictions and the DINOv2 latents of the N actual
    future frames, plus (B, N, C, 224, 392) predicted keypoint belief maps when
    `with_kp` (otherwise None).  Gradients flow through the whole chain unless
    `detach_prev`, which turns the loop into a stop-gradient rollout (still
    fixes exposure bias, but no backprop-through-time).
    """
    padded_frames, padded_actions = rollout_windows(
        frames, deltas, num_context, n_future)

    if cache_encoder:
        # one encoder pass over the 8 + N distinct frames of the window
        latents = model.encode_image(frames)                     # (B, T, P, D)
        context_latent = latents[:, :num_context]
        target = latents[:, num_context:].float()
        padded_latent = torch.cat(
            [context_latent,
             context_latent[:, -1:].expand(-1, n_future, -1, -1)], dim=1)
    else:
        target = model.encode_image(frames[:, num_context:]).float()
        padded_latent = None

    predictions, context_chain, heatmaps = [], [], []
    prev_emb = None
    for step in range(n_future):
        images = padded_frames[:, step:step + num_context + 1]
        actions = padded_actions[:, step:step + num_context]
        window = (padded_latent[:, step:step + num_context + 1]
                  if cache_encoder else None)
        guard = (frozen_encoder(model, window) if cache_encoder
                 else contextlib.nullcontext())
        with guard:
            pred, _goal, pred_kp, *_ = model(images, actions, prev_emb=prev_emb,
                                             action_diff=True)
        latest = pred[:, -1:]
        predictions.append(latest)
        if with_kp:
            heatmaps.append(pred_kp)
        context_chain.append(latest.detach() if detach_prev else latest)
        prev_emb = torch.cat(context_chain, dim=1)
    maps = torch.cat(heatmaps, dim=1) if with_kp else None
    return torch.cat(predictions, dim=1), target, maps


def multistep_loss(model, frames, deltas, num_context, n_future,
                   detach_prev=False, cache_encoder=True,
                   heatmaps=None, valid_kp=None, kp_weight=0.0,
                   kp_only=False, kp_target_real=False):
    """Rollout loss.  `emb + kp_weight * kp` when keypoint targets are supplied.

    The keypoint term reproduces upstream `train_multistep_wm.py:361-364`: mean
    squared error per heatmap, multiplied by the per-(step, channel) validity
    mask, then averaged over EVERY entry - so an invalid channel contributes a
    zero rather than being dropped from the denominator, and the calibrated
    `kp_weight=100` keeps its meaning.  `kp_masked` reports the same quantity
    normalised by the valid fraction, which is the number to compare across runs
    with different label coverage.
    """
    if kp_only and heatmaps is not None:
        return kponly_loss(model, frames, deltas, num_context, n_future,
                           heatmaps, valid_kp, cache_encoder, kp_target_real)
    predicted, target, maps = multistep_predict(
        model, frames, deltas, num_context, n_future, detach_prev, cache_encoder,
        with_kp=heatmaps is not None)
    per_step = (predicted.float() - target).square().mean(dim=(0, 2, 3))
    emb_loss = per_step.mean()
    stats = {"emb": emb_loss.detach(), "emb_per_step": per_step.detach()}
    if heatmaps is None:
        return emb_loss, stats
    squared = (maps.float() - heatmaps).square().mean(dim=(-2, -1))   # (B, N, C)
    masked = squared * valid_kp
    kp_loss = masked.mean()
    fraction = valid_kp.mean().clamp_min(1e-6)
    stats["kp"] = kp_loss.detach()
    stats["kp_masked"] = (kp_loss / fraction).detach()
    stats["kp_per_step"] = (masked.mean(dim=(0, 2))
                            / valid_kp.mean(dim=(0, 2)).clamp_min(1e-6)).detach()
    stats["kp_valid"] = fraction.detach()
    return emb_loss + kp_weight * kp_loss, stats


def chunk_predict(model, frames, deltas, num_context, n_future,
                  cache_encoder=True):
    """Non-autoregressive: s_{1..8} + {a_1..a_N} -> s_{8+N}.  Single forward."""
    action_chunk = deltas[:, num_context - 1 : num_context - 1 + n_future]

    context = frames[:, :num_context]
    model_frames = torch.cat([context, context[:, -1:]], dim=1)

    if cache_encoder:
        latents = model.encode_image(frames)
        target = latents[:, num_context + n_future - 1 :
                            num_context + n_future].float()
        ctx_latent = latents[:, :num_context]
        window = torch.cat([ctx_latent, ctx_latent[:, -1:]], dim=1)
        with frozen_encoder(model, window):
            pred, _goal, pred_kp, *_ = model(
                model_frames, action_chunk=action_chunk, action_diff=True)
    else:
        target = model.encode_image(
            frames[:, num_context + n_future - 1 :
                      num_context + n_future]).float()
        pred, _, pred_kp, *_ = model(
            model_frames, action_chunk=action_chunk, action_diff=True)

    predicted = pred[:, -1:]
    return predicted, target


def chunk_loss(model, frames, deltas, num_context, n_future,
               cache_encoder=True):
    """MSE loss for non-autoregressive chunk prediction."""
    predicted, target = chunk_predict(
        model, frames, deltas, num_context, n_future, cache_encoder)
    emb_loss = (predicted.float() - target).square().mean()
    return emb_loss, {"emb": emb_loss.detach()}


def seqext_predict(model, frames, deltas, num_context, n_future,
                   cache_encoder=True):
    """Non-autoregressive sequence extension: single forward, per-step actions.

    Feeds num_context real frames + n_future copies of the last context frame,
    with per-step future actions via adaLN.  The model predicts all N future
    latents in one pass; loss is taken against DINOv2 latents of the actual
    future frames.
    """
    context = frames[:, :num_context]
    padded_frames = torch.cat(
        [context, context[:, -1:].expand(-1, n_future, -1, -1, -1)], dim=1)

    future_actions = deltas[:, num_context - 1:num_context - 1 + n_future]
    zeros = torch.zeros(future_actions.shape[0], num_context - 1,
                        future_actions.shape[-1],
                        dtype=future_actions.dtype, device=future_actions.device)
    padded_actions = torch.cat([zeros, future_actions], dim=1)

    if cache_encoder:
        all_latents = model.encode_image(frames)
        target = all_latents[:, num_context:num_context + n_future].float()
        ctx_latent = all_latents[:, :num_context]
        padded_latent = torch.cat(
            [ctx_latent,
             ctx_latent[:, -1:].expand(-1, n_future, -1, -1)], dim=1)
        with frozen_encoder(model, padded_latent):
            pred, _goal, _kp, *_ = model(
                padded_frames, padded_actions, action_diff=True)
    else:
        target = model.encode_image(
            frames[:, num_context:num_context + n_future]).float()
        pred, _goal, _kp, *_ = model(
            padded_frames, padded_actions, action_diff=True)

    predicted = pred[:, num_context:]
    return predicted, target


def seqext_loss(model, frames, deltas, num_context, n_future,
                cache_encoder=True):
    """MSE loss for sequence-extension non-autoregressive prediction."""
    predicted, target = seqext_predict(
        model, frames, deltas, num_context, n_future, cache_encoder)
    per_step = (predicted.float() - target).square().mean(dim=(0, 2, 3))
    emb_loss = per_step.mean()
    return emb_loss, {"emb": emb_loss.detach(), "emb_per_step": per_step.detach()}


@torch.no_grad()
def validate(model, loader, device, num_context, n_future, limit=0,
             cache_encoder=True, kp_weight=0.0, with_kp=False,
             kp_only=False, kp_target_real=False, use_chunk=False,
             use_seqext=False):
    model.eval()
    totals, count = {}, 0
    for index, (frames, actions, _rel, heatmaps, valid_kp, *_rest) in enumerate(loader):
        if limit and index >= limit:
            break
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        if with_kp:
            heatmaps = heatmaps.to(device, non_blocking=True)
            valid_kp = valid_kp.to(device, non_blocking=True)
        else:
            heatmaps = valid_kp = None
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if use_seqext:
                _loss, stats = seqext_loss(
                    model, frames, actions, num_context, n_future,
                    cache_encoder)
            elif use_chunk:
                _loss, stats = chunk_loss(
                    model, frames, actions, num_context, n_future,
                    cache_encoder)
            else:
                _loss, stats = multistep_loss(
                    model, frames, actions, num_context, n_future,
                    cache_encoder=cache_encoder, heatmaps=heatmaps,
                    valid_kp=valid_kp, kp_weight=kp_weight, kp_only=kp_only,
                    kp_target_real=kp_target_real)
        for key, value in stats.items():
            contribution = value.float().cpu().numpy()
            totals[key] = contribution if key not in totals else totals[key] + contribution
        count += 1
    model.train()
    count = max(count, 1)
    out = {key: (value / count) for key, value in totals.items()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("/root/dexwm_ft/frames"))
    parser.add_argument("--resume", type=Path,
                        default=Path("/root/ckpt_local/dexwm/finetune/seed0/best.pth.tar"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--world-to-camera", type=Path,
                        default=Path("/root/dexwm_branch/data/branch_seed00_step00090.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-future", type=int, default=3)
    parser.add_argument("--num-context", type=int, default=8)
    parser.add_argument("--context-stride", type=int, default=5,
                        help="control steps between DexWM frames; demos=5 (per-control-step files), stride-5 traces=1")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=1,
                        help="gradient accumulation steps (effective batch = batch*accum)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--windows-per-episode", type=int, default=40)
    parser.add_argument("--val-windows-per-episode", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--detach-prev", action="store_true",
                        help="stop-gradient rollout (fallback if BPTT OOMs)")
    parser.add_argument("--no-cache-encoder", dest="cache_encoder",
                        action="store_false")
    parser.add_argument("--eval-only", action="store_true",
                        help="score --resume under this protocol and exit")
    parser.add_argument("--eval-tag", default="",
                        help="name written into the eval-only summary")
    parser.add_argument("--kp-labels", type=Path, default=None,
                        help="directory of per-episode 2-D keypoint labels "
                             "(scripts/dexwm/label_demo_keypoints.py); enables "
                             "the auxiliary heatmap loss")
    parser.add_argument("--kp-weight", type=float, default=100.0,
                        help="upstream train_wm.py weight on the heatmap loss")
    parser.add_argument("--kp-sigma", type=float, default=2.0)
    parser.add_argument("--kp-calibrate-batches", type=int, default=16,
                        help="training batches used to fit the inherited head's "
                             "gain in closed form before step 1 (0 disables)")
    parser.add_argument("--kp-head-lr", type=float, default=3e-4,
                        help="lr for the WHOLE auxiliary heatmap head "
                             "(kp_layer: decoder trunk + retargeted patch "
                             "projection); the predictor itself uses --lr. The "
                             "head is being repurposed from RoboCasa hand "
                             "keypoints to our 7 channels, so it has more to "
                             "learn than the predictor does")
    parser.add_argument("--best-metric", choices=["total", "emb"], default="total",
                        help="which validation number selects best.pth.tar")
    parser.add_argument("--kp-only", action="store_true",
                        help="train ONLY kp_layer, on detached predicted "
                             "latents: the predictor is frozen, so the "
                             "embedding objective cannot degrade and the "
                             "rollout runs without a graph")
    parser.add_argument("--kp-target-real", action="store_true",
                        help="with --kp-only, train the head on the DINOv2 "
                             "latents of the frames that actually happen "
                             "instead of the predicted ones (head-quality "
                             "control + the oracle head)")
    parser.add_argument("--freeze-blocks", type=int, default=0,
                        help="forgetting control: keep only the last N CDiT "
                             "blocks (+ final layer + kp head + pos_embed) "
                             "trainable; 0 trains the whole predictor")
    parser.add_argument("--chunk", action="store_true",
                        help="non-autoregressive chunk prediction: encode all "
                             "N future actions at once via ActionChunkEncoder "
                             "and predict s_{t+N} in a single forward pass")
    parser.add_argument("--chunk-encoder-lr", type=float, default=5e-5,
                        help="separate lr for ActionChunkEncoder (new module)")
    parser.add_argument("--seqext", action="store_true",
                        help="non-autoregressive sequence extension: extend "
                             "CDiT input to num_context+n_future frames with "
                             "per-step action conditioning (DriftWorld-style)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile-attention", dest="compile_attention",
                        action="store_false")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "curve.jsonl"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    demos = load_dataset_module()
    with np.load(args.world_to_camera, allow_pickle=False) as reference:
        world_to_camera = np.asarray(reference["world_to_camera"])

    common = dict(world_to_camera=world_to_camera, n_future=args.n_future,
                  num_context=args.num_context, seed=args.seed,
                  kp_root=args.kp_labels, kp_sigma=args.kp_sigma,
                  context_stride=args.context_stride)
    val_set = demos.DexJoCoDemoDataset(
        args.frames, train=False,
        windows_per_episode=args.val_windows_per_episode, **common)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False, persistent_workers=args.workers > 0)

    n_future_model = args.n_future if args.seqext else 0
    model = build_model(args.device, compile_attention=args.compile_attention,
                        n_future=n_future_model)
    load_weights(model, args.resume, strict=not (args.chunk or args.seqext))
    for parameter in model.image_embedder.parameters():
        parameter.requires_grad_(False)

    # --- seqext mode: same param groups as multistep, no new modules ---
    if args.seqext:
        model.forward_kp = lambda *_a, **_k: (None, torch.zeros((), device=args.device))
        for p in model.kp_layer.parameters():
            p.requires_grad_(False)
        for p in model.action_chunk_encoder.parameters():
            p.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": trainable, "lr": args.lr}]

    # --- chunk mode: separate param groups for CDiT body vs ActionChunkEncoder ---
    elif args.chunk:
        model.forward_kp = lambda *_a, **_k: (None, torch.zeros((), device=args.device))
        chunk_params = list(model.action_chunk_encoder.parameters())
        chunk_ids = {id(p) for p in chunk_params}
        for p in model.kp_layer.parameters():
            p.requires_grad_(False)
        body_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in chunk_ids]
        groups = [{"params": body_params, "lr": args.lr},
                  {"params": chunk_params, "lr": args.chunk_encoder_lr}]
        trainable = body_params + chunk_params
        print(f"[chunk] encoder params {sum(p.numel() for p in chunk_params)/1e3:.1f}K "
              f"| body lr {args.lr} encoder lr {args.chunk_encoder_lr}", flush=True)

    with_kp = args.kp_labels is not None
    if with_kp:
        channels = val_set.kp_channels
        retarget_kp_head(model, channels)
        model.forward_kp = kp_forward_last(model)
        if args.freeze_blocks:
            freeze_backbone(model, args.freeze_blocks)
        if args.kp_only:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        for parameter in model.kp_layer.parameters():
            parameter.requires_grad_(True)
        head = [p for p in model.kp_layer.parameters() if p.requires_grad]
        head_ids = {id(parameter) for parameter in head}
        body = [p for p in model.parameters()
                if p.requires_grad and id(p) not in head_ids]
        groups = ([{"params": head, "lr": args.kp_head_lr}] if args.kp_only
                  else [{"params": body, "lr": args.lr},
                        {"params": head, "lr": args.kp_head_lr}])
        trainable = head if args.kp_only else body + head
        print(f"[kp] channels {channels} weight {args.kp_weight} "
              f"head_lr {args.kp_head_lr} freeze_blocks {args.freeze_blocks}",
              flush=True)
    elif not args.chunk:
        model.forward_kp = lambda *_a, **_k: (None, torch.zeros((), device=args.device))
        for parameter in model.kp_layer.parameters():
            parameter.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": trainable, "lr": args.lr}]

    if args.eval_only:
        stats = validate(model, val_loader, args.device, args.num_context,
                         args.n_future, args.val_batches, args.cache_encoder,
                         args.kp_weight, with_kp, args.kp_only,
                         args.kp_target_real, use_chunk=args.chunk, use_seqext=args.seqext)
        payload = {"eval_only": True, "tag": args.eval_tag or str(args.resume),
                   "checkpoint": str(args.resume), "n_future": args.n_future,
                   "val_loss": float(stats["emb"]),
                   "val_loss_per_step": stats["emb_per_step"].tolist(),
                   "val_windows": len(val_set)}
        if with_kp:
            payload.update({
                "val_kp_loss": float(stats["kp"]),
                "val_kp_loss_masked": float(stats["kp_masked"]),
                "val_kp_loss_per_step": stats["kp_per_step"].tolist(),
                "val_kp_valid_fraction": float(stats["kp_valid"])})
        print(json.dumps(payload, indent=2), flush=True)
        with (args.out / "eval_only.jsonl").open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        return 0

    train_set = demos.DexJoCoDemoDataset(
        args.frames, train=True,
        windows_per_episode=args.windows_per_episode, **common)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    print(f"[data] N={args.n_future} train episodes {len(train_set.usable)} "
          f"windows/epoch {len(train_set)} (pool {train_set.n_windows_total}) | "
          f"val episodes {len(val_set.usable)} windows {len(val_set)}", flush=True)
    print(f"[model] trainable {sum(p.numel() for p in trainable)/1e6:.1f}M / "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| detach_prev={args.detach_prev}", flush=True)

    optimizer = torch.optim.AdamW(groups, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.max_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[group["lr"] for group in groups],
        total_steps=total_steps, pct_start=max(0.05, 2.0 / total_steps),
        final_div_factor=100.0, cycle_momentum=False)

    def report(stats):
        record = {"val_loss": float(stats["emb"])}
        if "emb_per_step" in stats:
            record["val_loss_per_step"] = stats["emb_per_step"].tolist()
        if with_kp:
            record.update({"val_kp_loss": float(stats["kp"]),
                           "val_kp_loss_masked": float(stats["kp_masked"]),
                           "val_kp_loss_per_step": stats["kp_per_step"].tolist(),
                           "val_kp_valid_fraction": float(stats["kp_valid"])})
            record["val_total"] = record["val_loss"] + args.kp_weight * record["val_kp_loss"]
        else:
            record["val_total"] = record["val_loss"]
        return record

    if with_kp and args.kp_calibrate_batches:
        gain = calibrate_kp_head(model, train_loader, args.device,
                                 args.num_context, args.n_future,
                                 args.kp_calibrate_batches)
        print(f"[kp] inherited-head gain calibrated to {gain:.3f} on "
              f"{args.kp_calibrate_batches} training batches", flush=True)

    model.train()
    baseline_stats = validate(model, val_loader, args.device, args.num_context,
                              args.n_future, args.val_batches, args.cache_encoder,
                              args.kp_weight, with_kp, args.kp_only,
                              args.kp_target_real, use_chunk=args.chunk, use_seqext=args.seqext)
    baseline_record = report(baseline_stats)
    baseline = baseline_record["val_loss"]
    baseline_steps = baseline_record.get("val_loss_per_step")
    mode_tag = "seqext" if args.seqext else ("chunk" if args.chunk else "multistep")
    step_str = f" per-step {['%.4f' % v for v in baseline_steps]}" if baseline_steps else ""
    print(f"[val] resumed-checkpoint baseline {mode_tag}(N={args.n_future}) "
          f"emb {baseline:.5f}{step_str}"
          + (f" | kp {baseline_record['val_kp_loss']:.3e} "
             f"masked {baseline_record['val_kp_loss_masked']:.3e}" if with_kp else ""),
          flush=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps({"step": 0, "baseline": True,
                                 "n_future": args.n_future,
                                 "resume": str(args.resume), **baseline_record}) + "\n")

    best = baseline_record["val_total" if args.best_metric == "total" else "val_loss"]
    step = 0
    micro = 0
    start_time = time.time()
    stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if stop:
            break
        for frames, actions, _rel, heatmaps, valid_kp, *_rest in train_loader:
            frames = frames.to(args.device, non_blocking=True)
            actions = actions.to(args.device, non_blocking=True)
            if with_kp:
                heatmaps = heatmaps.to(args.device, non_blocking=True)
                valid_kp = valid_kp.to(args.device, non_blocking=True)
            else:
                heatmaps = valid_kp = None
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if args.seqext:
                    loss, stats = seqext_loss(
                        model, frames, actions, args.num_context, args.n_future,
                        args.cache_encoder)
                elif args.chunk:
                    loss, stats = chunk_loss(
                        model, frames, actions, args.num_context, args.n_future,
                        args.cache_encoder)
                else:
                    loss, stats = multistep_loss(
                        model, frames, actions, args.num_context, args.n_future,
                        args.detach_prev, args.cache_encoder, heatmaps, valid_kp,
                        args.kp_weight, args.kp_only, args.kp_target_real)
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                rate = step / (time.time() - start_time)
                if "emb_per_step" in stats:
                    detail = " ".join(f"{v:.4f}" for v in
                                      stats["emb_per_step"].float().cpu().numpy())
                    detail_str = f" [{detail}]"
                else:
                    detail_str = ""
                extra = (f" kp {float(stats['kp']):.3e} "
                         f"masked {float(stats['kp_masked']):.3e}"
                         if with_kp else "")
                print(f"[train] epoch {epoch} step {step}/{total_steps} "
                      f"loss {float(loss):.5f} emb {float(stats['emb']):.5f} "
                      f"{detail_str}{extra} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {rate:.3f} it/s "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
                      flush=True)
                entry = {"step": step, "epoch": epoch, "train_loss": float(loss),
                         "train_emb_loss": float(stats["emb"]),
                         "lr": scheduler.get_last_lr()[0],
                         "n_future": args.n_future}
                if with_kp:
                    entry["train_kp_loss"] = float(stats["kp"])
                    entry["train_kp_loss_masked"] = float(stats["kp_masked"])
                with log_path.open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")

            if step % args.eval_freq == 0 or step == total_steps:
                record = report(validate(
                    model, val_loader, args.device, args.num_context,
                    args.n_future, args.val_batches, args.cache_encoder,
                    args.kp_weight, with_kp, args.kp_only,
                    args.kp_target_real, use_chunk=args.chunk, use_seqext=args.seqext))
                value = record["val_total" if args.best_metric == "total"
                               else "val_loss"]
                ps = record.get("val_loss_per_step")
                ps_str = f" per-step {['%.4f' % v for v in ps]}" if ps else ""
                print(f"[val] step {step} emb {record['val_loss']:.5f} "
                      f"(baseline {baseline:.5f}){ps_str}"
                      + (f" | kp {record['val_kp_loss']:.3e} masked "
                         f"{record['val_kp_loss_masked']:.3e} per-step "
                         f"{['%.2e' % v for v in record['val_kp_loss_per_step']]}"
                         if with_kp else ""), flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(
                        {"step": step, "epoch": epoch,
                         "n_future": args.n_future, **record}) + "\n")
                payload = {"model": model.state_dict(),
                           "args": {key: str(value_) if isinstance(value_, Path)
                                    else value_ for key, value_ in vars(args).items()},
                           "step": step, "epoch": epoch,
                           "kp_channels": (val_set.kp_channels if with_kp else None),
                           "baseline_val_loss": baseline, **record}
                torch.save(payload, args.out / "last.pth.tar")
                if not (args.out / "best.pth.tar").exists():
                    # a curriculum phase resumes from the previous phase's
                    # best.pth.tar, so that file has to exist even if this phase
                    # never beats the checkpoint it started from
                    torch.save(payload, args.out / "best.pth.tar")
                if value < best:
                    best = value
                    torch.save(payload, args.out / "best.pth.tar")
                    print(f"[val] new best {best:.5f}", flush=True)
                del payload

            if step >= total_steps:
                stop = True
                break

    elapsed = time.time() - start_time
    print(f"[done] best multistep val ({args.best_metric}) {best:.5f} "
          f"baseline {baseline:.5f} ({elapsed:.0f}s)", flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        {"seed": args.seed, "n_future": args.n_future,
         "resume": str(args.resume),
         "kp_labels": str(args.kp_labels) if with_kp else None,
         "kp_channels": (val_set.kp_channels if with_kp else None),
         "kp_weight": args.kp_weight if with_kp else None,
         "kp_head_lr": args.kp_head_lr if with_kp else None,
         "best_metric": args.best_metric,
         "freeze_blocks": args.freeze_blocks,
         "kp_only": args.kp_only,
         "kp_target_real": args.kp_target_real,
         "baseline": baseline_record,
         "baseline_val_loss": baseline, "baseline_val_loss_per_step": baseline_steps,
         "best_val_loss": best, "steps": step, "total_steps": total_steps,
         "batch": args.batch, "accum": args.accum, "lr": args.lr,
         "detach_prev": args.detach_prev, "wall_clock_s": elapsed,
         "it_per_s": step / max(elapsed, 1e-9),
         "peak_gib": float(torch.cuda.max_memory_allocated() / 2**30),
         "train_episodes": len(train_set.usable),
         "windows_per_epoch": len(train_set),
         "window_pool": train_set.n_windows_total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
