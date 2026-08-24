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
  * the auxiliary heatmap keypoint loss stays disabled (DexJoCo demos carry no
    2-D keypoint annotation), as in the single-step script.

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
                      detach_prev: bool = False, cache_encoder: bool = True):
    """Autoregressive rollout -> (predicted latents, target latents).

    Returns (B, N, P, D) predictions and the DINOv2 latents of the N actual
    future frames.  Gradients flow through the whole chain unless
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

    predictions, context_chain = [], []
    prev_emb = None
    for step in range(n_future):
        images = padded_frames[:, step:step + num_context + 1]
        actions = padded_actions[:, step:step + num_context]
        window = (padded_latent[:, step:step + num_context + 1]
                  if cache_encoder else None)
        guard = (frozen_encoder(model, window) if cache_encoder
                 else contextlib.nullcontext())
        with guard:
            pred, *_ = model(images, actions, prev_emb=prev_emb, action_diff=True)
        latest = pred[:, -1:]
        predictions.append(latest)
        context_chain.append(latest.detach() if detach_prev else latest)
        prev_emb = torch.cat(context_chain, dim=1)
    return torch.cat(predictions, dim=1), target


def multistep_loss(model, frames, deltas, num_context, n_future,
                   detach_prev=False, cache_encoder=True):
    predicted, target = multistep_predict(
        model, frames, deltas, num_context, n_future, detach_prev, cache_encoder)
    per_step = (predicted.float() - target).square().mean(dim=(0, 2, 3))
    return per_step.mean(), per_step


@torch.no_grad()
def validate(model, loader, device, num_context, n_future, limit=0,
             cache_encoder=True):
    model.eval()
    total, steps, count = 0.0, None, 0
    for index, (frames, actions, *_rest) in enumerate(loader):
        if limit and index >= limit:
            break
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, per_step = multistep_loss(
                model, frames, actions, num_context, n_future,
                cache_encoder=cache_encoder)
        total += float(loss)
        contribution = per_step.detach().float().cpu().numpy()
        steps = contribution if steps is None else steps + contribution
        count += 1
    model.train()
    return total / max(count, 1), (steps / max(count, 1)).tolist()


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
                  num_context=args.num_context, seed=args.seed)
    val_set = demos.DexJoCoDemoDataset(
        args.frames, train=False,
        windows_per_episode=args.val_windows_per_episode, **common)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False, persistent_workers=args.workers > 0)

    model = build_model(args.device, compile_attention=args.compile_attention)
    load_weights(model, args.resume)
    model.forward_kp = lambda *_a, **_k: (None, torch.zeros((), device=args.device))
    for parameter in model.image_embedder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.kp_layer.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]

    if args.eval_only:
        value, per_step = validate(model, val_loader, args.device,
                                   args.num_context, args.n_future,
                                   args.val_batches, args.cache_encoder)
        payload = {"eval_only": True, "tag": args.eval_tag or str(args.resume),
                   "checkpoint": str(args.resume), "n_future": args.n_future,
                   "val_loss": value, "val_loss_per_step": per_step,
                   "val_windows": len(val_set)}
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

    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.max_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=max(0.05, 2.0 / total_steps),
        final_div_factor=100.0, cycle_momentum=False)

    model.train()
    baseline, baseline_steps = validate(model, val_loader, args.device,
                                        args.num_context, args.n_future,
                                        args.val_batches, args.cache_encoder)
    print(f"[val] resumed-checkpoint baseline multistep(N={args.n_future}) "
          f"loss {baseline:.5f} per-step {['%.4f' % v for v in baseline_steps]}",
          flush=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps({"step": 0, "val_loss": baseline,
                                 "val_loss_per_step": baseline_steps,
                                 "baseline": True, "n_future": args.n_future,
                                 "resume": str(args.resume)}) + "\n")

    best = baseline
    step = 0
    micro = 0
    start_time = time.time()
    stop = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if stop:
            break
        for frames, actions, *_rest in train_loader:
            frames = frames.to(args.device, non_blocking=True)
            actions = actions.to(args.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, per_step = multistep_loss(
                    model, frames, actions, args.num_context, args.n_future,
                    args.detach_prev, args.cache_encoder)
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
                detail = " ".join(f"{v:.4f}" for v in
                                  per_step.detach().float().cpu().numpy())
                print(f"[train] epoch {epoch} step {step}/{total_steps} "
                      f"loss {float(loss):.5f} [{detail}] "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {rate:.3f} it/s "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
                      flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(
                        {"step": step, "epoch": epoch, "train_loss": float(loss),
                         "lr": scheduler.get_last_lr()[0],
                         "n_future": args.n_future}) + "\n")

            if step % args.eval_freq == 0 or step == total_steps:
                value, value_steps = validate(
                    model, val_loader, args.device, args.num_context,
                    args.n_future, args.val_batches, args.cache_encoder)
                print(f"[val] step {step} multistep loss {value:.5f} "
                      f"(baseline {baseline:.5f}) per-step "
                      f"{['%.4f' % v for v in value_steps]}", flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(
                        {"step": step, "epoch": epoch, "val_loss": value,
                         "val_loss_per_step": value_steps,
                         "n_future": args.n_future}) + "\n")
                payload = {"model": model.state_dict(),
                           "args": {key: str(value_) if isinstance(value_, Path)
                                    else value_ for key, value_ in vars(args).items()},
                           "step": step, "epoch": epoch, "val_loss": value,
                           "baseline_val_loss": baseline}
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
    print(f"[done] best multistep val {best:.5f} baseline {baseline:.5f} "
          f"({elapsed:.0f}s)", flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        {"seed": args.seed, "n_future": args.n_future,
         "resume": str(args.resume),
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
