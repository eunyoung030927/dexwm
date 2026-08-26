"""Reference loss level of DexWM on the AUTHORS' own RoboCasa Random data.

Our DexJoCo fine-tuning curves flatten early (1-step val 0.199 -> 0.133,
multistep N=3 0.157 -> 0.151, N=6 0.172 -> 0.167, and a whole extra epoch at
N=6 moved 0.170 -> 0.1697).  Flat can mean two very different things: the
predictor is under-trained on our data, or it is sitting on its natural loss
floor.  The way to tell them apart is to run the *same* objective on the data
the public checkpoint was fine-tuned on (facebook/dexwm "RoboCasa Random").  If
the public checkpoint is also flat and also lands around 0.13-0.17 there, our
curve is the floor; if it drops sharply, our data is the problem.

This script only ever measures - it does not train.  `--eval-matrix` is a
comma-separated list of `<source>:<kind>` protocols; every one of them reports
the same latent MSE our own scripts report.

Protocols
  stride windows  (--sampling stride) num_context consecutive frames + N future
                  frames, i.e. exactly the window train_dexjoco_{ft,ms}.py use.
  full_seq windows (--sampling full_seq) num_context frames drawn at random
                  from the clip plus its last frame - the window
                  configs/robocasa_random_multistep.yaml actually trains on.
  --upstream-rollout  reproduces train_multistep_wm.py:335-359 exactly: the
                  context is frame 0 repeated, the rollout is num_context steps
                  long and the loss is MSE over all of them.

The loss itself is identical in all cases to what our scripts report: for
1-step it is DexWM.forward's own emb_loss (MSE of the predicted next-frame
latent against the frozen DINOv2 latent, averaged over every context position),
for multistep it is train_dexjoco_ms.multistep_loss (per-step MSE averaged over
the N rollout steps, which for equal-sized steps equals upstream's single
MSE over the stacked predictions).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train_dexjoco_ft import build_model, load_weights  # noqa: E402
from train_dexjoco_ft import validate as validate_single  # noqa: E402
from train_dexjoco_ms import frozen_encoder  # noqa: E402
from train_dexjoco_ms import validate as validate_multistep  # noqa: E402


def load_ref_dataset_module():
    spec = importlib.util.spec_from_file_location(
        "robocasa_ref_windows", REPO / "datasets" / "robocasa_ref_windows.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream_multistep_loss(model, frames, deltas, num_context):
    """train_multistep_wm.py:335-359 with the encoder called once per window.

    The context is `frames[:, 0]` repeated (that is what upstream feeds), the
    actions are zero-padded on the left by num_context - 1 and the rollout runs
    for every action, so the loss covers frames 1..T against the DINOv2 latents
    of the real frames.
    """
    n_steps = deltas.shape[1]
    zeros = torch.zeros(deltas.shape[0], num_context - 1, deltas.shape[-1],
                        dtype=deltas.dtype, device=deltas.device)
    padded_actions = torch.cat([zeros, deltas], dim=1)

    latents = model.encode_image(frames)                    # (B, T + 1, P, D)
    target = latents[:, 1:].float()
    padded_latent = latents[:, 0:1].expand(-1, num_context + n_steps, -1, -1)
    padded_frames = frames[:, 0:1].expand(-1, num_context + n_steps, -1, -1, -1)

    predictions, prev_emb = [], None
    for step in range(n_steps):
        window = padded_latent[:, step:step + num_context + 1]
        with frozen_encoder(model, window):
            pred, *_ = model(padded_frames[:, step:step + num_context + 1],
                             padded_actions[:, step:step + num_context],
                             prev_emb=prev_emb, action_diff=True)
        predictions.append(pred[:, -1:])
        prev_emb = torch.cat(predictions, dim=1)
    predicted = torch.cat(predictions, dim=1)
    per_step = (predicted.float() - target).square().mean(dim=(0, 2, 3))
    return per_step.mean(), per_step


@torch.no_grad()
def validate_upstream(model, loader, device, num_context, limit=0):
    model.eval()
    total, steps, count = 0.0, None, 0
    for index, (frames, actions, *_rest) in enumerate(loader):
        if limit and index >= limit:
            break
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, per_step = upstream_multistep_loss(
                model, frames, actions, num_context)
        total += float(loss)
        contribution = per_step.detach().float().cpu().numpy()
        steps = contribution if steps is None else steps + contribution
        count += 1
    model.train()
    return total / max(count, 1), (steps / max(count, 1)).tolist()


def make_loader(module, args, n_future, train, sampling, batch, windows,
                stride=None, source="robocasa"):
    """Val/train loader over either the RoboCasa reference windows or ours.

    `source="dexjoco"` returns the very windows train_dexjoco_{ft,ms}.py score,
    so one eval run can put both datasets in the same table under the same
    checkpoint and the same code path.
    """
    if source == "dexjoco":
        demos = load_dexjoco_module()
        with np.load(args.dexjoco_extrinsic, allow_pickle=False) as reference:
            world_to_camera = np.asarray(reference["world_to_camera"])
        dataset = demos.DexJoCoDemoDataset(
            args.dexjoco_frames, world_to_camera=world_to_camera, train=train,
            windows_per_episode=windows, n_future=n_future,
            num_context=args.num_context, seed=args.seed)
    else:
        dataset = module.RobocasaRefDataset(
            args.data, num_context=args.num_context, n_future=n_future,
            train=train, context_stride=stride or args.context_stride,
            sampling=sampling, windows_per_episode=windows, seed=args.seed,
            split_file=args.split_file)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=train, num_workers=args.workers,
        pin_memory=True, drop_last=train, persistent_workers=args.workers > 0)
    return dataset, loader


def load_dexjoco_module():
    spec = importlib.util.spec_from_file_location(
        "dexjoco_demos", REPO / "datasets" / "dexjoco_demos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def validate_persistence(model, loader, device, num_context, n_future,
                         single=False, limit=0):
    """Null model: predict "nothing changes" and report the same MSE.

    For the 1-step protocol this is MSE(z_t, z_{t+1}) averaged over every
    context position - the exact quantity DexWM.forward's emb_loss is measured
    against.  For the multistep protocol it is the per-step MSE of the last
    context latent against each of the N future latents.  It is the scale of
    the data, so `model loss / persistence loss` says how much of the movement
    the predictor actually explains, which is what makes losses measured on two
    different datasets comparable at all.
    """
    total, steps, count = 0.0, None, 0
    for index, (frames, *_rest) in enumerate(loader):
        if limit and index >= limit:
            break
        frames = frames.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            latents = model.encode_image(frames).float()
        if single:
            per_step = (latents[:, 1:] - latents[:, :-1]).square().mean(dim=(0, 2, 3))
        else:
            hold = latents[:, num_context - 1:num_context]
            per_step = (latents[:, num_context:] - hold).square().mean(dim=(0, 2, 3))
        total += float(per_step.mean())
        contribution = per_step.detach().float().cpu().numpy()
        steps = contribution if steps is None else steps + contribution
        count += 1
    return total / max(count, 1), (steps / max(count, 1)).tolist()


def run_eval(args, module, model):
    results = []
    for spec in args.eval_matrix.split(","):
        spec = spec.strip()
        if not spec:
            continue
        source_spec, kind = spec.split(":")
        # source: "stride" / "strideK" (RoboCasa consecutive frames at stride K)
        #         "full_seq"          (RoboCasa upstream random-frame window)
        #         "dexjoco"           (our own DexJoCo demo windows)
        stride, sampling, source = args.context_stride, source_spec, "robocasa"
        if source_spec == "dexjoco":
            sampling, source = "stride", "dexjoco"
        elif source_spec.startswith("stride"):
            sampling = "stride"
            if len(source_spec) > len("stride"):
                stride = int(source_spec[len("stride"):])
        started = time.time()
        if kind == "upstream":
            n_future = args.num_context
            dataset, loader = make_loader(
                module, args, 1, False, sampling, args.batch,
                args.val_windows_per_episode, stride, source)
            value, per_step = validate_upstream(
                model, loader, args.device, args.num_context, args.val_batches)
        elif kind == "1step":
            n_future = 1
            dataset, loader = make_loader(
                module, args, 1, False, sampling, args.batch,
                args.val_windows_per_episode, stride, source)
            value = validate_single(model, loader, args.device, args.val_batches)
            per_step = None
        elif kind.startswith("null"):
            n_future = int(kind[4:] or 1)
            single = kind == "null1"
            batch = args.batch if n_future <= 1 else args.ms_batch
            dataset, loader = make_loader(
                module, args, n_future, False, sampling, batch,
                args.val_windows_per_episode, stride, source)
            value, per_step = validate_persistence(
                model, loader, args.device, args.num_context, n_future,
                single, args.val_batches)
        else:
            n_future = int(kind)
            dataset, loader = make_loader(
                module, args, n_future, False, sampling, args.ms_batch,
                args.val_windows_per_episode, stride, source)
            value, per_step = validate_multistep(
                model, loader, args.device, args.num_context, n_future,
                args.val_batches)
        row = {"protocol": spec, "source": source, "sampling": sampling,
               "stride": stride if source != "dexjoco" else 5, "kind": kind,
               "n_future": n_future, "loss": value, "loss_per_step": per_step,
               "val_clips": len(dataset.usable), "val_windows": len(dataset),
               "seconds": round(time.time() - started, 1)}
        print("[eval] " + json.dumps(row), flush=True)
        results.append(row)
    payload = {"checkpoint": str(args.resume), "data": str(args.data),
               "context_stride": args.context_stride,
               "num_context": args.num_context, "results": results}
    (args.out / "eval_matrix.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path("/root/dexwm_data/robocasa_random_data"))
    parser.add_argument("--split-file", type=Path,
                        default=Path("/root/dexwm_data/split_indices_robocasa_ref.json"))
    parser.add_argument("--resume", type=Path,
                        default=Path("/root/ckpt_local/dexwm/robocasa_random_finetune.pth.tar"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-matrix",
                        default="stride:1step,stride:3,stride:6,full_seq:1step,full_seq:upstream")
    parser.add_argument("--dexjoco-frames", type=Path,
                        default=Path("/root/dexwm_ft/frames"))
    parser.add_argument("--dexjoco-extrinsic", type=Path,
                        default=Path("/root/dexwm_branch/data/branch_seed00_step00090.npz"))
    parser.add_argument("--context-stride", type=int, default=1)
    parser.add_argument("--num-context", type=int, default=8)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--ms-batch", type=int, default=4)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--val-windows-per-episode", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile-attention", dest="compile_attention",
                        action="store_false")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    module = load_ref_dataset_module()
    model = build_model(args.device, compile_attention=args.compile_attention)
    load_weights(model, args.resume)
    model.forward_kp = lambda *_a, **_k: (None, torch.zeros((), device=args.device))
    for parameter in model.image_embedder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.kp_layer.parameters():
        parameter.requires_grad_(False)

    return run_eval(args, module, model)


if __name__ == "__main__":
    raise SystemExit(main())
