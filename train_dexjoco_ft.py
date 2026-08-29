"""Fine-tune the public DexWM predictor on DexJoCo demonstrations (single GPU).

This is train_wm.py's objective on one GPU without the submitit / FSDP / wandb
machinery: the model, the loss (MSE between the predicted next-frame latent and
the frozen DINOv2 latent of the actual next frame) and the checkpoint layout are
unchanged, so a checkpoint written here loads into
`scripts/dexwm/score_branch_candidates.load_model` with `strict=True`.

What is frozen: the DINOv2 encoder (already frozen upstream) and the heatmap
keypoint head (DexJoCo demos carry no 2-D keypoint annotation, so the auxiliary
kp loss is disabled rather than trained on fabricated targets; every DexWM
evaluation in this project stubs `forward_kp` anyway).

Usage:
    python train_dexjoco_ft.py --frames /root/dexwm_ft/frames \
        --resume /root/ckpt_local/dexwm/robocasa_random_finetune.pth.tar \
        --out /root/ckpt_local/dexwm/finetune/seed0 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def load_dataset_module():
    spec = importlib.util.spec_from_file_location(
        "dexjoco_demos", REPO / "datasets" / "dexjoco_demos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model(device, dtype=torch.float32, is_eval=False,
                gradient_checkpointing=True, compile_attention=True,
                n_future=0):
    original_hub_load = torch.hub.load

    def local_dinov2(_repo, model, *args, **kwargs):
        return original_hub_load(
            "/root/.cache/torch/hub/facebookresearch_dinov2_main",
            model, source="local", pretrained=False,
        )

    torch.hub.load = local_dinov2
    try:
        import models.model as dexwm_model
        from models.model import DexWM

        if compile_attention:
            # Without torch.compile, flex_attention falls back to the dense
            # eager kernel, which materialises a (B, H, 4032, 4032) score matrix
            # and OOMs in the backward pass.  The upstream config compiles the
            # whole model (do_compile: True); compiling just the attention op is
            # the documented lighter-weight equivalent and keeps the module
            # graph (and therefore the checkpoint keys) untouched.
            from torch.nn.attention.flex_attention import flex_attention
            dexwm_model.flex_attention = torch.compile(flex_attention, dynamic=False)
        model = DexWM(
            backbone_name="dinov2", num_patches=448, patch_size=14,
            hidden_dim=1024, action_dim=132, depth=32, num_heads=16,
            mlp_ratio=2.0, num_context=8, n_future=n_future, is_eval=is_eval,
            emb_loss_fn=nn.MSELoss(reduction="mean"),
            use_gradient_checkpointing=gradient_checkpointing,
            use_fsdp=False, num_keypoints=12,
        )
    finally:
        torch.hub.load = original_hub_load
    return model.to(device=device, dtype=dtype)


def load_weights(model, checkpoint: Path, strict=True):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    weights = {key.replace("_orig_mod.", ""): value
               for key, value in state["model"].items()}
    # Handle pos_embed shape mismatch (seqext extends sequence length)
    if "pos_embed" in weights and weights["pos_embed"].shape != model.pos_embed.shape:
        old_pe = weights.pop("pos_embed")
        T_old = old_pe.shape[0]
        T_new = model.pos_embed.shape[0]
        print(f"[load] pos_embed {T_old}→{T_new}: first {T_old} from ckpt, "
              f"rest initialised", flush=True)
        with torch.no_grad():
            model.pos_embed[:T_old].copy_(old_pe)
    missing, unexpected = model.load_state_dict(weights, strict=strict)
    if missing:
        print(f"[load] missing keys (new modules): {missing}", flush=True)
    if unexpected:
        print(f"[load] unexpected keys: {unexpected}", flush=True)
    del state, weights


@torch.no_grad()
def validate(model, loader, device, limit=0):
    model.eval()
    total, count = 0.0, 0
    for i, (frames, actions, *_rest) in enumerate(loader):
        if limit and i >= limit:
            break
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, emb_loss, _ = model(frames, actions, action_diff=True)
        total += float(emb_loss)
        count += 1
    model.train()
    return total / max(count, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=Path("/root/dexwm_ft/frames"))
    parser.add_argument("--resume", type=Path,
                        default=Path("/root/ckpt_local/dexwm/robocasa_random_finetune.pth.tar"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--world-to-camera", type=Path,
                        default=Path("/root/dexwm_branch/data/branch_seed00_step00090.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--windows-per-episode", type=int, default=40)
    parser.add_argument("--val-windows-per-episode", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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

    train_set = demos.DexJoCoDemoDataset(
        args.frames, world_to_camera, train=True, seed=args.seed,
        windows_per_episode=args.windows_per_episode)
    val_set = demos.DexJoCoDemoDataset(
        args.frames, world_to_camera, train=False, seed=args.seed,
        windows_per_episode=args.val_windows_per_episode)
    print(f"[data] train episodes {len(train_set.usable)} "
          f"windows/epoch {len(train_set)} (pool {train_set.n_windows_total}) | "
          f"val episodes {len(val_set.usable)} windows {len(val_set)} "
          f"(pool {val_set.n_windows_total})", flush=True)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
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
    print(f"[model] trainable {sum(p.numel() for p in trainable)/1e6:.1f}M / "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = (args.max_steps if args.max_steps
                   else steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=max(0.05, 2.0 / total_steps),
        final_div_factor=100.0, cycle_momentum=False)

    model.train()
    baseline = validate(model, val_loader, args.device, args.val_batches)
    print(f"[val] frozen-checkpoint baseline emb_loss {baseline:.5f}", flush=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps({"step": 0, "val_loss": baseline,
                                 "baseline": True, "seed": args.seed}) + "\n")

    best = baseline
    step = 0
    start_time = time.time()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for frames, actions, *_rest in train_loader:
            frames = frames.to(args.device, non_blocking=True)
            actions = actions.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, _, _, emb_loss, _ = model(frames, actions, action_diff=True)
            emb_loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 10 == 0:
                rate = step / (time.time() - start_time)
                print(f"[train] epoch {epoch} step {step}/{total_steps} "
                      f"loss {float(emb_loss):.5f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"{rate:.2f} it/s "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
                      flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(
                        {"step": step, "epoch": epoch, "train_loss": float(emb_loss),
                         "lr": scheduler.get_last_lr()[0], "seed": args.seed}) + "\n")

            if step % args.eval_freq == 0 or step == total_steps:
                value = validate(model, val_loader, args.device, args.val_batches)
                print(f"[val] step {step} emb_loss {value:.5f} "
                      f"(baseline {baseline:.5f})", flush=True)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(
                        {"step": step, "epoch": epoch, "val_loss": value,
                         "seed": args.seed}) + "\n")
                payload = {"model": model.state_dict(),
                           "args": vars(args) | {"out": str(args.out),
                                                 "frames": str(args.frames),
                                                 "resume": str(args.resume),
                                                 "world_to_camera": str(args.world_to_camera)},
                           "step": step, "epoch": epoch, "val_loss": value,
                           "baseline_val_loss": baseline}
                torch.save(payload, args.out / "last.pth.tar")
                if value < best:
                    best = value
                    torch.save(payload, args.out / "best.pth.tar")
                    print(f"[val] new best {best:.5f}", flush=True)
                del payload

            if step >= total_steps:
                stop = True
                break

    print(f"[done] best val {best:.5f} baseline {baseline:.5f} "
          f"({time.time() - start_time:.0f}s)", flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        {"seed": args.seed, "baseline_val_loss": baseline, "best_val_loss": best,
         "steps": step, "total_steps": total_steps,
         "train_episodes": len(train_set.usable), "val_episodes": len(val_set.usable),
         "windows_per_epoch": len(train_set), "window_pool": train_set.n_windows_total,
         "lr": args.lr, "batch": args.batch, "epochs": args.epochs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
