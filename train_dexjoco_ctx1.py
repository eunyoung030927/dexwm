"""Single-context (num_context=1) DexWM fine-tuning on ONE DexJoCo task.

Why this script exists
----------------------
`train_dexjoco_ft.py` trains the same single-step objective at
`num_context=8` (`:75`), which is the RoboCasa config value, not an
architectural constraint: `DexWM.__init__` takes `num_context` as a plain
argument (`models/model.py:310`, default 4) and sizes `pos_embed`
(`:393`, `total_frames = num_context + max(n_future, 1)`) and the two
flex-attention block masks (`:380-387`) from it, while `forward`
(`:498`) slices `pos_embed[:T_seq]` and is therefore length-agnostic.

The DynaGuide comparison (PLAN/plan/260902_comparisons_impl_plan.md §3.1)
needs a predictor that consumes the CURRENT observation only, because at
rollout time we have one live frame, not an 8-frame ring buffer.  Training
at `num_context=1` removes that requirement at the source and, per the §3.1
table, simultaneously removes the 8x DINOv2 encoding cost, so the 10-candidate
guidance batch fits in VRAM.

What is different from `train_dexjoco_ft.py`
--------------------------------------------
Only the things §3.1 requires; the model, the loss (MSE between the predicted
next-frame latent and the frozen DINOv2 latent of the actual next frame), the
dataset and the checkpoint layout are the same objects, imported from
`train_dexjoco_ft` / `train_dexjoco_ms` rather than reimplemented:

  * `--num-context` (default **1**) is threaded into `build_model` and into
    `DexJoCoDemoDataset`;
  * **fp32 forward path.**  `train_dexjoco_ft.py:215` wraps the step in
    `torch.amp.autocast(bfloat16)`.  The comparison's guidance path takes
    `autograd.grad` through this predictor and the auto-memory record
    ("bf16 상쇄 -> counterfactual은 fp32 필수, cos_pos_neg -0.99") says bf16
    cancels that gradient, so training and guidance both run fp32 here.
    `--amp bf16` restores the old behaviour for a speed comparison only.
  * **loss-plateau early stopping** ("로스 떨어질 때까지"): patience on the
    held-out latent MSE instead of a fixed step budget.
  * **decoder / keypoint-head losses are OFF by default.**  The comparison
    scores a latent distance, so no pixel decoder and no heatmap head are
    needed; `--kp-labels` can still switch the auxiliary head on for a
    diagnostic run (same code path as `train_dexjoco_ms.py`).
  * **no hardcoded `/root/...` paths.**  Every path is an argument, with an
    env-var default (`DEXWM_TASK_DATA_ROOT`, `DEXWM_W2C`, `DEXWM_INIT_CKPT`,
    `DEXWM_OUT_DIR`, `VLS_CODE_ROOT`, `DINOV2_HUB_DIR`).
  * curve + held-out latent MSE are written as BOTH `curve.jsonl` and
    `curve.csv` for the §7.1 appendix.

`--init-ckpt` is optional: with it the run starts from an existing checkpoint
(the 8-context one is fine - `pos_embed` is truncated to the shorter sequence,
see `_load_init_weights`), without it from the DINOv2-frozen random init.  The
file's existence is only *checked*, never required.

Checkpoint layout is unchanged, so what this writes loads back through
`scripts/dexwm/score_branch_candidates.load_model` - with `num_context=1`
passed there instead of 8.

NOTHING IS RUN BY IMPORTING THIS FILE.  Training is an approval item
(plan §6.3 (b)); see `code/comparisons/dynaguide/train/README.md`.

Usage (illustrative only - do not run without approval):
    VLS_CODE_ROOT=/workspace/vls/code python train_dexjoco_ctx1.py \
        --task pick_bucket \
        --task-data-root /root/dexwm_ft/frames \
        --world-to-camera /root/dexwm_branch/data/branch_seed00_step00090.npz \
        --out-dir /root/ckpt_local/dexwm/ctx1/pick_bucket \
        --init-ckpt /root/ckpt_local/dexwm/robocasa_random_finetune.pth.tar
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from train_dexjoco_ft import load_dataset_module  # noqa: E402
from train_dexjoco_ms import retarget_kp_head  # noqa: E402


def _env_path(name: str):
    value = os.environ.get(name)
    return Path(value) if value else None


DEFAULT_DINOV2_HUB = "/root/.cache/torch/hub/facebookresearch_dinov2_main"


def build_ctx_model(device, num_context: int, dtype=torch.float32,
                    compile_attention: bool = True,
                    gradient_checkpointing: bool = True,
                    dinov2_hub_dir: str | None = None):
    """`train_dexjoco_ft.build_model` with `num_context` opened up.

    Every other hyperparameter is the one that script pins (dinov2 backbone,
    448 patches, 132-D action, depth 32, 16 heads, mlp 2.0, 12 keypoints), so
    a checkpoint written here keeps the same key set and still loads through
    `score_branch_candidates.load_model` - with the matching `num_context`.

    `num_context` reaches `pos_embed` (models/model.py:393) and the two
    flex-attention block masks (:380-387) through `total_frames`; `forward`
    slices `pos_embed[:T_seq]` and is length-agnostic.

    The DINOv2 hub load is redirected to a local clone exactly as upstream
    does, because the container has no network at train time.
    """
    import torch.nn as nn

    original_hub_load = torch.hub.load
    hub_dir = dinov2_hub_dir or DEFAULT_DINOV2_HUB

    def local_dinov2(_repo, name, *_args, **_kwargs):
        return original_hub_load(hub_dir, name, source="local", pretrained=False)

    torch.hub.load = local_dinov2
    try:
        import models.model as dexwm_model
        from models.model import DexWM

        if compile_attention:
            # Without torch.compile, flex_attention falls back to the dense
            # eager kernel and materialises the full score matrix.  At
            # num_context=1 the window is 2*448 tokens instead of 9*448, so
            # this is far cheaper than at 8 context - but compiling still
            # helps and keeps the module graph (and checkpoint keys) untouched.
            from torch.nn.attention.flex_attention import flex_attention
            dexwm_model.flex_attention = torch.compile(flex_attention,
                                                       dynamic=False)
        model = DexWM(
            backbone_name="dinov2", num_patches=448, patch_size=14,
            hidden_dim=1024, action_dim=132, depth=32, num_heads=16,
            mlp_ratio=2.0, num_context=num_context, n_future=0, is_eval=False,
            emb_loss_fn=nn.MSELoss(reduction="mean"),
            use_gradient_checkpointing=gradient_checkpointing,
            use_fsdp=False, num_keypoints=12,
        )
    finally:
        torch.hub.load = original_hub_load
    return model.to(device=device, dtype=dtype)


# --------------------------------------------------------------------- weights
def _load_init_weights(model, checkpoint: Path) -> dict:
    """Load `checkpoint` into a possibly SHORTER-sequence model.

    Same key cleaning as `train_dexjoco_ft.load_weights` (strip the
    `_orig_mod.` prefix torch.compile adds), with one addition it does not
    handle: going from `num_context=8` (pos_embed (9, P, D)) to
    `num_context=1` (pos_embed (2, P, D)) SHRINKS the sequence axis, so the
    upstream branch (`model.pos_embed[:T_old].copy_(old_pe)`) would index out
    of range.  Here the first `T_new` slots are copied and the rest dropped:
    slot 0 is the context frame and slot -1 the prediction slot in both
    layouts, so the surviving slots keep their meaning.

    Everything else must match exactly, hence `strict=True` on the remainder.
    Returns a small report for the run summary.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    weights = {key.replace("_orig_mod.", ""): value
               for key, value in state["model"].items()}
    report: dict[str, object] = {"checkpoint": str(checkpoint)}

    if "pos_embed" in weights and weights["pos_embed"].shape != model.pos_embed.shape:
        old = weights.pop("pos_embed")
        keep = min(old.shape[0], model.pos_embed.shape[0])
        with torch.no_grad():
            model.pos_embed[:keep].copy_(old[:keep].to(model.pos_embed.dtype))
        report["pos_embed"] = f"{tuple(old.shape)} -> {tuple(model.pos_embed.shape)} (kept {keep})"
        print(f"[load] pos_embed {tuple(old.shape)} -> "
              f"{tuple(model.pos_embed.shape)}: kept first {keep} slots", flush=True)

    missing, unexpected = model.load_state_dict(weights, strict=False)
    # `action_chunk_encoder` is always built but is unused here, so a
    # pre-chunk checkpoint legitimately lacks it; anything else missing is a
    # real mismatch and must not pass silently.
    hard_missing = [k for k in missing
                    if not k.startswith("action_chunk_encoder")
                    and not k.startswith("pos_embed")]
    if hard_missing:
        raise RuntimeError(f"init checkpoint is missing required keys: {hard_missing}")
    if missing:
        report["missing_ok"] = missing
        print(f"[load] missing (unused modules / handled above): {missing}", flush=True)
    if unexpected:
        report["unexpected"] = unexpected
        print(f"[load] unexpected keys ignored: {unexpected}", flush=True)
    del state, weights
    return report


# ---------------------------------------------------------------------- losses
def _precision(amp_dtype):
    """fp32 (the plan's requirement) unless an autocast dtype is requested.

    `train_dexjoco_ft.py:215` wraps every step in
    `torch.amp.autocast(bfloat16)`.  Plan §3.1 requires the 1-context
    predictor's forward to be fp32, because the DynaGuide comparison takes
    `autograd.grad` through it and the recorded bf16 measurement
    (`cos_pos_neg` -0.99) shows bf16 cancels that gradient.  `amp_dtype=None`
    therefore disables autocast outright rather than merely not enabling it,
    so an outer autocast (if any) cannot leak in.
    """
    if amp_dtype is None:
        return torch.amp.autocast("cuda", enabled=False)
    return torch.amp.autocast("cuda", dtype=amp_dtype)


def singlestep_loss(model, frames, deltas, amp_dtype=None, heatmaps=None,
                    valid_kp=None, kp_weight=0.0):
    """One forward pass; DexWM's own `emb_loss` is the objective.

    Same call as `train_dexjoco_ft.py:216` (`action_diff=True`, the 5-tuple
    return of `models/model.py:569`) under `_precision`.

    `heatmaps` adds the OPTIONAL auxiliary heatmap term - not needed by the
    comparison (which scores a latent distance and needs no decoder and no
    keypoint head), kept behind `--kp-labels` so a diagnostic run can check
    whether the 1-context predictor still supports the 2-D readout.  Its shape
    follows `train_dexjoco_ms.multistep_loss`: masked mean over every entry, so
    an invalid channel contributes zero rather than shrinking the denominator.
    """
    with _precision(amp_dtype):
        _pred, _goal, maps, emb_loss, _kp_loss = model(frames, deltas,
                                                       action_diff=True)
    stats = {"emb": emb_loss.detach()}
    if heatmaps is None:
        return emb_loss, stats
    squared = (maps.float() - heatmaps).square().mean(dim=(-2, -1))
    masked = squared * valid_kp
    kp_loss = masked.mean()
    fraction = valid_kp.mean().clamp_min(1e-6)
    stats.update({"kp": kp_loss.detach(),
                  "kp_masked": (kp_loss / fraction).detach(),
                  "kp_valid": fraction.detach()})
    return emb_loss + kp_weight * kp_loss, stats


@torch.no_grad()
def validate(model, loader, device, limit=0, amp_dtype=None, with_kp=False,
             kp_weight=0.0):
    """Held-out latent MSE (plus the kp term when it is trained)."""
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for index, batch in enumerate(loader):
        if limit and index >= limit:
            break
        frames = batch[0].to(device, non_blocking=True)
        actions = batch[1].to(device, non_blocking=True)
        heatmaps = batch[3].to(device, non_blocking=True) if with_kp else None
        valid_kp = batch[4].to(device, non_blocking=True) if with_kp else None
        _loss, stats = singlestep_loss(model, frames, actions, amp_dtype,
                                       heatmaps, valid_kp, kp_weight)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    model.train()
    count = max(count, 1)
    return {key: value / count for key, value in totals.items()}


# ---------------------------------------------------------------------- logging
class CurveLog:
    """Append-only training curve, written as JSONL and CSV (plan §7.1)."""

    FIELDS = ("step", "epoch", "kind", "train_loss", "train_emb", "val_emb",
              "val_kp", "val_kp_masked", "lr", "wall_s")

    def __init__(self, out_dir: Path):
        self.jsonl = out_dir / "curve.jsonl"
        self.csv = out_dir / "curve.csv"
        if not self.csv.exists():
            with self.csv.open("w", newline="") as handle:
                csv.writer(handle).writerow(self.FIELDS)

    def write(self, record: dict) -> None:
        with self.jsonl.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        with self.csv.open("a", newline="") as handle:
            csv.writer(handle).writerow([record.get(f, "") for f in self.FIELDS])


# ------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="num_context=1 single-task DexWM fine-tune (DynaGuide baseline)")

    # --- data / paths (no hardcoded /root, env-var defaults) -----------------
    parser.add_argument("--task", required=True,
                        help="task name, used only for logging/summary "
                             "(pick_bucket | click_mouse | fold_glasses | water_plant)")
    parser.add_argument("--task-data-root", type=Path,
                        default=_env_path("DEXWM_TASK_DATA_ROOT"),
                        help="directory of exported demo frames for ONE task "
                             "(manifest.json + episode_XXX.npz), i.e. the "
                             "DexJoCoDemoDataset root_folder. "
                             "Env: DEXWM_TASK_DATA_ROOT")
    parser.add_argument("--world-to-camera", type=Path,
                        default=_env_path("DEXWM_W2C"),
                        help="npz holding a 4x4 'world_to_camera' for this "
                             "task's camera (the action adapter projects hand "
                             "keypoints through it). Env: DEXWM_W2C")
    parser.add_argument("--out-dir", type=Path, default=_env_path("DEXWM_OUT_DIR"),
                        help="checkpoints + curve.{jsonl,csv} + summary.json. "
                             "Env: DEXWM_OUT_DIR")
    parser.add_argument("--init-ckpt", type=Path, default=_env_path("DEXWM_INIT_CKPT"),
                        help="optional checkpoint to initialise from (an "
                             "8-context one is fine: pos_embed is truncated). "
                             "Omit for a DINOv2-frozen random init. "
                             "Env: DEXWM_INIT_CKPT")
    parser.add_argument("--vls-code-root", type=Path, default=_env_path("VLS_CODE_ROOT"),
                        help="the VLS repo whose core.dexwm_action_adapter the "
                             "dataset imports; sets VLS_CODE_ROOT for the "
                             "loader. Env: VLS_CODE_ROOT (loader default is "
                             "/workspace/vls/code_dexwm, which does not exist "
                             "on this host - pass /workspace/vls/code)")
    parser.add_argument("--dinov2-hub-dir", type=Path, default=_env_path("DINOV2_HUB_DIR"),
                        help="local torch.hub DINOv2 clone; only recorded in "
                             "the summary (build_model resolves it itself). "
                             "Env: DINOV2_HUB_DIR")

    # --- the point of this script -------------------------------------------
    parser.add_argument("--num-context", type=int, default=1,
                        help="context frames the predictor conditions on "
                             "(plan §3.1: 1 = current observation only)")

    # --- optimisation --------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8,
                        help="1-context windows are ~1/8 the encoder work of "
                             "the 8-context ones, so this can exceed the "
                             "8-context script's default of 6")
    parser.add_argument("--accum", type=int, default=1,
                        help="gradient accumulation (effective batch = batch*accum)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="upper bound only; early stopping decides")
    parser.add_argument("--max-steps", type=int, default=6000,
                        help="hard cap on optimiser steps (0 = epochs only)")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-stride", type=int, default=5,
                        help="control steps between DexWM frames; the exported "
                             "per-control-step demos use 5")
    parser.add_argument("--windows-per-episode", type=int, default=40)
    parser.add_argument("--val-windows-per-episode", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=10,
                        help="every val_every-th episode is held out (dataset split)")
    parser.add_argument("--workers", type=int, default=8)

    # --- "로스 떨어질 때까지" -------------------------------------------------
    parser.add_argument("--eval-every", type=int, default=200,
                        help="optimiser steps between held-out evaluations")
    parser.add_argument("--val-batches", type=int, default=0,
                        help="0 = the whole validation split")
    parser.add_argument("--patience", type=int, default=5,
                        help="stop after this many consecutive evaluations "
                             "without a new best held-out latent MSE "
                             "(0 disables early stopping)")
    parser.add_argument("--min-delta", type=float, default=1e-5,
                        help="improvement below this does not reset patience")

    # --- precision (plan §3.1: fp32) ----------------------------------------
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off",
                        help="'off' (default) = fp32 forward/backward, which is "
                             "what the guidance path needs; bf16 is offered "
                             "only for a speed comparison")

    # --- optional auxiliary heads (decoder/kp NOT needed by the comparison) --
    parser.add_argument("--kp-labels", type=Path, default=None,
                        help="OPTIONAL diagnostic: directory of per-episode 2-D "
                             "keypoint npz labels; enables the auxiliary "
                             "heatmap loss. Leave unset - the DynaGuide "
                             "comparison scores a latent distance and needs no "
                             "decoder and no keypoint head")
    parser.add_argument("--kp-weight", type=float, default=100.0)
    parser.add_argument("--kp-sigma", type=float, default=2.0)
    parser.add_argument("--kp-head-lr", type=float, default=3e-4)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile-attention", dest="compile_attention",
                        action="store_false",
                        help="without compiled flex_attention the dense eager "
                             "kernel materialises the full score matrix; at "
                             "num_context=1 the window is short enough that "
                             "this may be affordable")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve paths, build the datasets, print the plan "
                             "and exit WITHOUT touching the GPU or training")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    for name in ("task_data_root", "world_to_camera", "out_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required "
                         f"(or set its env var)")
    if args.vls_code_root is not None:
        # DexJoCoDemoDataset._vls_root() reads this to import
        # core.dexwm_action_adapter; its built-in default does not exist here.
        os.environ["VLS_CODE_ROOT"] = str(args.vls_code_root)
    if args.num_context < 1:
        parser.error("--num-context must be >= 1")
    if args.init_ckpt is not None and not args.init_ckpt.exists():
        print(f"[init] --init-ckpt {args.init_ckpt} does not exist; "
              f"training from a DINOv2-frozen random init instead", flush=True)
        args.init_ckpt = None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    demos = load_dataset_module()
    with np.load(args.world_to_camera, allow_pickle=False) as reference:
        world_to_camera = np.asarray(reference["world_to_camera"])

    with_kp = args.kp_labels is not None
    common = dict(world_to_camera=world_to_camera,
                  num_context=args.num_context, n_future=1,
                  context_stride=args.context_stride, val_every=args.val_every,
                  seed=args.seed, kp_root=args.kp_labels,
                  kp_sigma=args.kp_sigma)
    train_set = demos.DexJoCoDemoDataset(
        args.task_data_root, train=True,
        windows_per_episode=args.windows_per_episode, **common)
    val_set = demos.DexJoCoDemoDataset(
        args.task_data_root, train=False,
        windows_per_episode=args.val_windows_per_episode, **common)
    print(f"[data] task {args.task} num_context {args.num_context} "
          f"| train episodes {len(train_set.usable)} windows/epoch "
          f"{len(train_set)} (pool {train_set.n_windows_total}) "
          f"| val episodes {len(val_set.usable)} windows {len(val_set)}",
          flush=True)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "task": args.task,
                          "num_context": args.num_context,
                          "task_data_root": str(args.task_data_root),
                          "world_to_camera": str(args.world_to_camera),
                          "out_dir": str(args.out_dir),
                          "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
                          "amp": args.amp, "with_kp": with_kp,
                          "train_windows": len(train_set),
                          "val_windows": len(val_set)}, indent=2), flush=True)
        return 0

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False, persistent_workers=args.workers > 0)

    amp_dtype = {"off": None, "bf16": torch.bfloat16,
                 "fp16": torch.float16}[args.amp]
    model = build_ctx_model(
        args.device, args.num_context, dtype=torch.float32,
        compile_attention=args.compile_attention,
        dinov2_hub_dir=str(args.dinov2_hub_dir) if args.dinov2_hub_dir else None)

    init_report = None
    if args.init_ckpt is not None:
        init_report = _load_init_weights(model, args.init_ckpt)

    # DINOv2 stays frozen (it is frozen upstream too).
    for parameter in model.image_embedder.parameters():
        parameter.requires_grad_(False)
    # The unused chunk encoder never receives gradient here.
    for parameter in model.action_chunk_encoder.parameters():
        parameter.requires_grad_(False)

    if with_kp:
        retarget_kp_head(model, val_set.kp_channels)
        for parameter in model.kp_layer.parameters():
            parameter.requires_grad_(True)
        head = [p for p in model.kp_layer.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head}
        body = [p for p in model.parameters()
                if p.requires_grad and id(p) not in head_ids]
        groups = [{"params": body, "lr": args.lr},
                  {"params": head, "lr": args.kp_head_lr}]
        trainable = body + head
    else:
        # No decoder, no keypoint head: the comparison reads a latent distance.
        model.forward_kp = lambda *_a, **_k: (
            None, torch.zeros((), device=args.device))
        for parameter in model.kp_layer.parameters():
            parameter.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        groups = [{"params": trainable, "lr": args.lr}]

    print(f"[model] trainable {sum(p.numel() for p in trainable)/1e6:.1f}M / "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"| amp {args.amp} | kp head {'on' if with_kp else 'off'}", flush=True)

    optimizer = torch.optim.AdamW(groups, lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader) // args.accum, 1)
    total_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[group["lr"] for group in groups],
        total_steps=total_steps, pct_start=max(0.05, 2.0 / total_steps),
        final_div_factor=100.0, cycle_momentum=False)

    curve = CurveLog(args.out_dir)
    model.train()
    start_time = time.time()

    baseline = validate(model, val_loader, args.device, args.val_batches,
                        amp_dtype, with_kp, args.kp_weight)
    print(f"[val] init held-out latent MSE {baseline['emb']:.5f}", flush=True)
    curve.write({"step": 0, "epoch": 0, "kind": "baseline",
                 "val_emb": baseline["emb"],
                 "val_kp": baseline.get("kp", ""),
                 "val_kp_masked": baseline.get("kp_masked", ""),
                 "wall_s": round(time.time() - start_time, 1)})

    best = baseline["emb"]
    best_step = 0
    stale = 0
    step = 0
    micro = 0
    stop_reason = "max_steps"
    stop = False
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        if stop:
            break
        for batch in train_loader:
            frames = batch[0].to(args.device, non_blocking=True)
            actions = batch[1].to(args.device, non_blocking=True)
            heatmaps = batch[3].to(args.device, non_blocking=True) if with_kp else None
            valid_kp = batch[4].to(args.device, non_blocking=True) if with_kp else None
            loss, stats = singlestep_loss(model, frames, actions, amp_dtype,
                                          heatmaps, valid_kp, args.kp_weight)
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
                rate = step / max(time.time() - start_time, 1e-9)
                print(f"[train] epoch {epoch} step {step}/{total_steps} "
                      f"loss {float(loss):.5f} emb {float(stats['emb']):.5f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} {rate:.2f} it/s "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
                      flush=True)
                curve.write({"step": step, "epoch": epoch, "kind": "train",
                             "train_loss": float(loss),
                             "train_emb": float(stats["emb"]),
                             "lr": scheduler.get_last_lr()[0],
                             "wall_s": round(time.time() - start_time, 1)})

            if step % args.eval_every == 0 or step == total_steps:
                record = validate(model, val_loader, args.device,
                                  args.val_batches, amp_dtype, with_kp,
                                  args.kp_weight)
                value = record["emb"]
                improved = value < best - args.min_delta
                print(f"[val] step {step} held-out latent MSE {value:.5f} "
                      f"(best {best:.5f} @ {best_step}"
                      f"{', improved' if improved else f', stale {stale + 1}'})",
                      flush=True)
                curve.write({"step": step, "epoch": epoch, "kind": "val",
                             "val_emb": value,
                             "val_kp": record.get("kp", ""),
                             "val_kp_masked": record.get("kp_masked", ""),
                             "lr": scheduler.get_last_lr()[0],
                             "wall_s": round(time.time() - start_time, 1)})

                payload = {
                    "model": model.state_dict(),
                    "args": {k: (str(v) if isinstance(v, Path) else v)
                             for k, v in vars(args).items()},
                    "num_context": args.num_context, "n_future": 0,
                    "task": args.task, "step": step, "epoch": epoch,
                    "val_loss": value, "baseline_val_loss": baseline["emb"],
                }
                torch.save(payload, args.out_dir / "last.pth.tar")
                if improved or not (args.out_dir / "best.pth.tar").exists():
                    torch.save(payload, args.out_dir / "best.pth.tar")
                del payload

                if improved:
                    best, best_step, stale = value, step, 0
                else:
                    stale += 1
                    if args.patience and stale >= args.patience:
                        stop_reason = f"early_stop(patience={args.patience})"
                        print(f"[stop] held-out latent MSE has not improved for "
                              f"{stale} evaluations; best {best:.5f} @ step "
                              f"{best_step}", flush=True)
                        stop = True
                        break

            if step >= total_steps:
                stop = True
                break

    elapsed = time.time() - start_time
    summary = {
        "task": args.task, "num_context": args.num_context,
        "amp": args.amp, "seed": args.seed,
        "task_data_root": str(args.task_data_root),
        "world_to_camera": str(args.world_to_camera),
        "init_ckpt": str(args.init_ckpt) if args.init_ckpt else None,
        "init_report": init_report,
        "kp_labels": str(args.kp_labels) if with_kp else None,
        "baseline_val_loss": baseline["emb"],
        "best_val_loss": best, "best_step": best_step,
        "steps": step, "total_steps": total_steps,
        "stop_reason": stop_reason,
        "batch": args.batch, "accum": args.accum, "lr": args.lr,
        "train_episodes": len(train_set.usable),
        "val_episodes": len(val_set.usable),
        "windows_per_epoch": len(train_set),
        "window_pool": train_set.n_windows_total,
        "wall_clock_s": round(elapsed, 1),
        "it_per_s": step / max(elapsed, 1e-9),
        "peak_gib": (float(torch.cuda.max_memory_allocated() / 2**30)
                     if args.device != "cpu" else None),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {stop_reason}: best held-out latent MSE {best:.5f} "
          f"@ step {best_step} (init {baseline['emb']:.5f}, {elapsed:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
