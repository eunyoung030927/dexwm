#!/usr/bin/env python
"""K1: how accurate is the keypoint head on held-out demos, and does it fire on
an object it never saw?

Two questions, both answered off the same checkpoint:

  * `--mode demo`  - roll the model out on the held-out DexJoCo validation
    episodes exactly as training did (autoregressive, N future steps), decode
    the predicted belief maps back to pixels and compare against the labels.
    Reported per channel per horizon: median / mean peak error in crop pixels
    (392x224) and PCK at 5 / 10 / 20 px.  Both the argmax peak and the softmax
    expectation (`get_soft_keypoints_from_beliefmap`, which is what the K2
    readout uses) are reported, because a multi-modal map can have a good peak
    and a useless expectation.

  * `--mode branch` - the object channel on the Set-A branch scenes, where the
    graspable object is a CAN that never appears in training (the demos only
    ever contain the boxed food).  Reports how much of the object channel's
    heat lands inside the can's projected footprint and whether the argmax peak
    is inside it, both for the model's estimate of the present (head on the real
    context latent) and for the h=30 prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from train_dexjoco_ft import build_model, load_dataset_module, load_weights  # noqa: E402
from train_dexjoco_ms import (  # noqa: E402
    HEATMAP_GRID, kp_forward_last, multistep_predict, retarget_kp_head,
)
from utils.image_utils import (  # noqa: E402
    get_keypoints_from_beliefmap, get_soft_keypoints_from_beliefmap,
)

CROP_WIDTH, CROP_HEIGHT = 392, 224
PCK_THRESHOLDS = (5.0, 10.0, 20.0)


def vls_root() -> Path:
    root = os.environ.get("VLS_CODE_ROOT", "/workspace/vls/code_dexwm")
    for extra in (root, str(Path(root) / "scripts" / "dexwm")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    return Path(root)


def prepare_model(checkpoint: Path, device: str, channels: int):
    model = build_model(device, compile_attention=True)
    retarget_kp_head(model, channels)
    load_weights(model, checkpoint)
    model.forward_kp = kp_forward_last(model)
    model.eval()
    return model


def checkpoint_channels(checkpoint: Path) -> list[str]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    names = state.get("kp_channels")
    if names is None:
        weight = state["model"]["kp_layer.vit_decoder.head.0.weight"]
        names = [f"kp{i}" for i in range(weight.shape[0] // (14 * 14))]
    del state
    return list(names)


# --------------------------------------------------------------- demo mode
@torch.no_grad()
def evaluate_demos(model, loader, device, num_context, n_future, limit,
                   channels) -> dict:
    errors = {tag: [[[] for _ in channels] for _ in range(n_future)]
              for tag in ("peak", "soft")}
    for index, (frames, actions, _rel, heatmaps, valid_kp, *_rest) in enumerate(loader):
        if limit and index >= limit:
            break
        frames = frames.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, _, maps = multistep_predict(model, frames, actions, num_context,
                                           n_future, with_kp=True)
        maps = maps.float().cpu()                     # (B, N, C, H, W)
        batch, steps, count = maps.shape[:3]
        for step in range(steps):
            predicted = {
                "peak": get_keypoints_from_beliefmap(maps[:, step]),
                "soft": get_soft_keypoints_from_beliefmap(maps[:, step]),
            }
            truth = get_keypoints_from_beliefmap(heatmaps[:, step])
            for tag, values in predicted.items():
                distance = torch.linalg.norm(values - truth, dim=-1).numpy()
                for sample in range(batch):
                    for channel in range(count):
                        if valid_kp[sample, step, channel] > 0:
                            errors[tag][step][channel].append(
                                float(distance[sample, channel]))
    out = {}
    for tag, per_step in errors.items():
        out[tag] = []
        for step, per_channel in enumerate(per_step):
            entry = {"horizon": (step + 1) * 5}
            for channel, values in zip(channels, per_channel):
                array = np.asarray(values)
                entry[channel] = {
                    "n": int(array.size),
                    "median_px": float(np.median(array)) if array.size else None,
                    "mean_px": float(array.mean()) if array.size else None,
                    **{f"pck@{int(threshold)}": (float((array <= threshold).mean())
                                                 if array.size else None)
                       for threshold in PCK_THRESHOLDS},
                }
            out[tag].append(entry)
    return out


# --------------------------------------------------------------- branch mode
def overlay(frame_bgr, heat, centre, radius, title):
    """Heatmap of one channel on the 392x224 crop, with the GT footprint."""
    import cv2
    positive = np.clip(heat, 0.0, None)
    scaled = positive / max(float(positive.max()), 1e-9)
    colour = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    blended = cv2.addWeighted(frame_bgr, 0.55, colour, 0.45, 0)
    cv2.circle(blended, (int(centre[0]), int(centre[1])), int(radius),
               (255, 255, 255), 1)
    peak = np.unravel_index(int(heat.argmax()), heat.shape)
    cv2.drawMarker(blended, (int(peak[1]), int(peak[0])), (60, 255, 60),
                   cv2.MARKER_CROSS, 12, 2)
    cv2.putText(blended, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0), 3)
    cv2.putText(blended, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1)
    return blended


@torch.no_grad()
def evaluate_branches(model, data_dir: Path, device, steps, horizon, channels,
                      conditioning, radius_scale, figure_dir=None,
                      figure_branches=3) -> dict:
    vls_root()
    from label_demo_keypoints import project, to_crop
    from score_branch_candidates import CONTEXT_LEN, WM_STEP, decode, preprocess
    from rollout_branch_candidates import chunk_deltas, _frozen_encoder

    object_channel = channels.index("object")
    rows = []
    figure_branches_done = []
    # the rollout only needs the latents; decoding a heatmap on every one of the
    # 12 x 20 intermediate steps would be pure waste
    kp_head = model.forward_kp
    model.forward_kp = lambda *_a, **_k: (None, None)
    for path in sorted(data_dir.glob("branch_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            keys = sorted(key for key in data.files
                          if re.fullmatch(r"context_\d+", key))
            context = torch.stack([preprocess(decode(data[key]))
                                   for key in keys[-CONTEXT_LEN:]])
            deltas = chunk_deltas(data, steps)[conditioning]
            world_to_camera = np.asarray(data["world_to_camera"])
            # the can is a capsule between the two object keypoints; its
            # projected footprint is approximated by the disc that covers the
            # projected bottom/top pair plus the projected radius
            live = data["gt_live_keypoints"]
            # inputs stay float32 and autocast does the casting, exactly as in
            # training (the rollout evaluator instead casts the whole model)
            dtype = torch.float32
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                latent = model.encode_image(context[None].to(device, dtype))[0]
                current = model.kp_layer(latent[-1:][None].float(),
                                         HEATMAP_GRID)[0, 0].float().cpu()
                padded_images = torch.cat(
                    [context, context[-1:].repeat(steps, 1, 1, 1)], dim=0)
                padded_latent = torch.cat(
                    [latent, latent[-1:].repeat(steps, 1, 1)], dim=0)
                zeros = torch.zeros(deltas.shape[0], CONTEXT_LEN - 1,
                                    deltas.shape[-1])
                padded_actions = torch.cat([zeros, deltas], dim=1)
                predicted = []
                for start in range(0, deltas.shape[0], 2):
                    stop = min(start + 2, deltas.shape[0])
                    span = stop - start
                    prev_emb = None
                    for step in range(steps):
                        images = padded_images[step:step + CONTEXT_LEN + 1]
                        images = images.unsqueeze(0).repeat(
                            span, 1, 1, 1, 1).to(device, dtype)
                        actions = padded_actions[start:stop,
                                                 step:step + CONTEXT_LEN].to(device, dtype)
                        window = padded_latent[step:step + CONTEXT_LEN + 1].unsqueeze(0)
                        with _frozen_encoder(model, window):
                            pred, *_ = model(images, actions, prev_emb=prev_emb,
                                             action_diff=True)
                        latest = pred[:, -1:].detach()
                        prev_emb = latest if prev_emb is None else torch.cat(
                            [prev_emb, latest], dim=1)
                    predicted.append(model.kp_layer(latest.float(),
                                                    HEATMAP_GRID)[:, 0].float().cpu())
                predicted = torch.cat(predicted, dim=0)

            def footprint(points_world):
                uv, _ = project(np.asarray(points_world), world_to_camera)
                uv = to_crop(uv)
                centre = uv.mean(axis=0)
                extent = max(float(np.linalg.norm(uv[1] - uv[0])) / 2.0, 8.0)
                return centre, radius_scale * (extent + 12.0)

            grid_y, grid_x = np.meshgrid(np.arange(CROP_HEIGHT),
                                         np.arange(CROP_WIDTH), indexing="ij")

            def measure(channel_map, keypoints):
                centre, radius = footprint(np.stack([keypoints[3], keypoints[4]]))
                inside = ((grid_x - centre[0]) ** 2
                          + (grid_y - centre[1]) ** 2) <= radius ** 2
                values = channel_map.numpy()
                positive = np.clip(values, 0.0, None)
                peak = np.unravel_index(int(values.argmax()), values.shape)
                return {
                    "mass_inside": float(positive[inside].sum()
                                         / max(positive.sum(), 1e-9)),
                    "peak_inside": bool(inside[peak]),
                    "peak_px": [float(peak[1]), float(peak[0])],
                    "gt_px": [float(centre[0]), float(centre[1])],
                    "peak_error_px": float(np.hypot(peak[1] - centre[0],
                                                    peak[0] - centre[1])),
                    "radius_px": float(radius),
                    "map_max": float(values.max()),
                }

            # the present is the same scene for every candidate, so it is
            # measured once per branch point
            rows.append({"branch": path.stem, "candidate": -1, "when": "present",
                         **measure(current[object_channel], live[0, 0])})
            if figure_dir is not None and len(figure_branches_done) < figure_branches:
                import cv2
                figure_branches_done.append(path.stem)
                frame = decode(data[keys[-1]])
                height, width = frame.shape[:2]
                crop = min(height, int(round(width / (CROP_WIDTH / CROP_HEIGHT))))
                top = (height - crop) // 2
                base = cv2.cvtColor(
                    cv2.resize(frame[top:top + crop], (CROP_WIDTH, CROP_HEIGHT)),
                    cv2.COLOR_RGB2BGR)
                tiles = []
                centre, radius = footprint(np.stack([live[0, 0][3], live[0, 0][4]]))
                for name in channels:
                    tiles.append(overlay(base, current[channels.index(name)].numpy(),
                                         centre, radius, f"present {name}"))
                for index in (0, min(1, predicted.shape[0] - 1)):
                    centre_h, radius_h = footprint(
                        np.stack([live[index, horizon][3], live[index, horizon][4]]))
                    tiles.append(overlay(
                        base, predicted[index][object_channel].numpy(),
                        centre_h, radius_h, f"h{horizon} cand{index} object"))
                columns = 3
                usable = len(tiles) - len(tiles) % columns
                grid = np.vstack([np.hstack(tiles[i:i + columns])
                                  for i in range(0, usable, columns)])
                Path(figure_dir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(figure_dir) / f"heat_{path.stem}.png"), grid)
            for index in range(predicted.shape[0]):
                rows.append({"branch": path.stem, "candidate": index,
                             "when": f"h{horizon}",
                             **measure(predicted[index][object_channel],
                                       live[index, horizon])})
                rows.append({"branch": path.stem, "candidate": index,
                             "when": f"h{horizon}_vs_start",
                             **measure(predicted[index][object_channel],
                                       live[index, 0])})
        print(f"{path.name}: can-firing checked", flush=True)
    model.forward_kp = kp_head

    summary = {}
    for tag in sorted({row["when"] for row in rows}):
        subset = [row for row in rows if row["when"] == tag]
        summary[tag] = {
            "n": len(subset),
            "mass_inside_median": float(np.median([r["mass_inside"] for r in subset])),
            "peak_inside_rate": float(np.mean([r["peak_inside"] for r in subset])),
            "peak_error_median_px": float(np.median(
                [r["peak_error_px"] for r in subset])),
            "map_max_median": float(np.median([r["map_max"] for r in subset])),
        }
    return {"summary": summary, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["demo", "branch", "both"], default="both")
    parser.add_argument("--frames", type=Path, default=Path("/root/dexwm_ft/frames"))
    parser.add_argument("--kp-labels", type=Path,
                        default=Path("/root/dexwm_ft/kp_labels"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/root/dexwm_branch/mixed_setA"))
    parser.add_argument("--world-to-camera", type=Path,
                        default=Path("/root/dexwm_branch/data/branch_seed00_step00090.npz"))
    parser.add_argument("--n-future", type=int, default=6)
    parser.add_argument("--num-context", type=int, default=8)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-windows-per-episode", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--conditioning", default="cmd_rel")
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--figure-dir", type=Path,
                        default=Path("/root/dexwm_branch/figs/kp"))
    parser.add_argument("--figure-branches", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    channels = checkpoint_channels(args.checkpoint)
    model = prepare_model(args.checkpoint, args.device, len(channels))
    payload = {"checkpoint": str(args.checkpoint), "channels": channels,
               "n_future": args.n_future}

    if args.mode in ("demo", "both"):
        demos = load_dataset_module()
        with np.load(args.world_to_camera, allow_pickle=False) as reference:
            world_to_camera = np.asarray(reference["world_to_camera"])
        val_set = demos.DexJoCoDemoDataset(
            args.frames, world_to_camera=world_to_camera, train=False,
            n_future=args.n_future, num_context=args.num_context,
            windows_per_episode=args.val_windows_per_episode,
            kp_root=args.kp_labels)
        loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers,
            pin_memory=True, drop_last=False)
        payload["val_episodes"] = val_set.usable
        payload["demo"] = evaluate_demos(model, loader, args.device,
                                         args.num_context, args.n_future,
                                         args.limit, channels)
    if args.mode in ("branch", "both"):
        payload["branch"] = evaluate_branches(
            model, args.data_dir, args.device, args.horizon // 5, args.horizon,
            channels, args.conditioning, args.radius_scale, args.figure_dir,
            args.figure_branches)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    printable = {key: value for key, value in payload.items() if key != "branch"}
    if "branch" in payload:
        printable["branch"] = payload["branch"]["summary"]
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
