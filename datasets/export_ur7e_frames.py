"""Pre-decode UR7E+RH5DG2 third_person frames -> fast JPEG cache for training.

On-the-fly av1 decode per DexWM window is decode-bound (6000 steps x batch x
window frames = hundreds of thousands of decodes).  This decodes each episode's
frames ONCE, sequentially, and writes them JPEG-encoded so
UR7eRH5DG2DemoDataset(frames_root=...) reads them with a cheap cv2.imdecode --
the same two-tier design as the DexJoCo export_demo_frames.py path.

Output layout (mirrors the DexJoCo frame cache):
    <out>/manifest.json           {"episodes":[{"episode":i,"length":L}, ...]}
    <out>/episode_000.npz         frame_00000 ... frame_{L-1:05d}  (uint8 JPEG buffers)

state/action are NOT copied here -- the loader reads those from the source
parquet (small); this cache is frames only.

Usage:
    VLS_CODE_ROOT=/workspace/vls/code python export_ur7e_frames.py \
        --src /root/dexwm_ur7e/raw/cup_hang_abs_ee \
        --out /root/dexwm_ur7e/frames/cup_hang \
        --jpeg-quality 95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from ur7e_rh5dg2_demos import THIRD_PERSON_KEY, _read_meta, _video_path


def _encode(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.squeeze()


def _decode_episode(video: Path, from_ts: float, length: int, fps: float):
    """Sequentially decode ``length`` frames of one episode from its shared mp4."""
    targets = [from_ts + k / fps for k in range(length)]
    out, ptr = {}, 0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        seek_ts = int(max(targets[0] - 0.5, 0.0) / stream.time_base)
        container.seek(seek_ts, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            while ptr < length and t + 1e-6 >= targets[ptr]:
                out[ptr] = frame.to_ndarray(format="rgb24")
                ptr += 1
            if ptr >= length:
                break
    if len(out) != length:
        missing = [k for k in range(length) if k not in out]
        raise RuntimeError(f"{video.name}: missing frames {missing[:5]}... (from_ts={from_ts})")
    return [out[k] for k in range(length)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="LeRobot *_abs_ee subdir")
    ap.add_argument("--out", type=Path, required=True, help="frame cache dir to write")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    info, episodes = _read_meta(args.src)
    fps = float(info.get("fps", 50))
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {"episodes": []}
    for e in episodes:
        ei, length = e["episode_index"], e["length"]
        dst = args.out / f"episode_{ei:03d}.npz"
        manifest["episodes"].append({"episode": ei, "length": length})
        if dst.exists() and not args.overwrite:
            print(f"[skip] episode {ei} ({length}) exists", flush=True)
            continue
        video = _video_path(args.src, info, e["video_chunk"], e["video_file"])
        frames = _decode_episode(video, e["video_from_ts"], length, fps)
        payload = {f"frame_{k:05d}": _encode(f, args.jpeg_quality)
                   for k, f in enumerate(frames)}
        np.savez(dst, **payload)
        print(f"[ok] episode {ei}: {length} frames -> {dst.name}", flush=True)

    (args.out / "manifest.json").write_text(json.dumps(manifest))
    print(f"wrote {len(manifest['episodes'])} episodes + manifest.json to {args.out}")


if __name__ == "__main__":
    main()
