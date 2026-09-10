# UR7E + RH5DG2 demonstration loader for DexWM fine-tuning (DynaGuide comparison).
#
# Sibling of datasets/dexjoco_demos.DexJoCoDemoDataset, matching its return
# contract exactly so train_dexjoco_ctx1.py / train_dexjoco_ms.py drive it
# unchanged:
#
#   __getitem__ -> (frames[T,3,224,392], actions[T-1,132], rel_t, 0, 0, metadata)
#
# where T = num_context + n_future and the 132-D action deltas come from
# core.dexwm_rh5dg2_action_adapter (RH5DG2 FK -> DexWM's MANO-like keypoint
# contract), i.e. the SAME function a scored DynaGuide candidate uses.
#
# Source dataset: the `*_abs_ee` variant of DexSteer/isaaclab_ur7e_3task_fixed
# (LeRobot v3.0).  Per frame it stores
#     observation.state[24] = [6 UR7e joints, 18 RH5DG2 full hand joints]
#     action[22]            = [ee_pos(3), ee_rot6d(6), hand_active_offsets(13)]
# and the images as av1-encoded mp4 (third_person = the fixed external view,
# analogous to the DexJoCo front camera).
#
# The kp-head auxiliary loss is OFF here (kp_root path from the DexJoCo loader is
# intentionally not reproduced): the DynaGuide comparison scores a latent
# distance, so only frames + 132-D action deltas + rel_t are needed.
#
# TODOs (runtime-sourced constants, tracked in the bridge too):
#   * world_to_camera: fixed third_person extrinsic per task (else root frame).
#   * default_active(13)/default_full(18) + hand_scale + rot6d_layout: the
#     RH5DG2 articulation defaults / action scaling the env uses.  Verify by
#     reproducing one env keypoint; wrong defaults only bias the hand keypoints,
#     not the pipeline shape.
#   * context_stride: dataset is 50 fps; pick the stride that matches DexWM's
#     training temporal horizon (DexJoCo used stride 5 at its control rate).

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
CONTEXT_STRIDE = 5
THIRD_PERSON_KEY = "observation.images.third_person"


def _vls_root() -> Path:
    root = os.environ.get("VLS_CODE_ROOT", "/workspace/vls/code")
    if root not in sys.path:
        sys.path.insert(0, root)
    return Path(root)


def preprocess(image: np.ndarray, img_size: int = 224, patch_size: int = 14) -> torch.Tensor:
    """RGB uint8 HxWx3 -> normalised (3,224,392). Byte-identical to the DexJoCo path."""
    import cv2

    width_target = 392 if patch_size == 14 else 384
    height, width = image.shape[:2]
    target_aspect = width_target / float(img_size)
    crop_height = min(height, int(round(width / target_aspect)))
    top = (height - crop_height) // 2
    image = image[top: top + crop_height]
    image = cv2.resize(image, (width_target, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


# --------------------------------------------------------------------------
# LeRobot v3.0 metadata / parquet reading
# --------------------------------------------------------------------------
def _read_meta(root: Path):
    """Return (info, episodes) where episodes is a list of per-episode dicts."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    ep_dir = root / "meta" / "episodes"
    tables = [pq.read_table(p) for p in sorted(ep_dir.rglob("*.parquet"))]
    if not tables:
        raise FileNotFoundError(f"no episode metadata under {ep_dir}")
    import pyarrow as pa

    rows = pa.concat_tables(tables).to_pylist()
    episodes = []
    for r in rows:
        episodes.append({
            "episode_index": int(r["episode_index"]),
            "length": int(r["length"]),
            "from": int(r["dataset_from_index"]),
            "to": int(r["dataset_to_index"]),
            "video_chunk": int(r[f"videos/{THIRD_PERSON_KEY}/chunk_index"]),
            "video_file": int(r[f"videos/{THIRD_PERSON_KEY}/file_index"]),
            "video_from_ts": float(r[f"videos/{THIRD_PERSON_KEY}/from_timestamp"]),
        })
    episodes.sort(key=lambda e: e["episode_index"])
    return info, episodes


def _read_columns(root: Path):
    """Concatenate observation.state[N,24] and action[N,22] across data parquets."""
    import pyarrow.parquet as pq

    data_dir = root / "data"
    paths = sorted(data_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no data parquet under {data_dir}")
    states, actions = [], []
    for p in paths:
        t = pq.read_table(p, columns=["observation.state", "action"])
        states.append(np.asarray(t["observation.state"].to_pylist(), dtype=np.float32))
        actions.append(np.asarray(t["action"].to_pylist(), dtype=np.float32))
    return np.concatenate(states, 0), np.concatenate(actions, 0)


def _video_path(root: Path, info, chunk: int, file: int) -> Path:
    rel = info["video_path"].format(video_key=THIRD_PERSON_KEY, chunk_index=chunk, file_index=file)
    return root / rel


def _decode_window(video: Path, from_ts: float, frame_indices, fps: float):
    """Decode the requested (monotonically increasing) frame indices of one episode.

    Frame k of the episode sits at ``from_ts + k / fps`` inside the shared mp4.
    We seek once to the first needed frame and decode forward, which is the
    cheap access pattern for a contiguous DexWM window.
    """
    import av

    wanted = list(frame_indices)
    target_ts = [from_ts + k / fps for k in wanted]
    out = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        seek_ts = int(max(target_ts[0] - 0.5, 0.0) / stream.time_base)
        container.seek(seek_ts, stream=stream, any_frame=False, backward=True)
        ptr = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            while ptr < len(wanted) and t + 1e-6 >= target_ts[ptr]:
                out[wanted[ptr]] = frame.to_ndarray(format="rgb24")
                ptr += 1
            if ptr >= len(wanted):
                break
    if len(out) != len(wanted):
        missing = [k for k in wanted if k not in out]
        raise RuntimeError(f"{video.name}: could not decode frames {missing} (from_ts={from_ts})")
    return [out[k] for k in wanted]


def _load_cached(frames_root: Path, episode: int, frame_indices):
    """Read pre-decoded RGB frames from the export cache (JPEG npz)."""
    import cv2

    path = frames_root / f"episode_{episode:03d}.npz"
    with np.load(path, allow_pickle=False) as store:
        out = []
        for i in frame_indices:
            buf = store[f"frame_{i:05d}"]
            bgr = cv2.imdecode(np.asarray(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"episode {episode} frame {i}: cached JPEG decode failed")
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return out


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class UR7eRH5DG2DemoDataset(Dataset):
    """Successful UR7E+RH5DG2 demos, windowed for DexWM's next-latent loss.

    Args mirror DexJoCoDemoDataset; the extra ones configure the RH5DG2 bridge.
    """

    def __init__(self, root_folder, world_to_camera=None, num_context=8,
                 patch_size=14, img_size=224, train=True, val_every=10,
                 windows_per_episode=40, seed=0, context_stride=CONTEXT_STRIDE,
                 n_future=1, default_active=None, default_full=None,
                 hand_scale=1.0, rot6d_layout="columns", frames_root=None,
                 **_ignored):
        super().__init__()
        self.root = Path(root_folder)
        # Optional pre-decoded frame cache (export_ur7e_frames.py).  When set,
        # frames come from JPEG npz instead of on-the-fly av1 mp4 decode, which
        # is the difference between a fast training run and a decode-bound one.
        # state/action/lengths still come from root_folder's parquet (small).
        self.frames_root = Path(frames_root) if frames_root else None
        self.info, episodes = _read_meta(self.root)
        self.fps = float(self.info.get("fps", 50))
        self.states, self.actions = _read_columns(self.root)

        idx = [e["episode_index"] for e in episodes]
        held_out = {e for i, e in enumerate(idx) if (i + 1) % val_every == 0}
        self.episodes = [e for e in episodes
                         if (e["episode_index"] in held_out) != bool(train)]
        self.by_index = {e["episode_index"]: e for e in episodes}

        self.num_context = num_context
        self.n_future = int(n_future)
        if self.n_future < 1:
            raise ValueError("n_future must be >= 1")
        self.context_stride = context_stride
        self.span = (num_context - 1 + self.n_future) * context_stride
        self.patch_size = patch_size
        self.img_size = img_size
        self.train = train
        self.windows_per_episode = windows_per_episode
        self.seed = seed
        self.world_to_camera = (None if world_to_camera is None else
                                torch.tensor(np.asarray(world_to_camera), dtype=torch.float32))
        self.hand_scale = hand_scale
        self.rot6d_layout = rot6d_layout

        _vls_root()
        from core.dexwm_rh5dg2_action_adapter import dexwm_rh5dg2_action_deltas
        from core.geometry.rh5dg2_kinematics import RH5DG2Kinematics

        self._deltas = dexwm_rh5dg2_action_deltas
        self._kin = RH5DG2Kinematics()
        self._default_active = default_active
        self._default_full = default_full

        self.usable = [e for e in self.episodes if e["length"] - 1 - self.span > 0]
        if not self.usable:
            raise RuntimeError("no episode long enough for the requested window span")
        self.n_windows_total = sum(e["length"] - self.span for e in self.usable)

    def __len__(self):
        return len(self.usable) * self.windows_per_episode

    def _start(self, ep, slot: int) -> int:
        last = ep["length"] - 1 - self.span
        low = int(round(slot * last / self.windows_per_episode))
        high = max(int(round((slot + 1) * last / self.windows_per_episode)), low + 1)
        if self.train:
            return int(np.random.randint(low, min(high, last + 1)))
        return int(min((low + high) // 2, last))

    def __getitem__(self, index):
        ep = self.usable[index // self.windows_per_episode]
        slot = index % self.windows_per_episode
        start = self._start(ep, slot)
        frame_ids = [start + k * self.context_stride
                     for k in range(self.num_context + self.n_future)]

        if self.frames_root is not None:
            rgb = _load_cached(self.frames_root, ep["episode_index"], frame_ids)
        else:
            rgb = _decode_window(
                _video_path(self.root, self.info, ep["video_chunk"], ep["video_file"]),
                ep["video_from_ts"], frame_ids, self.fps)
        frames = torch.stack([preprocess(f, self.img_size, self.patch_size) for f in rgb])

        rows = [ep["from"] + f for f in frame_ids]
        absolute = torch.tensor(self.actions[rows], dtype=torch.float32)  # (T, 22) abs_ee
        actions = self._deltas(
            absolute, self._kin, world_to_camera=self.world_to_camera,
            default_active=self._default_active, default_full=self._default_full,
            hand_scale=self.hand_scale, rot6d_layout=self.rot6d_layout)

        rel_t = np.full(self.num_context - 1 + self.n_future,
                        self.context_stride, dtype=np.int64)
        metadata = {"episode": ep["episode_index"], "start": start}
        return frames, actions, rel_t, 0, 0, metadata


if __name__ == "__main__":  # smoke test: python ur7e_rh5dg2_demos.py <local_abs_ee_dir>
    import sys as _sys

    ds = UR7eRH5DG2DemoDataset(_sys.argv[1], num_context=4, n_future=1,
                               context_stride=5, windows_per_episode=4)
    print("episodes:", len(ds.usable), "len:", len(ds))
    frames, actions, rel_t, _, _, meta = ds[0]
    print("frames:", tuple(frames.shape), "actions:", tuple(actions.shape),
          "rel_t:", rel_t.shape, "meta:", meta)
