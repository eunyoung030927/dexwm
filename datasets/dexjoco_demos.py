# DexJoCo pick_bucket demonstration loader for DexWM fine-tuning.
#
# Mirrors datasets/robocasa_random_movement.RobocasaRandomDataset:
#   * returns a (num_context + 1)-frame window sampled at a fixed stride,
#   * returns (num_context, action_dim) keypoint deltas, one per transition,
#   * returns rel_t, and placeholders for the heatmap keypoint auxiliary loss
#     (DexJoCo demos carry no 2-D keypoint annotation, so the kp loss is
#     disabled by the trainer instead of being fed zeros).
#
# The frames come from `scripts/dexwm/export_demo_frames.py` in the VLS repo,
# which decodes the LeRobot episodes and re-encodes each 640x640 front frame
# with the same JPEG settings the branch collector uses.  The 22-D keypoint
# action is produced by the VLS repo's `core.dexwm_action_adapter`, i.e. by the
# exact function every DexWM evaluation in this project already uses, so the
# action a training window sees and the action a scored candidate sees are the
# same object.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CONTEXT_STRIDE = 5   # control steps between DexWM frames, as in every VLS trace
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _vls_root() -> Path:
    root = os.environ.get("VLS_CODE_ROOT", "/workspace/vls/code_dexwm")
    if root not in sys.path:
        sys.path.insert(0, root)
    return Path(root)


def preprocess(image: np.ndarray, img_size: int = 224, patch_size: int = 14) -> torch.Tensor:
    """Byte-identical to scripts/dexwm/score_branch_candidates.preprocess."""
    width_target = 392 if patch_size == 14 else 384
    height, width = image.shape[:2]
    target_aspect = width_target / float(img_size)
    crop_height = min(height, int(round(width / target_aspect)))
    top = (height - crop_height) // 2
    image = image[top: top + crop_height]
    image = cv2.resize(image, (width_target, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def state_to_action(raw_state: np.ndarray) -> np.ndarray:
    """23-D LeRobot observation.state -> 22-D DexJoCo absolute action.

    Same conversion as scripts/dexwm/score_branch_candidates.state_to_action,
    which reads the first 23 entries of the 38-D simulator state.
    """
    from scipy.spatial.transform import Rotation

    state = np.asarray(raw_state, dtype=np.float64).reshape(-1)
    action = np.empty(22, dtype=np.float64)
    action[:3] = state[:3]
    action[3:6] = Rotation.from_quat(state[[4, 5, 6, 3]]).as_rotvec()
    action[6:22] = state[7:23]
    return action


class DexJoCoDemoDataset(Dataset):
    """Successful DexJoCo demonstrations, windowed for DexWM's next-latent loss.

    Args:
        root_folder: directory written by export_demo_frames.py.
        world_to_camera: 4x4 world->camera extrinsic of the front camera; the
            DexJoCo front camera is fixed, and it is identical in the demo
            dataset and in every branch npz (verified numerically).
        train: train split (True) or held-out validation episodes (False).
        val_every: every val_every-th episode (by index) is held out.
        windows_per_episode: windows drawn per episode per epoch.
        n_future: number of future frames after the `num_context` context
            frames.  ``1`` (default) is the single-step window this class was
            written for; ``N > 1`` returns ``num_context + N`` frames and
            ``num_context - 1 + N`` consecutive deltas, which is what the
            multistep (autoregressive rollout) trainer needs.  The last ``N``
            deltas are the ones that drive the rollout; the leading
            ``num_context - 1`` are the context transitions.
    """

    def __init__(self, root_folder, world_to_camera, max_context_len=40,
                 num_context=8, patch_size=14, img_size=224, train=True,
                 val_every=10, windows_per_episode=40, seed=0, aug=None,
                 backbone_name="dinov2", context_stride=CONTEXT_STRIDE,
                 n_future=1, **_ignored):
        super().__init__()
        self.root = Path(root_folder)
        manifest = json.loads((self.root / "manifest.json").read_text())
        episodes = sorted(entry["episode"] for entry in manifest["episodes"])
        held_out = [e for i, e in enumerate(episodes) if (i + 1) % val_every == 0]
        self.episodes = [e for e in episodes
                         if (e not in held_out) == bool(train)]
        self.lengths = {entry["episode"]: entry["length"]
                        for entry in manifest["episodes"]}
        self.num_context = num_context
        self.n_future = int(n_future)
        if self.n_future < 1:
            raise ValueError("n_future must be >= 1")
        self.context_stride = context_stride
        # frames per window = num_context + n_future, so the window spans
        # (num_context - 1 + n_future) strides.  n_future = 1 reproduces the
        # original num_context * context_stride span exactly.
        self.span = (num_context - 1 + self.n_future) * context_stride
        self.patch_size = patch_size
        self.img_size = img_size
        self.train = train
        self.windows_per_episode = windows_per_episode
        self.seed = seed
        self.world_to_camera = torch.tensor(np.asarray(world_to_camera),
                                            dtype=torch.float32)
        self.backbone_name = backbone_name
        _vls_root()
        # imported lazily so the module is importable without the VLS repo
        from core.dexwm_action_adapter import dexwm_action_deltas
        self._deltas = dexwm_action_deltas
        self.usable = [e for e in self.episodes
                       if self.lengths[e] - 1 - self.span > 0]
        self.n_windows_total = sum(self.lengths[e] - self.span
                                   for e in self.usable)

    def __len__(self):
        return len(self.usable) * self.windows_per_episode

    def _start(self, episode: int, slot: int) -> int:
        last = self.lengths[episode] - 1 - self.span
        low = int(round(slot * last / self.windows_per_episode))
        high = int(round((slot + 1) * last / self.windows_per_episode))
        high = max(high, low + 1)
        if self.train:
            return int(np.random.randint(low, min(high, last + 1)))
        return int(min((low + high) // 2, last))

    def __getitem__(self, index):
        episode = self.usable[index // self.windows_per_episode]
        slot = index % self.windows_per_episode
        start = self._start(episode, slot)
        indices = [start + k * self.context_stride
                   for k in range(self.num_context + self.n_future)]

        with np.load(self.root / f"episode_{episode:03d}.npz",
                     allow_pickle=False) as store:
            frames = []
            for i in indices:
                buffer = store[f"frame_{i:05d}"]
                bgr = cv2.imdecode(np.asarray(buffer, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"episode {episode} frame {i}: JPEG decode failed")
                frames.append(preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                         self.img_size, self.patch_size))
            states = store["state"][indices]

        absolute = torch.tensor(np.stack([state_to_action(s) for s in states]),
                                dtype=torch.float32)
        actions = self._deltas(absolute, world_to_camera=self.world_to_camera)

        curr_frames = torch.stack(frames)
        rel_t = np.full(self.num_context - 1 + self.n_future,
                        self.context_stride, dtype=np.int64)
        metadata = {"episode": episode, "start": start}
        return curr_frames, actions, rel_t, 0, 0, metadata
