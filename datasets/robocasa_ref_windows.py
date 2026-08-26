# RoboCasa Random reference windows for the DexWM loss-floor measurement.
#
# Why this file exists: our DexJoCo fine-tuning curves (train_dexjoco_ft.py /
# train_dexjoco_ms.py) flatten out at emb_loss ~0.13 (1-step) and ~0.15-0.17
# (multistep).  To tell "under-trained" from "this model's natural floor" we
# need the same loss measured on the data the public checkpoint
# (robocasa_random_finetune.pth.tar) was actually fine-tuned on.
#
# datasets/robocasa_random_movement.RobocasaRandomDataset already reads that
# data, but (a) it imports decord, which is not installed in the eval env, and
# (b) its windows are the upstream `full_seq` ones (num_context frames sampled
# uniformly at random from the whole episode), which is NOT the window our
# DexJoCo trainers use (num_context consecutive frames at a fixed stride).
# This module reproduces the upstream annotation maths byte-for-byte and offers
# BOTH window samplers, so the reference numbers can be quoted either
# upstream-faithfully or apples-to-apples with our runs.
#
# Everything numeric below (crop 600->548, resize to 392x224, ImageNet
# normalisation, keypoint re-ordering, camera extrinsics, left-hand zero
# padding, the (88, 3) curr||next action layout) is copied from
# datasets/robocasa_random_movement.py.  The only addition is that the
# next-minus-curr difference that DexWM.prepare_actions(action_diff=False)
# performs internally is done here instead, so windows can be fed with
# action_diff=True exactly like the DexJoCo windows are.

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

# world -> camera-base transform of the MURP robotview_2 camera and the
# camera-base -> optical-frame flip (robocasa_random_movement.py:135-145)
T_CAMERA_IN_BASE = np.array([[-0.0000000, 0.5000000, -0.8660254, 0.212],
                             [-1.0000000, -0.0000000, 0.0000000, 0.0],
                             [0.0000000, 0.8660254, 0.5000000, 1.614],
                             [0.0, 0.0, 0.0, 1.0]])
T_OPTICAL = np.array([[1, 0, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, -1, 0],
                      [0, 0, 0, 1]], dtype=float)
# egodex ordering + ring finger duplicated as a little-finger proxy
KEYPOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                  9, 10, 11, 12]
TASK_CLASSES = ("gripper_open_and_close", "exploratory_movements")


def image_transform(img, img_size=224, patch_size=14):
    """600x960 RoboCasa frame -> normalised (3, 224, 392) tensor."""
    ext_pix = (img.shape[0] - 548) // 2
    img = img[ext_pix:-ext_pix]
    height, width = img.shape[:2]
    aspect = width / height
    img = cv2.resize(img, (int(img_size * aspect), img_size),
                     interpolation=cv2.INTER_LINEAR)
    if patch_size == 16:
        img = img[:, 4:-4]
    tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def base_pose(obs, idx):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(obs["robot0_base_quat"][idx]).as_matrix()
    T[:3, 3] = obs["robot0_base_pos"][idx]
    return T


def process_annotation(obs, idx, prev_cam_ext=None):
    """(44, 3) pose block for frame `idx`, expressed in `prev_cam_ext`.

    Same as RobocasaRandomDataset.process_annotation with do_belief_map=False:
    21 right-hand keypoints in the (first-frame) camera frame, then the camera
    position and camera Euler angles in world, then the whole thing prefixed by
    21 zero rows standing in for the absent left hand.
    """
    T_camera_in_world = base_pose(obs, idx) @ T_CAMERA_IN_BASE
    cam_ext = np.linalg.inv(T_camera_in_world) if prev_cam_ext is None else prev_cam_ext

    keypoints = obs["robot0_right_gripper_keypoint_pose"][idx].reshape(-1, 3)[-16:]
    hand = obs["robot0_right_hand_T_world_pose_mat"][idx].reshape(4, 4)[:3, 3:].T
    keypoints = np.concatenate([hand, keypoints], axis=0)[KEYPOINT_ORDER]

    poses = []
    for key in keypoints:
        point = np.append(key, 1.0)[..., None]
        poses.append((T_OPTICAL @ (cam_ext @ point))[:3, 0])
    poses.append(T_camera_in_world[:3, 3])
    poses.append(R.from_matrix(T_camera_in_world[:3, :3]).as_euler("xyz"))
    poses = np.array(poses)
    poses = np.concatenate([poses[:21] * 0.0, poses])          # (44, 3)
    return poses, cam_ext


def deltas_from_poses(curr, nxt):
    """DexWM.prepare_actions(action_diff=False) done ahead of time.

    curr, nxt: (T, 44, 3).  Returns (T, 132).  The modulo on the last row is
    upstream's (models/model.py:434) - it treats the trailing camera Euler row
    as an angle.
    """
    # float32 first: upstream hands the model a float32 (T, 88, 3) tensor and
    # takes the difference there, and for the near-static camera row float64
    # vs float32 is the difference between a 1e-9 residual (-> 2*pi after the
    # modulo) and an exact zero.
    curr = np.asarray(curr, dtype=np.float32)
    nxt = np.asarray(nxt, dtype=np.float32)
    diff = nxt - curr
    diff[:, -1] = diff[:, -1] % np.float32(2 * np.pi)
    return diff.reshape(diff.shape[0], -1)


class RobocasaRefDataset(Dataset):
    """Windows over the authors' RoboCasa Random hdf5 files.

    sampling="stride"   : num_context + n_future consecutive frames at
                          `context_stride` (what train_dexjoco_{ft,ms}.py feed).
    sampling="full_seq" : num_context frames drawn uniformly at random from the
                          episode plus its last frame (what
                          configs/robocasa_random_multistep.yaml trains on).
    """

    def __init__(self, root_folder, num_context=8, n_future=1, train=True,
                 context_stride=1, sampling="stride", windows_per_episode=8,
                 img_size=224, patch_size=14, seed=0, split_file=None,
                 val_fraction=0.1, task_classes=TASK_CLASSES, **_ignored):
        super().__init__()
        self.root = Path(root_folder)
        self.num_context = int(num_context)
        self.n_future = int(n_future)
        self.context_stride = int(context_stride)
        self.sampling = sampling
        self.windows_per_episode = int(windows_per_episode)
        self.img_size = img_size
        self.patch_size = patch_size
        self.train = train
        self.seed = seed
        self.span = (self.num_context - 1 + self.n_future) * self.context_stride

        clips = []
        for task_class in task_classes:
            folder = self.root / task_class
            if not folder.is_dir():
                continue
            for name in sorted(os.listdir(folder)):
                if not name.endswith(".hdf5"):
                    continue
                with h5py.File(folder / name, "r") as handle:
                    for demo in sorted(handle["data"].keys()):
                        length = handle["data"][demo]["obs"][
                            "robot0_robotview_2_image"].shape[0]
                        clips.append([task_class, name, demo, int(length)])
        if not clips:
            raise RuntimeError(f"no hdf5 clips under {self.root}")

        split_file = Path(split_file) if split_file else (
            self.root.parent / "split_indices_robocasa_ref.json")
        if split_file.exists():
            split = json.loads(split_file.read_text())
        else:
            shuffled = list(clips)
            random.Random(seed).shuffle(shuffled)
            cut = int(len(shuffled) * (1.0 - val_fraction))
            split = {"train": shuffled[:cut], "test": shuffled[cut:]}
            split_file.parent.mkdir(parents=True, exist_ok=True)
            split_file.write_text(json.dumps(split, indent=2))
        self.split_file = split_file

        chosen = split["train"] if train else split["test"]
        # a window needs span + 1 frames; full_seq needs num_context + 1
        need = self.span + 1 if sampling == "stride" else self.num_context + 1
        self.usable = [c for c in chosen if c[3] >= need]
        self.clip_lengths = {(c[0], c[1], c[2]): c[3] for c in self.usable}
        self.n_windows_total = sum(max(c[3] - self.span, 1) for c in self.usable)

    def __len__(self):
        return len(self.usable) * self.windows_per_episode

    def _start(self, length, slot):
        last = length - 1 - self.span
        if last <= 0:
            return 0
        low = int(round(slot * last / self.windows_per_episode))
        high = max(int(round((slot + 1) * last / self.windows_per_episode)), low + 1)
        if self.train:
            return int(np.random.randint(low, min(high, last + 1)))
        return int(min((low + high) // 2, last))

    def _indices(self, length, slot, index):
        if self.sampling == "stride":
            start = self._start(length, slot)
            return [start + k * self.context_stride
                    for k in range(self.num_context + self.n_future)]
        # upstream full_seq window (robocasa_random_movement.py:311-320)
        rng = random.Random() if self.train else random.Random(self.seed * 100003 + index)
        frame_id = length - 1
        picks = rng.sample(range(0, frame_id), min(self.num_context, frame_id))
        picks = sorted(picks)
        if len(picks) < self.num_context:
            picks = [picks[0]] * (self.num_context - len(picks)) + picks
        return picks + [frame_id]

    def __getitem__(self, index):
        clip = self.usable[index // self.windows_per_episode]
        slot = index % self.windows_per_episode
        task_class, hdf5_file, demo, length = clip
        indices = self._indices(length, slot, index)

        frames, poses = [], []
        cam_ext = None
        with h5py.File(self.root / task_class / hdf5_file, "r") as handle:
            obs = handle["data"][demo]["obs"]
            images = obs["robot0_robotview_2_image"]
            for step, idx in enumerate(indices):
                frames.append(image_transform(images[idx], self.img_size,
                                              self.patch_size))
                pose, cam_ext_now = process_annotation(obs, idx, cam_ext)
                if step == 0:
                    cam_ext = cam_ext_now
                poses.append(pose)

        poses = np.stack(poses)                       # (T + 1, 44, 3)
        actions = torch.from_numpy(
            deltas_from_poses(poses[:-1].copy(), poses[1:].copy())).float()
        rel_t = np.asarray(np.diff(indices), dtype=np.int64)
        metadata = {"clip": f"{task_class}/{hdf5_file}/{demo}",
                    "start": int(indices[0])}
        return torch.stack(frames), actions, rel_t, 0, 0, metadata
