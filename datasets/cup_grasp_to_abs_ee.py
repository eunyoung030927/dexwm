"""cup-grasp (19-D) -> 22-D abs_ee 변환기 (DexWM ctx1 학습 Route A).

DexSteer/isaaclab-ur7e-cup-grasp-scripted 는
    observation.state[19] = [UR7e arm joint 6, RH5DG2 active hand joint 13]
    action[19]            = [EE-delta 6(스케일드), hand 13(절대 타깃)]
로, 기존 3-task DexWM loader(datasets/ur7e_rh5dg2_demos.py)가 읽는
    action[22] = [ee_pos(3), ee_rot6d(6), hand_active_offsets(13)]  (절대 tool0, root frame)
계약과 다르다.  기존 loader/train/bridge 무수정 원칙을 지키기 위해, 데이터를 abs_ee 계약으로
재작성해 기존 `--dataset ur7e` 경로를 그대로 태운다.

핵심 (핸드오프 §2 실측으로 확정):
  * state[0:6] 은 EE pose 가 아니라 UR7e ARM JOINT 6개 → tool0 pose 는 FK 로 얻는다
    (core.env_adapters._ur7e_ee_exec.fk_root, ROBOT ROOT frame; bridge 가 기대하는 프레임).
  * ee_rot6d 는 "columns" = [R[:,0], R[:,1]] (rot6d_to_matrix 가 Gram-Schmidt 로 정확 복원, max|dR|~3e-18).
  * hand_off = state[6:19] - default_active.  bridge 가 q = default_active + hand_scale*offset 로
    되돌리므로, 이 변환기와 학습이 **같은 default_active** 만 쓰면 achieved joint 를 정확 복원한다
    (default 값 자체는 상쇄; cup-grasp defaults = zeros(13)).  hand13 순서 == ACTIVE_HAND_JOINTS.

Route A 는 delta(action[0:6])를 쓰지 않는다 → 핸드오프 §4.2 의 commanded↔achieved 상관 붕괴와 무관.
단 학습 대상이 achieved pose 라, 3-task 의 commanded 계약과는 다르다(각주: 추론 정합은 §6).

행 순서/개수를 보존해 meta 의 dataset_from/to_index 정합을 유지한다 (재정렬 금지).

사용:
    python cup_grasp_to_abs_ee.py \
        --src /workspace/vls/datasets/lerobot/isaaclab-ur7e-cup-grasp-scripted \
        --out /root/dexwm_ur7e/raw/cup_grasp_abs_ee \
        [--defaults-npz /root/dexwm_ur7e/rh5dg2_defaults.npz] \
        [--frames-root /root/dexwm_ur7e/frames/cup_grasp]      # 있으면 JPEG 캐시도 생성
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _vls_root() -> str:
    root = os.environ.get("VLS_CODE_ROOT", "/workspace/vls/code")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _encode_rot6d_columns(R: np.ndarray) -> np.ndarray:
    """columns layout: rot6d = [R[:,0], R[:,1]] (bridge.rot6d_to_matrix 의 역)."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def build_abs_ee(states: np.ndarray, default_active: np.ndarray) -> np.ndarray:
    """(N,19) state -> (N,22) abs_ee. state[0:6]=arm joint, state[6:19]=active hand joint."""
    from core.env_adapters._ur7e_ee_exec import fk_root

    n = states.shape[0]
    out = np.zeros((n, 22), dtype=np.float32)
    for i in range(n):
        pos, R = fk_root(states[i, 0:6])
        out[i, 0:3] = pos
        out[i, 3:9] = _encode_rot6d_columns(np.asarray(R, dtype=np.float64))
        out[i, 9:22] = states[i, 6:19] - default_active
    return out


def rewrite_parquets(src: Path, out: Path, default_active: np.ndarray) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((src / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no data parquet under {src/'data'}")
    total = 0
    for p in paths:
        t = pq.read_table(p)
        if "action" not in t.column_names or "observation.state" not in t.column_names:
            raise ValueError(f"{p}: missing action/observation.state")
        states = np.asarray(t["observation.state"].to_pylist(), dtype=np.float32)
        if states.shape[1] != 19:
            raise ValueError(f"{p}: expected state[.,19], got {states.shape}")
        abs22 = build_abs_ee(states, default_active)          # (N,22), row order preserved
        new_action = pa.array(abs22.tolist(), type=pa.list_(pa.float32(), 22))
        idx = t.column_names.index("action")
        field = pa.field("action", new_action.type)
        t = t.set_column(idx, field, new_action)
        rel = p.relative_to(src)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst)
        total += len(states)
        print(f"[parquet] {rel}  rows={len(states)} -> action[22]")
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="cup-grasp LeRobot dir (원본; videos/meta 보유)")
    ap.add_argument("--out", type=Path, required=True, help="abs_ee raw 출력 (기존 ur7e loader 가 읽음)")
    ap.add_argument("--defaults-npz", type=Path, default=None,
                    help="default_active(13). 생략 시 zeros(13) (cup-grasp env 와 동일).")
    ap.add_argument("--frames-root", type=Path, default=None,
                    help="주면 export_ur7e_frames.py 로 third_person JPEG 캐시도 생성.")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    _vls_root()

    default_active = np.zeros(13, dtype=np.float32)
    if args.defaults_npz is not None:
        with np.load(args.defaults_npz, allow_pickle=False) as d:
            default_active = np.asarray(d["default_active"], dtype=np.float32)
    if default_active.shape != (13,):
        raise ValueError(f"default_active must be (13,), got {default_active.shape}")
    print(f"[defaults] default_active all-zero={bool(np.allclose(default_active,0))}")

    # meta 는 그대로 복사 (info.json + episodes/*: dataset_from/to_index, video refs).
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "meta").exists():
        shutil.rmtree(args.out / "meta")
    shutil.copytree(args.src / "meta", args.out / "meta")
    print(f"[meta] copied {args.src/'meta'} -> {args.out/'meta'}")

    total = rewrite_parquets(args.src, args.out, default_active)
    print(f"[done] abs_ee raw: {args.out}  total_rows={total}")

    # 프레임 캐시(옵션): 원본 src 의 videos 에서 디코드 (out 엔 videos 를 복사하지 않는다 = 디스크 절약).
    if args.frames_root is not None:
        exporter = Path(__file__).with_name("export_ur7e_frames.py")
        cmd = [sys.executable, str(exporter), "--src", str(args.src),
               "--out", str(args.frames_root), "--jpeg-quality", str(args.jpeg_quality)]
        print("[frames] " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[frames] cache: {args.frames_root}")


if __name__ == "__main__":
    main()
