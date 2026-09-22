# 핸드오프 — DexWM ctx1 fine-tune on UR7e+RH5DG2 **cup-grasp**

> 수신: 다른 서버에서 이 작업을 이어받는 에이전트
> 동봉: `dexwm/train_ctx1_cup_grasp.sh` (런처, 이미 작성·패치 완료)
> 작성일: 2026-09-15 / 실측 환경: RTX 3090 24GB, conda `vls_openpi`

---

## §0. 임무와 경계

**임무.** pi0.5 cup-grasp 체크포인트(`DexSteer/pi05-isaaclab-ur7e-cup-grasp-old`)와 **완전히 같은 데이터셋**으로
DynaGuide 비교군용 world model(DexWM, `num_context=1`)을 학습한다.

**산출물.**
1. `dexwm/datasets/cup_grasp_to_abs_ee.py` — 신규 변환기 (§3). **네가 구현할 유일한 코드.**
2. `/root/ckpt_local/dexwm/ctx1_isaac/cup_grasp/best.pth.tar` + `curve.csv` / `summary.json`.

**절대 수정 금지.** 아래는 3-task 런과의 비교 가능성을 유지하기 위해 byte-for-byte 보존한다.
- `dexwm/datasets/ur7e_rh5dg2_demos.py`
- `dexwm/train_dexjoco_ctx1.py`, `train_dexjoco_ft.py`, `train_dexjoco_ms.py`
- `dexwm/models/**`
- `code/core/dexwm_rh5dg2_action_adapter.py`, `code/core/geometry/rh5dg2_kinematics.py`

계약 차이는 **전부 데이터 쪽에서** 흡수한다(Route A). 코드로 흡수하려 들지 말 것.

---

## §1. 전제 환경

| 항목 | 값 |
|---|---|
| python | `/opt/conda/envs/vls_openpi/bin/python` (3.11, torch 2.7.1+cu126) |
| 필요 패키지 | torch / torchvision / timm / transformers / einops / **av** / **pyarrow** / cv2 / h5py — 전부 설치 확인됨 |
| 불필요 | `decord`, `smplx`, `xformers`, `submitit` (egodex/droid/robocasa 로더·submitit 런처 전용. 이 경로에서 import 되지 않음) |
| `VLS_CODE_ROOT` | `/workspace/vls/code` (bridge + UR7e FK 위치) |
| `DINOV2_HUB_DIR` | `/root/.cache/torch/hub/facebookresearch_dinov2_main` (+ `checkpoints/dinov2_vits14_pretrain.pth`) |
| 원본 데이터 | `/workspace/vls/datasets/lerobot/isaaclab-ur7e-cup-grasp-scripted` (LeRobot v3.0) |
| GPU | 24GB면 충분 (ctx1 + batch 8 + fp32) |

**다른 서버에서 먼저 확인할 것**
- 원본 데이터 경로가 다르면 `CUP_LEROBOT=` 로 override.
- `INIT_CKPT` (`/root/ckpt_local/dexwm/robocasa_random_finetune.pth.tar`) 가 **없으면** 학습은 죽지 않고
  DINOv2-frozen random init 으로 조용히 떨어진다(`train_dexjoco_ctx1.py` 가 경고만 출력).
  3-task 런과의 비교를 하려면 **반드시 같은 init 을 확보**하고 시작할 것. 없으면 거기서 멈추고 보고.
- 디스크: 변환 산출물(parquet 재작성 + 비디오 하드링크/심링크)은 작지만, 프레임 캐시는 수 GB.

---

## §2. 데이터 계약 — 실측 결과 (착수 게이트는 **이미 해소됨**)

원래 이 핸드오프는 "state[0:6]이 EE pose인지 arm joint인지 확정하라"는 게이트를 걸 예정이었다.
**로컬 parquet + LeRobot 메타 + IsaacLab articulation cfg 로 전부 확정했으므로 재조사 불필요하다.**
아래는 결론과, 다른 서버 복사본이 동일한지 1회 확인할 재현 커맨드다.

### 2.1 계약 비교

| | 3-task (`*_abs_ee`, 기존 파이프라인) | cup-grasp (신규) |
|---|---|---|
| `action` | **22-D** `[ee_pos3, ee_rot6d6, hand_active_offsets13]` (절대) | **19-D** `[ee_delta6(스케일드, ≈[-1,1]), hand13(절대 joint 타깃)]` |
| `observation.state` | **24-D** `[arm6, hand_full18]` (IsaacLab actuator 순서) | **19-D** `[arm6, hand_active13]` |
| 규모 | cup_hang 기준 200 ep / 98,351 fr | **100 ep / 14,485 fr** (길이 134–157, median 144) |
| fps / 코덱 | 50 / av1 | 50 / av1 (`third_person` + `eye_in_hand`, 480×640) |

### 2.2 확정된 사실 4가지 (각각 근거 포함)

**(a) `state[0:6]` 은 EE pose 가 아니라 UR7e ARM JOINT 6개다.**
`meta/info.json` 의 feature names 가 `shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint,
wrist_2_joint, wrist_3_joint` 이고, 값 범위도 라디안(예: ch0 ∈ [0, 1.104])이지 미터가 아니다.
→ **슬라이스가 아니라 FK 가 필요하다.** 런처 헤더의 옛 설명(`ee_pos = state[0:3]`)은 이 사실로 교정해 두었다.

**(b) UR7e FK 는 이미 레포에 있다. 새로 쓰지 마라.**
`code/core/env_adapters/_ur7e_ee_exec.py`:
- `fk_root(q6) -> (p(3,), R(3,3))` — tool0 pose, **ROBOT ROOT frame**. 3-task `abs_ee` 가 쓰는 바로 그 프레임
  (`dexwm_rh5dg2_action_adapter` docstring: "ABSOLUTE tool0 pose (root frame)").
- `encode_rot6d(R) -> (6,)` — R 의 **1·2열** concat. 즉 `--rot6d-layout columns` 와 일치.
- 내부적으로 `_ur7e_kinematics.fk_urdf` (analytic, pure numpy) + `RZPI` 프레임 보정을 쓴다.

실측: ep0(142프레임)에 FK 적용 시 tool0 위치가 `x∈[0.162,0.817] y∈[0.233,0.740] z∈[-0.054,0.357]`,
총 이동 0.82 m — UR7e 워크스페이스로 물리적으로 타당하다.

**(c) `default_active = zeros(13)`, `default_full = zeros(18)`.**
`isaac-tasks/.../robots/ur7e_rh5dg2.py` 의 `UR7E_RH5DG2_CFG.init_state.joint_pos` 가
`"right_.*_joint": 0.0` (주석: "all 18 finger joints open") 으로 18개 손가락 관절 전부 0.
데이터도 일치한다 — ep0 frame 0 의 hand 13채널이 정확히 전부 0이고 거기서부터 증가한다.
→ **`--defaults-npz` 는 넘길 필요가 없다.** adapter 가 `default_active=None` 일 때 `zeros(13)` 을 쓴다
(`dexwm_rh5dg2_action_adapter.py:169-171`). 경로만 주고 파일이 없으면 `np.load` 가 죽으니 주의.

**(d) cup-grasp 의 hand 13 순서 == `ACTIVE_HAND_JOINTS` 순서. 순열 불필요.**
둘 다
`thumb_yaw, thumb_mcp, thumb_dip, index_yaw, index_mcp, index_pip, middle_yaw, middle_mcp, middle_pip, ring_mcp, ring_pip, pinky_mcp, pinky_pip`.
→ 3-task 의 `STATE_HAND_JOINT_ORDER`(18-D, actuator 순서) 재정렬 로직은 **cup-grasp 에 적용하면 안 된다.**
그건 3-task state 전용이다. cup-grasp 은 그대로 복사하면 된다.

### 2.3 재현 커맨드 (다른 서버에서 1회 실행, 5초 — 작성 서버에서 통과 확인됨)

```bash
cd /workspace/vls/code && VLS_CODE_ROOT=$PWD /opt/conda/envs/vls_openpi/bin/python - <<'EOF'
import sys, glob, numpy as np, pyarrow.parquet as pq, json
sys.path.insert(0, '/workspace/vls/code')
from core.env_adapters._ur7e_ee_exec import fk_root, encode_rot6d
from core.geometry.rh5dg2_kinematics import ACTIVE_HAND_JOINTS

SRC = '/workspace/vls/datasets/lerobot/isaaclab-ur7e-cup-grasp-scripted'
info = json.load(open(f'{SRC}/meta/info.json'))
names = info['features']['observation.state']['names']
assert names[:6] == ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                     'wrist_1_joint','wrist_2_joint','wrist_3_joint'], names[:6]   # (a)
assert tuple(names[6:]) == ACTIVE_HAND_JOINTS, names[6:]                            # (d)
assert info['features']['action']['shape'] == [19]

t = pq.read_table(sorted(glob.glob(f'{SRC}/data/**/*.parquet', recursive=True))[0])
s = np.stack(t['observation.state'].to_numpy(zero_copy_only=False))
assert np.allclose(s[0], 0), s[0]                                                   # (c)
P = np.stack([fk_root(q)[0] for q in s[:142, :6]])
print('OK  ee_pos min', P.min(0).round(3), 'max', P.max(0).round(3))
print('OK  rot6d dim', encode_rot6d(fk_root(s[10,:6])[1]).shape)
EOF
```
세 `assert` 가 전부 통과하면 §3 으로 간다. 하나라도 깨지면 **데이터 복사본이 다른 것**이므로
변환기를 쓰지 말고 즉시 보고할 것.

---

## §3. 구현 대상 — `dexwm/datasets/cup_grasp_to_abs_ee.py`

### 3.1 핵심 통찰: 비디오와 메타는 손대지 않는다

`ur7e_rh5dg2_demos.py` 가 소스 디렉토리에서 실제로 읽는 것은 딱 셋이다.

| 읽는 것 | 어디서 | cup-grasp 에서 |
|---|---|---|
| `fps`, `video_path` | `meta/info.json` | **그대로 사용 가능** |
| `episode_index / length / dataset_from_index / dataset_to_index` 와 `videos/observation.images.third_person/{chunk_index,file_index,from_timestamp}` | `meta/episodes/**/*.parquet` | **컬럼 전부 존재. 그대로 사용 가능** |
| `observation.state`, `action` | `data/**/*.parquet` | `action` 만 22-D 로 재작성 필요 |

`__getitem__` 은 `self.actions` 만 쓰고 `self.states` 는 읽어만 두고 안 쓴다 → **`observation.state` 는 19-D 그대로 둬도 된다.**
`stats.json` 은 로더가 읽지 않는다.

→ **변환기가 할 일: `meta/` 와 `videos/` 는 심링크(또는 하드링크), `data/` parquet 만 `action` 컬럼 교체.**
비디오 재인코딩 없음, 메타 재생성 없음.

### 3.2 변환식 (프레임 t마다)

```python
p, R   = fk_root(state[t, 0:6])              # tool0, ROBOT ROOT frame
action22[t] = np.concatenate([
    p,                                        # [0:3]   ee_pos
    encode_rot6d(R),                          # [3:9]   ee_rot6d, columns layout
    state[t, 6:19] - default_active,          # [9:22]  hand offsets, default_active = zeros(13)
]).astype(np.float32)
```
`default_active = zeros(13)` 이므로 hand 블록은 사실상 `state[t, 6:19]` 복사다.
그래도 `--defaults-npz` 를 받아 빼주는 코드 경로는 남겨둘 것(다른 embodiment 재사용 대비).

### 3.3 CLI

`train_ctx1_cup_grasp.sh` 가 이렇게 호출한다. 시그니처를 맞출 것.

```
--src <원본 LeRobot 루트>  --out <abs_ee 산출물 루트>
[--defaults-npz <npz: default_active(13), default_full(18)>]   # 선택
[--frames-root <JPEG 프레임 캐시 출력 경로>]                     # 선택
```

`--frames-root` 가 주어지면 변환 후 프레임 캐시까지 만들어라. **직접 디코더를 짜지 말고**
기존 `dexwm/datasets/export_ur7e_frames.py` 를 재사용한다:
```bash
VLS_CODE_ROOT=/workspace/vls/code python datasets/export_ur7e_frames.py \
    --src <out>  --out <frames-root>  --jpeg-quality 95
```
(이 도구는 `<src>` 를 `*_abs_ee` 레이아웃으로 가정하므로 **변환 산출물**을 가리켜야 한다.
비디오가 심링크여도 `av` 가 따라간다.)

### 3.4 변환기 자체 검증 (변환기 안에 넣고, 실패 시 non-zero exit)

1. `action22.shape == (N, 22)`, dtype float32, NaN/Inf 없음.
2. 랜덤 20프레임에 대해 `decode_rot6d(action22[t,3:9])` 가 정규직교(‖RᵀR − I‖∞ < 1e-5, det > 0).
3. **왕복 검증**: `dexwm_rh5dg2_action_deltas(action22[t:t+2], RH5DG2Kinematics())` 가
   예외 없이 `(pos, quat_wxyz, q_full(18))` 을 돌려주고, `q_full` 이 `ACTIVE_JOINT_LIMITS` 범위 안.
4. **로더 스모크**: 변환 산출물로 아래가 통과해야 한다.
   ```bash
   VLS_CODE_ROOT=/workspace/vls/code python datasets/ur7e_rh5dg2_demos.py <out>
   ```
   이 `__main__` 스모크는 `num_context=4, n_future=1, stride=5, windows_per_episode=4` 로 돈다.
   100 ep 중 `val_every=10` 이 10개를 held-out 으로 빼므로 기대 출력은
   `episodes: 90  len: 360  frames: (5, 3, 224, 392)  actions: (4, 132)  rel_t: (4,)`.
   (실제 학습은 `num_context=1` 이라 window 가 훨씬 짧다 — 이건 로더 배선 확인용일 뿐이다.)

---

## §4. 아직 열려 있는 판단 항목 (구현 전에 읽을 것)

### 4.1 commanded vs achieved — 의도적 이탈이며, 문서에 남길 것

3-task `abs_ee` action 은 **commanded** 절대 tool0 pose다. Route A 가 만드는 것은
`state` FK 에서 나온 **achieved** pose다. 이건 계약상 이탈이다.

이렇게 가도 된다고 판단한 근거: DexWM 의 action 은 "현재 프레임 → 다음 프레임 사이의 keypoint delta"로
소비되며(`ur7e_rh5dg2_demos.__getitem__` → `dexwm_rh5dg2_action_deltas`), 예측 대상이 **실제 다음 프레임의
DINOv2 latent** 이므로 achieved 쪽이 오히려 프레임과 정합적이다. adapter docstring 도 "realized-state
ablation path" 를 명시적으로 상정하고 있다.

**그러나 3-task 런과 나란히 표에 올릴 때 이 차이는 반드시 각주로 달아야 한다.** `summary.json` 에
`"action_source": "achieved_state_fk"` 같은 키를 남겨라.

### 4.2 검증 실패한 항목 하나 — 정직하게 기록

`action[0:6]`(스케일드 delta)을 `POS_SCALE=0.03` / `ROT_SCALE=0.05` 로 풀어 연속 프레임 FK delta 와
비교했더니 채널별 상관이 `[-0.66, 0.90, -0.01, -0.18, 0.83, -0.21]` 로 절반이 맞지 않았다.

이것이 **의미하는 것**: 기록된 command 로부터 절대 pose 를 **적분해서 복원하는 경로(Route B)는 신뢰할 수 없다.**
이것이 **의미하지 않는 것**: Route A 가 틀렸다는 뜻은 아니다. Route A 는 delta 를 전혀 쓰지 않는다.

관측된 원인 후보(미확정): ep0 141프레임 중 **79프레임(56%)이 ±1 에 포화**되어 있다 — scripted 수집기가
bang-bang 에 가깝게 명령했고, 포화 구간에서는 achieved ≠ commanded 다. 여기에 actuator lag 이 더해진다.
**Route B 로 우회하고 싶어지면 이 수치를 먼저 설명할 수 있어야 한다.**

### 4.3 `world_to_camera` 없음 → root frame 학습

cup-grasp 용 `third_person` extrinsic npz 가 없다(`/root/dexwm_ft/w2c/` 에는 DexJoCo 태스크 것만 있다).
`--world-to-camera` 를 생략하면 keypoint 가 root frame 에 남는다. delta 는 프레임 간 일관되므로 학습은 되지만,
RoboCasa init 이 기대하는 camera frame 은 아니다. **3-task ctx1 런이 w2c 를 썼는지 확인하고 맞춰라.**
안 맞으면 비교가 오염된다.

### 4.4 데이터 규모

cup-grasp 은 3-task 대비 프레임 수가 ~15% 다. 학습 가능 episode 90 / held-out 10
(로더가 `val_every=10` 으로 10번째마다 held-out). `windows_per_episode=40` → epoch 당 3,600 window,
batch 8 → **450 step/epoch**, 6000 step ≈ 13.3 epoch. `--patience 5` 조기종료가 6000 step 전에 걸릴 가능성이
3-task 보다 높다. 조기종료가 1,000 step 이전에 걸리면 과소적합을 의심하고 보고할 것.

---

## §5. 학습 실행

```bash
cd /workspace/vls/dexwm
bash train_ctx1_cup_grasp.sh
```

런처가 하는 일: abs_ee 산출물이 없으면 §3 변환기를 돌리고(없으면 exit 2), 그 다음
`train_dexjoco_ctx1.py --dataset ur7e` 를 **수정 없이** 호출한다. override 는 전부 환경변수:
`CUP_LEROBOT / ABS_EE_ROOT / FRAMES_ROOT / INIT_CKPT / OUT_DIR / PYTHON / BATCH / LR / MAX_STEPS ...`

**하이퍼파라미터 (3-task ctx1_isaac 런 재현, 변경 금지)**

| | 값 |
|---|---|
| `--num-context` | 1 |
| `--batch` / `--accum` | 8 / 1 |
| `--lr` | 3e-5 |
| `--max-steps` | 6000 |
| `--context-stride` | 5 |
| `--amp` | off (fp32 — guidance 가 `autograd.grad` 를 통과시켜야 하므로 bf16 금지) |
| `--rot6d-layout` | columns |
| `--hand-scale` | 1.0 |
| init | `robocasa_random_finetune.pth.tar` |

기대치: **~2.5h / best held-out latent MSE ≈ 0.25**. 크게 벗어나면 멈추고 보고.

---

## §6. 완료 기준

- [ ] §2.3 재현 커맨드 3개 assert 통과
- [ ] `cup_grasp_to_abs_ee.py` 의 §3.4 자체 검증 4개 통과
- [ ] `datasets/ur7e_rh5dg2_demos.py <out>` 스모크가 `episodes: 90  len: 360` 출력
- [ ] 학습 완료, `best.pth.tar` + `curve.csv` + `summary.json` 생성
- [ ] `summary.json` 에 `action_source` 표기(§4.1)와 w2c 사용 여부(§4.3) 기록
- [ ] 금지 파일 목록(§0) `git diff` 가 비어 있음

---

## §7. 이미 밟은 지뢰 (런처에서 수정 완료 — 되돌리지 말 것)

1. **`--world-to-camera None` 금지.** `type=Path` 라 `Path("None")` 이 되고 `np.load` 가 FileNotFoundError 로 죽는다.
   ur7e 경로에서는 optional 이므로 **플래그 자체를 생략**해야 한다.
2. **`--defaults-npz` 를 무조건 넘기면 죽는다.** 파일이 존재할 때만 넘긴다(런처의 `DEFAULTS_ARG` 게이트).
3. **`CUP_LEROBOT` 기본값**이 `/workspace/datasets/...` 였으나 실제 경로는 `/workspace/vls/datasets/...` 다.
4. `datasets` 라는 이름은 HF `datasets` 패키지와 충돌한다. `import datasets.xxx` 가 아니라
   레포 관례대로 `importlib.util.spec_from_file_location` 로 파일 경로 로드할 것.
5. 원본 hdf5(`isaaclab-ur7e-cup-grasp-scripted_raw*.hdf5`)는 **3개 전부 손상**되어 h5py 로 안 열린다.
   LeRobot parquet 만이 유효한 소스다. hdf5 를 쳐다보지 마라.
