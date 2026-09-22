#!/usr/bin/env bash
# ============================================================================
# DexWM ctx1 fine-tune -- UR7e + RH5DG2 CUP-GRASP  (DynaGuide comparison WM)
# ============================================================================
# 목적: pi0.5 cup-grasp 체크포인트(DexSteer/pi05-isaaclab-ur7e-cup-grasp-old)와
#   "완전히 같은 데이터셋"으로 DexWM(비교군 DynaGuide용 world model)을 학습한다.
#   데이터: DexSteer/isaaclab-ur7e-cup-grasp-scripted (LeRobot v3.0, 100ep/14485fr/50fps).
#
# ⚠ 기존 3-task(pour_cup/grasp_pan/cup_hang) DexWM 파이프라인은 22-D abs_ee action
#   ( [ee_pos3, ee_rot6d6, hand_active_offsets13] )을 읽는다. cup-grasp 데이터는
#   19-D  ( action=[EE-delta6(스케일드, ≈[-1,1]), hand13(절대 joint 타깃)],
#           state =[arm6, hand13] ) 로 계약이 다르다.  따라서 기존 loader/bridge를
#   그대로 먹일 수 없다.  본 스크립트는 코드 무수정(기존 loader/train 그대로) 원칙을
#   지키기 위해 **Route A**를 쓴다:
#
#     [1] DATA-PREP : cup-grasp(19-D) --> 22-D abs_ee 로 변환한 raw 디렉토리를
#                     기존 `ur7e` loader 가 읽는 레이아웃( {data,meta,videos} )으로 만든다.
#                     ★ state[0:6] 은 EE pose 가 아니라 UR7e ARM JOINT 6개다(메타 이름 +
#                       라디안 레인지로 확정). 따라서 슬라이스가 아니라 FK 를 써야 한다:
#                        p, R        = fk_root(state[0:6])     # core.env_adapters._ur7e_ee_exec
#                        ee_pos(3)   = p                       # tool0, ROBOT ROOT frame
#                        ee_rot6d(6) = encode_rot6d(R)         # R 의 1,2열 (= layout "columns")
#                        hand_off(13)= state[6:19] - default_active,  default_active = zeros(13)
#                                      (cup-grasp hand13 순서 == ACTIVE_HAND_JOINTS, 순열 불필요)
#                     -> 변환기: datasets/cup_grasp_to_abs_ee.py  (핸드오프 §3에서 구현)
#     [2] TRAIN     : train_dexjoco_ctx1.py --dataset ur7e  (기존 코드 그대로) 로 학습.
#
#   ‼ 착수 전 검증 게이트는 핸드오프 §2 에서 이미 실측으로 해소됐다. 다른 서버에서 데이터
#     복사본이 동일한지만 §2 의 재현 커맨드로 1회 확인하고 넘어갈 것.
#
# 하이퍼파라미터는 3-task ctx1_isaac 런과 동일(재현): lr 3e-5, batch 8, 6000 step,
#   num_context 1, amp off, stride 5, init=robocasa_random_finetune. (task당 ~2.5h)
# ----------------------------------------------------------------------------
set -euo pipefail

# ---- paths / env (다른 서버면 override) -----------------------------------
DEXWM_REPO="${DEXWM_REPO:-$(cd "$(dirname "$0")" && pwd)}"        # 이 스크립트가 있는 dexwm 레포
VLS_CODE_ROOT="${VLS_CODE_ROOT:-/workspace/vls/code}"            # bridge(core/dexwm_rh5dg2_action_adapter) 위치
PYTHON="${PYTHON:-/opt/conda/envs/vls_openpi/bin/python}"        # 3-task 학습에 쓰인 env (다른 서버면 dexwm env로)
export VLS_CODE_ROOT
export DINOV2_HUB_DIR="${DINOV2_HUB_DIR:-/root/.cache/torch/hub/facebookresearch_dinov2_main}"

TASK="${TASK:-cup_grasp}"
# cup-grasp 원본 LeRobot(로컬 복사본; CIFS 직독 회피) 과 변환 산출물 위치
CUP_LEROBOT="${CUP_LEROBOT:-/workspace/vls/datasets/lerobot/isaaclab-ur7e-cup-grasp-scripted}"
ABS_EE_ROOT="${ABS_EE_ROOT:-/root/dexwm_ur7e/raw/${TASK}_abs_ee}"     # [1] 변환 산출물
FRAMES_ROOT="${FRAMES_ROOT:-/root/dexwm_ur7e/frames/${TASK}}"         # (옵션) JPEG 프레임 캐시
# RH5DG2 default_active(13)/full(18). UR7E_RH5DG2_CFG 는 18개 손가락 관절을 전부 0.0 으로
# 두므로(= zeros) 이 파일은 선택사항이다. 없으면 --defaults-npz 를 아예 넘기지 않고,
# adapter 의 기본값(zeros(13))을 그대로 쓴다. (경로만 주고 파일이 없으면 np.load 가 죽는다.)
DEFAULTS_NPZ="${DEFAULTS_NPZ:-/root/dexwm_ur7e/rh5dg2_defaults.npz}"
INIT_CKPT="${INIT_CKPT:-/root/ckpt_local/dexwm/robocasa_random_finetune.pth.tar}"
OUT_DIR="${OUT_DIR:-/root/ckpt_local/dexwm/ctx1_isaac/${TASK}}"
MIRROR_DIR="${MIRROR_DIR:-/workspace/vls/code/outputs/dexwm/ctx1_isaac/${TASK}}"  # 작은 텍스트 로그 NAS 미러

# ---- hyperparams (3-task 재현) --------------------------------------------
NUM_CONTEXT="${NUM_CONTEXT:-1}"; BATCH="${BATCH:-8}"; ACCUM="${ACCUM:-1}"
LR="${LR:-3e-5}"; MAX_STEPS="${MAX_STEPS:-6000}"; STRIDE="${STRIDE:-5}"
AMP="${AMP:-off}"; SEED="${SEED:-0}"; HAND_SCALE="${HAND_SCALE:-1.0}"; ROT6D_LAYOUT="${ROT6D_LAYOUT:-columns}"
# cup-grasp 데이터는 3-task의 ~22% (window pool 12578 vs 57566)라 default --patience 5 가
# 6000 step 전에 조기종료할 수 있다(핸드오프 §4). 3-task는 전부 max_steps 완주했으므로
# apples-to-apples 위해 크게 둔다(best.pth.tar 는 어차피 best val 로 추적됨). 조기종료 원하면 override.
PATIENCE="${PATIENCE:-1000}"

# defaults npz 는 "있을 때만" 넘긴다 (train_dexjoco_ctx1.py 는 경로가 주어지면 무조건 np.load).
DEFAULTS_ARG=""
if [[ -f "$DEFAULTS_NPZ" ]]; then DEFAULTS_ARG=1; else
  echo "[defaults] $DEFAULTS_NPZ 없음 -> default_active=zeros(13) (UR7E_RH5DG2_CFG 와 동일) 사용"
fi
# world_to_camera 는 ur7e 경로에서 optional 이며, 넘기면 npz 를 반드시 읽는다.
# cup-grasp 용 third_person extrinsic 이 없으므로 플래그 자체를 넘기지 않는다(-> root frame).

mkdir -p "$OUT_DIR" "$MIRROR_DIR"
LOG="$OUT_DIR/${TASK}.log"

# ---------------------------------------------------------------------------
# [1] DATA-PREP 게이트 : abs_ee raw 가 없으면 변환기를 돌리라고 안내하고 중단.
#     (변환기는 코드 무수정 원칙상 기존 loader 를 못 건드리므로 별도 신규 파일이다.)
# ---------------------------------------------------------------------------
if [[ ! -d "$ABS_EE_ROOT/data" || ! -f "$ABS_EE_ROOT/meta/info.json" ]]; then
  echo "[data-prep] abs_ee raw 없음: $ABS_EE_ROOT"
  CONV="$DEXWM_REPO/datasets/cup_grasp_to_abs_ee.py"
  if [[ -f "$CONV" ]]; then
    echo "[data-prep] 변환 실행: $CONV"
    "$PYTHON" "$CONV" \
      --src "$CUP_LEROBOT" \
      --out "$ABS_EE_ROOT" \
      ${DEFAULTS_ARG:+--defaults-npz "$DEFAULTS_NPZ"} \
      --frames-root "$FRAMES_ROOT"        # 프레임 캐시도 함께(export). 구현은 핸드오프 §3.
  else
    echo "‼ 변환기 미구현: $CONV"
    echo "  handoff_dexwm_cup_grasp.md §2.3 재현 커맨드 1회 실행 후, §3 대로 cup_grasp_to_abs_ee.py 를 만들고 다시 실행."
    echo "  (기존 datasets/ur7e_rh5dg2_demos.py / train_dexjoco_ctx1.py 는 절대 수정하지 말 것.)"
    exit 2
  fi
fi

# 프레임 캐시가 없고 변환기가 안 만들었으면, 기존 export 도구로 생성(av1 디코드 회피 최적화; 옵션).
FRAMES_ARG=()
if [[ -d "$FRAMES_ROOT" ]]; then FRAMES_ARG=(--frames-root "$FRAMES_ROOT"); fi

# ---------------------------------------------------------------------------
# [1.5] INIT-CKPT 가드 : 핸드오프 §1 "없으면 멈추고 보고"를 실행 경로에서 강제.
#   train_dexjoco_ctx1.py:448-451 은 --init-ckpt 파일이 없으면 경고만 찍고
#   args.init_ckpt=None 으로 **조용히 random init** 진행한다 -> 비교 오염. 여기서 하드-실패.
#   (진짜 random init 이 목적이면 ALLOW_RANDOM_INIT=1 로 명시적 opt-in.)
# ---------------------------------------------------------------------------
if [[ ! -f "$INIT_CKPT" && "${ALLOW_RANDOM_INIT:-0}" != "1" ]]; then
  echo "‼ init ckpt 없음: $INIT_CKPT"
  echo "  이대로 두면 train_dexjoco_ctx1.py 가 random init 으로 조용히 떨어져 3-task 대비 비교가 오염됨."
  echo "  robocasa_random_finetune.pth.tar 를 이 서버로 가져온 뒤 재실행하거나(권장),"
  echo "  정말 random init 을 원하면 ALLOW_RANDOM_INIT=1 로 재실행. -> 여기서 멈추고 보고."
  exit 3
fi

# ---------------------------------------------------------------------------
# [2] TRAIN : 기존 train_dexjoco_ctx1.py --dataset ur7e 그대로.
# ---------------------------------------------------------------------------
echo "[train] task=$TASK  steps=$MAX_STEPS batch=$BATCH lr=$LR ctx=$NUM_CONTEXT amp=$AMP -> $OUT_DIR"
"$PYTHON" "$DEXWM_REPO/train_dexjoco_ctx1.py" \
  --dataset ur7e \
  --task "$TASK" \
  --task-data-root "$ABS_EE_ROOT" \
  --init-ckpt "$INIT_CKPT" \
  ${DEFAULTS_ARG:+--defaults-npz "$DEFAULTS_NPZ"} \
  --hand-scale "$HAND_SCALE" \
  --rot6d-layout "$ROT6D_LAYOUT" \
  --num-context "$NUM_CONTEXT" \
  --batch "$BATCH" --accum "$ACCUM" --lr "$LR" \
  --max-steps "$MAX_STEPS" --context-stride "$STRIDE" \
  --patience "$PATIENCE" \
  --amp "$AMP" --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  "${FRAMES_ARG[@]}" \
  2>&1 | tee "$LOG"

# ---- 작은 텍스트 로그 NAS 미러 (curve/summary/log) -------------------------
for f in curve.csv curve.jsonl summary.json "${TASK}.log"; do
  [[ -f "$OUT_DIR/$f" ]] && cp -f "$OUT_DIR/$f" "$MIRROR_DIR/$f" || true
done
echo "[done] ckpt: $OUT_DIR/best.pth.tar   mirror: $MIRROR_DIR"
echo "[next] goal bank: comparisons/dynaguide/goal_latents.py --embodiment isaaclab ... (핸드오프 §5)"
