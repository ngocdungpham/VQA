#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/ailab/Documents/KC4.0_Final_annots_data_png/GEMeX-Project"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data-v1-subset/bronchoscopy_train_data.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/}"
MODEL_PATH="${MODEL_PATH:-BoKelvin/GEMeX-VQA-Model-Simple}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/bronchoscopy_vqa}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
REPORT_TO="${REPORT_TO:-none}"
ENABLE_FSDP="${ENABLE_FSDP:-0}"
LORA_ENABLE="${LORA_ENABLE:-1}"
BITS="${BITS:-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TRAIN_ENTRYPOINT="llava/train/train_mem.py"
if [[ "${USE_FLASH_ATTN}" == "0" ]]; then
    TRAIN_ENTRYPOINT="llava/train/train.py"
fi

cd "${PROJECT_ROOT}/llava-med"
export PYTHONPATH="${PROJECT_ROOT}/llava-med:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export LLAVA_DEFER_VISION_TOWER="${LLAVA_DEFER_VISION_TOWER:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"

VISIBLE_GPU_COUNT=$(CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
if (( NPROC_PER_NODE > VISIBLE_GPU_COUNT )); then
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} > visible GPU count=${VISIBLE_GPU_COUNT}; using ${VISIBLE_GPU_COUNT}."
    NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
fi
if (( NPROC_PER_NODE < 1 )); then
    echo "No visible CUDA GPU found. Set CUDA_VISIBLE_DEVICES before training."
    exit 1
fi

TRAIN_ARGS=()
if [[ "${LORA_ENABLE}" == "1" ]]; then
    TRAIN_ARGS+=(--lora_enable True)
    TRAIN_ARGS+=(--bits "${BITS}")
    TRAIN_ARGS+=(--lora_r "${LORA_R}")
    TRAIN_ARGS+=(--lora_alpha "${LORA_ALPHA}")
    TRAIN_ARGS+=(--lora_dropout "${LORA_DROPOUT}")
else
    TRAIN_ARGS+=(--bits 16)
fi
if [[ "${ENABLE_FSDP}" == "1" ]]; then
    TRAIN_ARGS+=(--fsdp "full_shard auto_wrap")
    TRAIN_ARGS+=(--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer')
fi

echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_PATH=${DATA_PATH}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "USE_FLASH_ATTN=${USE_FLASH_ATTN}"
echo "LORA_ENABLE=${LORA_ENABLE}"
echo "BITS=${BITS}"
echo "TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ENTRYPOINT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 3 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    "${TRAIN_ARGS[@]}"
