# ====================== Global Env ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1
export OVERLAP=1
export ROPE_FUSE=1
export ADALN_FUSE=1

# Optional cache switches:
# export COND_CACHE=1
# export UNCOND_CACHE=1
# export CACHE_STEP_START=10
# export CACHE_STEP_INTERVAL=3
# export CACHE_STEP_END=35
# export CACHE_BLOCK_START=10
# export CACHE_BLOCK_END=50

# ====================== Service Config ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/root/work/filestorage/cyy/Qwen-Image-Edit-2511"
LORA_PATH="/root/work/filestorage/cyy/Qwen-Image-Edit-2511-Lightning/lora"
WARMUP_IMAGE="/root/work/filestorage/cyy/Qwen-Image-Lightning-scripts/result0306/yarn-art-pikachu.png"

DEVICE_IDS="0"
SERVICE_HOST="0.0.0.0"
SERVICE_PORT=8000

export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== Launch ======================
python server_edit_1card_current_best.py \
    --host ${SERVICE_HOST} \
    --port ${SERVICE_PORT} \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --lora_path ${LORA_PATH} \
    --warmup_image_paths "${WARMUP_IMAGE}" \
    --num_inference_steps 8 \
    --width 1024 \
    --height 1024 \
    --target_infer_area 1048576 \
    --max_infer_area 1048576 \
    --device_id 0
