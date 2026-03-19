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

DEVICE_IDS="0,1"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)
MASTER_PORT=29508
SERVICE_HOST="0.0.0.0"
SERVICE_PORT=8000

export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== Launch ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} server_edit_2card_current_best.py \
    --host ${SERVICE_HOST} \
    --port ${SERVICE_PORT} \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --lora_path ${LORA_PATH} \
    --warmup_image_paths "${WARMUP_IMAGE}" \
    --num_inference_steps 8 \
    --width 1024 \
    --height 1024 \
    --max_infer_width 1024 \
    --max_infer_height 1024 \
    --cfg_size 1 \
    --ulysses_size 2
