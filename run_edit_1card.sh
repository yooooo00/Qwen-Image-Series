# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1
export OVERLAP=1
export ROPE_FUSE=1
export ADALN_FUSE=1
# export COND_CACHE=1
# export UNCOND_CACHE=1

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/root/work/filestorage/cyy/Qwen-Image-Edit-2511"
DEVICE_IDS="0"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "/root/work/filestorage/cyy/Qwen-Image-Lightning-scripts/result0306/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 8 \
    --seed 42 \
    --output_file "./output/image_edit_2511_1card.png" \
    --lora_path /root/work/filestorage/cyy/Qwen-Image-Edit-2511-Lightning/lora
