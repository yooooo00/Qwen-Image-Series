# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=1            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/root/work/filestorage/cyy/Qwen-Image-Edit-2511"
DEVICE_IDS="0,1"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
# 2卡并行(cfg_size=2 ulysses_size=1 优于 cfg_size=1 ulysses_size=2 )
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "/root/work/filestorage/cyy/Qwen-Image-Lightning-scripts/result0306/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 8 \
    --seed 42 \
    --output_file "./output/image_edit_2511_.png" \
    --lora_path /root/work/filestorage/cyy/Qwen-Image-Edit-2511-Lightning/lora \
    --cfg_size 2 \
    --ulysses_size 1