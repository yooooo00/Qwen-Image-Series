# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1             # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关
export ROPE_FUSE=1         # RoPE算子融合
export ADALN_FUSE=1        # ADALN算子融合

# export COND_CACHE=1          # 条件缓存
# export UNCOND_CACHE=1        # 无条件缓存
# export CACHE_STEP_START=10   # 缓存开始步骤
# export CACHE_STEP_INTERVAL=3 # 缓存步骤间隔
# export CACHE_STEP_END=35     # 缓存结束步骤
# export CACHE_BLOCK_START=10  # 缓存开始块
# export CACHE_BLOCK_END=50    # 缓存结束块

# # # ====================== 任务配置 ======================
TASK="Qwen-Image"
MODEL_PATH="/root/work/filestorage/cyy/Qwen-Image"
DEVICE_IDS="0,1"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --prompt  '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问 ". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 8 \
    --seed 42 \
    --output_file "./output/text_to_image_2512_lighting_lora_8steps_NoNegative_FAbsnd_ADALNSD.png" \
    --lora_path /root/work/filestorage/cyy/Qwen-Image-Lightning/lora \
    --ulysses_size 2