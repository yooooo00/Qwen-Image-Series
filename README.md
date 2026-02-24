---
pipeline_tag: image-to-image
license: apache-2.0
frameworks:
  - PyTorch
---
## 一、准备运行环境

  **表 1**  版本配套表

  | 配套  | 版本 | 环境准备指导 |
  | ----- | ----- |-----|
  | Python | 3.11.10 | - |
  | torch | 2.6.0 | - |

### 1.1 获取CANN&MindIE安装包&环境准备
- 设备支持
Atlas 800I/800T A2(8*64G)推理设备：支持的卡数最小为1
- [Atlas 800I/800T A2(8*64G)](https://www.hiascend.com/developer/download/community/result?module=pt+ie+cann&product=4&model=32)
- [环境准备指导](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/softwareinst/instg/instg_0001.html)

### 1.2 CANN安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构，{soc}表示昇腾AI处理器的版本。
chmod +x ./Ascend-cann-toolkit_{version}_linux-{arch}.run
chmod +x ./Ascend-cann-kernels-{soc}_{version}_linux.run
# 校验软件包安装文件的一致性和完整性
./Ascend-cann-toolkit_{version}_linux-{arch}.run --check
./Ascend-cann-kernels-{soc}_{version}_linux.run --check
# 安装
./Ascend-cann-toolkit_{version}_linux-{arch}.run --install
./Ascend-cann-kernels-{soc}_{version}_linux.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 1.3 MindIE安装
```shell
# 增加软件包可执行权限，{version}表示软件版本号，{arch}表示CPU架构。
chmod +x ./Ascend-mindie_${version}_linux-${arch}.run
./Ascend-mindie_${version}_linux-${arch}.run --check

# 方式一：默认路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install
# 设置环境变量
cd /usr/local/Ascend/mindie && source set_env.sh

# 方式二：指定路径安装
./Ascend-mindie_${version}_linux-${arch}.run --install-path=${AieInstallPath}
# 设置环境变量
cd ${AieInstallPath}/mindie && source set_env.sh
```

### 1.4 Torch_npu安装
下载 pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
```shell
tar -xzvf pytorch_v{pytorchversion}_py{pythonversion}.tar.gz
# 解压后，会有whl包
pip install torch_npu-{pytorchversion}.xxxx.{arch}.whl
```

## 二、下载权重

### 2.1 权重及配置文件说明
-  Huggingface

|  模型 | 链接  |
| ------------ | ------------ |
| Qwen-Image  |  [🤗 huggingface](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen-Image-2512  |  [🤗 huggingface](https://huggingface.co/Qwen/Qwen-Image-2512) |
| Qwen-Image-Edit  |  [🤗 huggingface](https://huggingface.co/Qwen/Qwen-Image-Edit) |
| Qwen-Image-Edit-2509  |  [🤗 huggingface](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) |
| Qwen-Image-Edit-2511  |  [🤗 huggingface](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) |
| Qwen-Image-Layered  |  [🤗 huggingface](https://huggingface.co/Qwen/Qwen-Image-Layered) |


- ModelScope

|  模型 | 链接  |
| ------------ | ------------ |
| Qwen-Image  |  [🤖 ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image) |
| Qwen-Image-2512  |  [🤖 ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-2512) |
| Qwen-Image-Edit  |  [🤖 ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) |
| Qwen-Image-Edit-2509  |  [🤖 ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509) |
| Qwen-Image-Edit-2511  |  [🤖 ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511) |
| Qwen-Image-Layered  |  [🤖 ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-Layered) |
## 三、Qwen-Image使用

### 3.1 下载到本地，安装模型依赖
```shell
git clone https://modelers.cn/MindIE/Qwen-Image-series.git
cd Qwen-Image-series
pip3 install -r requirements.txt
```
### 3.2 Qwen-Image-2512
#### 3.2.1 单卡性能测试
#### 3.2.1.1 等价优化
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合

# ====================== 任务配置 ======================
TASK="Qwen-Image-2512"
MODEL_PATH="/weights/Qwen-Image-2512"
DEVICE_IDS="0"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --prompt  '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/text_to_image_2512.png" \
    --vae_tiling \
    --vae_slicing \
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- task: 推理任务选择
- ckpt_dir: 模型权重路径
- prompt: 文本正面提示词
- negative_prompt: 文本负面提示词
- width: 生成图片的宽
- height: 生成图片的高
- num_inference_steps: 推理的步数
- seed: 设置随机种子
- output_file: 生成图片的保存路径
- vae_tiling: 使能 VAE tiling 来减少显存占用
- vae_slicing: 使能 VAE slicing 来减少显存占用

#### 3.2.1.2 等价优化+算法优化
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合

export COND_CACHE=1          # 条件缓存
export UNCOND_CACHE=1        # 无条件缓存
export CACHE_STEP_START=10   # 缓存开始步骤
export CACHE_STEP_INTERVAL=3 # 缓存步骤间隔
export CACHE_STEP_END=35     # 缓存结束步骤
export CACHE_BLOCK_START=10  # 缓存开始块
export CACHE_BLOCK_END=50    # 缓存结束块

# ====================== 任务配置 ======================
TASK="Qwen-Image-2512"
MODEL_PATH="/weights/Qwen-Image-2512"
DEVICE_IDS="0"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --prompt  '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/text_to_image_2512.png" \
    --vae_tiling \
    --vae_slicing \
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子

#### 3.2.2 多卡性能测试
##### 3.2.2.1 2卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-2512"
MODEL_PATH="/weights/Qwen-Image-2512"
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
    --prompt '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 50 \
    --seed 42 \
    --output_file "./output/text_to_image_2512.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 1
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.2.2.2 4卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-2512"
MODEL_PATH="/weights/Qwen-Image-2512"
DEVICE_IDS="0,1,2,3"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
# 4卡并行(cfg_size=2 ulysses_size=2 优于 cfg_size=1 ulysses_size=4)
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --prompt '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 50 \
    --seed 42 \
    --output_file "./output/text_to_image_2512.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 2
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.2.2.3 8卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-2512"
MODEL_PATH="/weights/Qwen-Image-2512"
DEVICE_IDS="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
# 8卡并行(cfg_size=2 ulysses_size=4 优于 cfg_size=1 ulysses_size=8)
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --prompt '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 50 \
    --seed 42 \
    --output_file "./output/text_to_image_2512.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 4
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.2.2.4 16卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-2512"
MODEL_PATH="/weights/Qwen-Image-2512"
DEVICE_IDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --prompt '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''' \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 50 \
    --seed 42 \
    --output_file "./output/text_to_image_2512.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 8
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数


### 3.3 Qwen-Image-Edit-2511
#### 3.3.1 单卡性能测试
#### 3.3.1.1 等价优化
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/weights/Qwen-Image-Edit-2511"
DEVICE_IDS="0"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/image_edit_2511.png" \
    --vae_tiling \
    --vae_slicing \
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- task: 推理任务选择
- ckpt_dir: 模型权重路径
- image: 输入图片路径，多图则用空格分隔，如`img1 img2`
- prompt: 文本正面提示词
- negative_prompt: 文本负面提示词
- width: 生成图片的宽
- height: 生成图片的高
- num_inference_steps: 推理的步数
- seed: 设置随机种子
- output_file: 生成图片的保存路径
- vae_tiling: 使能 VAE tiling 来减少显存占用
- vae_slicing: 使能 VAE slicing 来减少显存占用

#### 3.3.1.2 等价优化+算法优化
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1                # 0:普通, 1:FA, 2:LA
export OVERLAP=0             # 通信-计算重叠开关
export ROPE_FUSE=1           # RoPE算子融合
export ADALN_FUSE=1          # ADALN算子融合

export COND_CACHE=1          # 条件缓存
export UNCOND_CACHE=1        # 无条件缓存
export CACHE_STEP_START=10   # 缓存开始步骤
export CACHE_STEP_INTERVAL=3 # 缓存步骤间隔
export CACHE_STEP_END=35     # 缓存结束步骤
export CACHE_BLOCK_START=10  # 缓存开始块
export CACHE_BLOCK_END=50    # 缓存结束块

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/weights/Qwen-Image-Edit-2511"
DEVICE_IDS="0"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/image_edit_2511.png" \
    --vae_tiling \
    --vae_slicing \
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子

#### 3.3.2 多卡性能测试
##### 3.3.2.1 2卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/weights/Qwen-Image-Edit-2511"
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
    --image "./examples/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/image_edit_2511_.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 1
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.3.2.2 4卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/weights/Qwen-Image-Edit-2511"
DEVICE_IDS="0,1,2,3"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
# 4卡并行(cfg_size=2 ulysses_size=2 优于 cfg_size=1 ulysses_size=4)
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/image_edit_2511_.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 2
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.3.2.3 8卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/weights/Qwen-Image-Edit-2511"
DEVICE_IDS="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
# 8卡并行(cfg_size=2 ulysses_size=4 优于 cfg_size=1 ulysses_size=8)
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/image_edit_2511_.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 4
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.3.2.4 16卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/weights/Qwen-Image-Edit-2511"
DEVICE_IDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/image_edit_2511_.png" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 8
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

### 3.4 Qwen-Image-Layered
#### 3.4.1 单卡性能测试
#### 3.4.1.1 等价优化
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合

# ====================== 任务配置 ======================
TASK="Qwen-Image-Layered"
MODEL_PATH="/weights/Qwen-Image-Layered"
DEVICE_IDS="0"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/2.png" \
    --negative_prompt " " \
    --num_inference_steps 50 \
    --seed 42 \
    --layers 3 \
    --resolution 640 \
    --color_format "RGBA" \
    --output_file "./image_layered" \
    --vae_tiling \
    --vae_slicing \
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- task: 推理任务选择
- ckpt_dir: 模型权重路径
- image: 输入图片路径
- negative_prompt: 文本负面提示词
- num_inference_steps: 推理的步数
- seed: 设置随机种子
- layers: 图片分解层数
- resolution: 输出图片分辨率
- color_format: 图片颜色格式
- output_file: 生成图片的保存路径
- vae_tiling: 使能 VAE tiling 来减少显存占用
- vae_slicing: 使能 VAE slicing 来减少显存占用

#### 3.4.1.2 等价优化+算法优化
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合

export COND_CACHE=1          # 条件缓存
export UNCOND_CACHE=1        # 无条件缓存
export CACHE_STEP_START=10   # 缓存开始步骤
export CACHE_STEP_INTERVAL=3 # 缓存步骤间隔
export CACHE_STEP_END=35     # 缓存结束步骤
export CACHE_BLOCK_START=10  # 缓存开始块
export CACHE_BLOCK_END=50    # 缓存结束块

# ====================== 任务配置 ======================
TASK="Qwen-Image-Layered"
MODEL_PATH="/weights/Qwen-Image-Layered"
DEVICE_IDS="0"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/2.png" \
    --negative_prompt " " \
    --num_inference_steps 50 \
    --seed 42 \
    --layers 3 \
    --resolution 640 \
    --color_format "RGBA" \
    --output_file "./image_layered" \
    --vae_tiling \
    --vae_slicing \
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子

#### 3.4.2 多卡性能测试
##### 3.4.2.1 2卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Layered"
MODEL_PATH="/weights/Qwen-Image-Layered"
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
    --image "./examples/2.png" \
    --negative_prompt " " \
    --num_inference_steps 50 \
    --seed 42 \
    --layers 3 \
    --resolution 640 \
    --color_format "RGBA" \
    --output_file "./image_layered" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 1
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.4.2.2 4卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Layered"
MODEL_PATH="/weights/Qwen-Image-Layered"
DEVICE_IDS="0,1,2,3"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
# 4卡并行(cfg_size=2 ulysses_size=2 优于 cfg_size=1 ulysses_size=4 )
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/2.png" \
    --negative_prompt " " \
    --num_inference_steps 50 \
    --seed 42 \
    --layers 3 \
    --resolution 640 \
    --color_format "RGBA" \
    --output_file "./image_layered" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 2
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.4.2.3 8卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Layered"
MODEL_PATH="/weights/Qwen-Image-Layered"
DEVICE_IDS="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
# 8卡并行(cfg_size=2 ulysses_size=4 优于 cfg_size=1 ulysses_size=8 )
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/2.png" \
    --negative_prompt " " \
    --num_inference_steps 50 \
    --seed 42 \
    --layers 3 \
    --resolution 640 \
    --color_format "RGBA" \
    --output_file "./image_layered" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 4
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

##### 3.4.2.4 16卡性能测试
执行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关，0:关闭，1:开启
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合
# export COND_CACHE=1       # 条件缓存（按需开启）
# export UNCOND_CACHE=1     # 无条件缓存（按需开启）

# ====================== 任务配置 ======================
TASK="Qwen-Image-Layered"
MODEL_PATH="/weights/Qwen-Image-Layered"
DEVICE_IDS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/2.png" \
    --negative_prompt " " \
    --num_inference_steps 50 \
    --seed 42 \
    --layers 3 \
    --resolution 640 \
    --color_format "RGBA" \
    --output_file "./image_layered" \
    --vae_tiling \
    --vae_slicing \
    --cfg_size 2 \
    --ulysses_size 8
```
参数说明：
- ALGO: 为0表示默认SDPA算子；设置为1表示使用FA算子；设置为2表示使用高性能FA算子
- OVERLAP: 通信-计算重叠开关，为0表示关闭，为1表示开启。仅在开启Ulysses序列并行（指定--ulysses_size）且多卡环境下生效
- cfg_size: cfg并行数，使用时只能设定为2
- ulysses_size: ulysses并行数，使用时设定为24的因数

## 四、量化功能支持
新增Qwen-Image、Qwen-Image-2512、Qwen-Image-Edit、Qwen-Image-Edit-2509、Qwen-Image-Edit-2511、Qwen-Image-Layered模型的量化支持，支持权重 8 位（w8）与激活 8 位 / 16 位（a8/a16）的量化组合，针对DiT模型进行量化，降低显存占用，提高模型推理性能
### 4.1 安装量化工具msModelSlim
```shell
git clone https://gitcode.com/Ascend/msit
cd msit/msmodelslim
bash install.sh
```
### 4.2 导出量化权重
通过`quant_qwenimage.py`脚本生成量化模型及描述文件，需基于原始模型权重进行量化。
##### 4.2.1 量化脚本运行示例
以Qwen-Image-Edit-2511模型为例
###### 4.2.1.1 生成8bit权重+8bit激活的动态量化模型（w8a8）：
执行命令：
```shell
model_path="/weights/Qwen-Image-Edit-2511"
python quant_qwenimage.py \
    --model_path ${model_path} \
    --device_id 2 \
    --quant_mode w8a8 \
    --w_sym \
    --is_dynamic \
    --act_method 3 \
    --quant_save_dir ./quant_w8a8_dynamic_withoutData_use_disable_quant_layers \
```
参数说明：
- model_path: 原始模型权重路径
- device_id: 执行模型推理的芯片id
- quant_mode: 量化模式（权重+激活位宽）
- w_sym: 是否对权重使用对称量化（默认False，加此参数表示启用） 
- is_dynamic: 是否启用动态量化（默认False，加此参数表示启用）
- act_method: 激活量化方法（1=min-max，2=histogram，3=auto-mixed，推荐3）
- quant_save_dir: 量化模型保存路径

执行后，`quant_w8a8_dynamic_withoutData_use_disable_quant_layers`目录下会生成两个文件：
- `quant_model_description_w8a8_dynamic.json`：量化配置描述文件（包含量化位宽、层映射等元信息）
- `quant_model_weight_w8a8_dynamic.safetensors`：量化后的权重文件（采用safe tensor格式）

###### 4.2.1.2 生成8bit权重+16bit激活的量化模型（w8a16）：
执行命令：
```shell
model_path="/mnt/weights/Qwen-Image-Edit-2511"
python quant_qwenimage.py \
    --model_path ${model_path} \
    --device_id 0 \
    --quant_mode w8a16 \
    --w_sym \
    --act_method 3 \
    --quant_save_dir ./quant_w8a16_withoutData_use_disable_quant_layers

```
参数说明：
- model_path: 原始模型权重路径
- device_id: 执行模型推理的芯片id
- quant_mode: 量化模式（权重+激活位宽）
- w_sym: 是否对权重使用对称量化（默认False，加此参数表示启用） 
- act_method: 激活量化方法（1=min-max，2=histogram，3=auto-mixed，推荐3）
- quant_save_dir: 量化模型保存路径

执行后，`quant_w8a16_withoutData_use_disable_quant_layers`目录下会生成两个文件：
- `quant_model_description_w8a16.json`：量化配置描述文件（包含量化位宽、层映射等元信息）
- `quant_model_weight_w8a16.safetensors`：量化后的权重文件（采用safe tensor格式）


### 4.3 量化模型推理
以Qwen-Image-Edit-2511模型的w8a8量化为例子，运行命令：
```shell
# ====================== 全局环境变量配置 ======================
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export ALGO=1               # 0:普通, 1:FA, 2:LA
export OVERLAP=0            # 通信-计算重叠开关
export ROPE_FUSE=1          # RoPE算子融合
export ADALN_FUSE=1         # ADALN算子融合

# ====================== 任务配置 ======================
TASK="Qwen-Image-Edit-2511"
MODEL_PATH="/weights/Qwen-Image-Edit-2511"
DEVICE_IDS="0"  # 单卡：0；多卡：0,1
NPROC_PER_NODE=$(echo $DEVICE_IDS | tr ',' '\n' | wc -l)  # 自动计算进程数
MASTER_PORT=29508

# ====================== 设备配置 ======================
export ASCEND_RT_VISIBLE_DEVICES=${DEVICE_IDS}

# ====================== 执行推理 ======================
torchrun --nproc_per_node=${NPROC_PER_NODE} --master-port ${MASTER_PORT} generate.py \
    --task ${TASK} \
    --ckpt_dir ${MODEL_PATH} \
    --image "./examples/yarn-art-pikachu.png" \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --width 1024 \
    --height 1024 \
    --num_inference_steps 40 \
    --seed 42 \
    --output_file "./output/image_edit_2511.png" \
    --vae_tiling \
    --vae_slicing \
    --quant_dit_path ./quant_w8a8_dynamic_withoutData_use_disable_quant_layers
```
参数说明：
- quant_dit_path：量化DiT模型权重的路径，传入该参数则使能量化。

## 五、常见问题
- 问题 1：显存不足 → 开启 --vae_tiling --vae_slicing，降低 width/height；
- 问题 2：多卡通信失败 → 检查 ASCEND_RT_VISIBLE_DEVICES 和 master-port 未被占用；

## 声明
- 本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。
- 如您在使用本代码仓的过程中，发现任何问题（包括但不限于功能问题、合规问题），请在本代码仓提交issue，我们将及时审视并解答。
--------------------
上面是我给我的代码工程写的readme，帮我看看有没有问题