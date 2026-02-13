import sys
import os
import json
import argparse
import time
import torch
import torch_npu
import torch.distributed as dist
import numpy as np
import gc  
from PIL import Image
import logging
from typing import List, Optional, Dict, Any

from mindiesd import CacheConfig, CacheAgent, quantize

from qwenimage.utils import validate_quant_files
from qwenimage.pipeline_qwenimage import QwenImagePipeline
from qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.pipeline_qwenimage_layered import QwenImageLayeredPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from qwenimage.ulysses import parallelize_qwen_image_transformer
from qwenimage.distributed.parallel_mgr import ParallelConfig, init_parallel_env

SUPPORT_LIST = [
    "Qwen-Image",
    "Qwen-Image-2512",
    "Qwen-Image-Edit",
    "Qwen-Image-Edit-2509",
    "Qwen-Image-Edit-2511",
    "Qwen-Image-Layered",
    ]

# 环境变量配置（带默认值和注释，用户可通过环境变量灵活控制）
COND_CACHE = bool(int(os.environ.get('COND_CACHE', 0)))  # 条件输入缓存开关：1=启用，0=禁用
UNCOND_CACHE = bool(int(os.environ.get('UNCOND_CACHE', 0)))  # 无条件输入缓存开关：1=启用，0=禁用
CACHE_STEP_START = int(os.environ.get('CACHE_STEP_START', 10))  # 缓存开始步骤
CACHE_STEP_INTERVAL = int(os.environ.get('CACHE_STEP_INTERVAL', 3))  # 缓存步骤间隔
CACHE_STEP_END = int(os.environ.get('CACHE_STEP_END', 35))  # 缓存结束步骤
CACHE_BLOCK_START = int(os.environ.get('CACHE_BLOCK_START', 10))  # 缓存开始块
CACHE_BLOCK_END = int(os.environ.get('CACHE_BLOCK_END', 50))  # 缓存结束块
CACHE_BLOCKS_COUNT = int(os.environ.get('CACHE_BLOCKS_COUNT', 60))  # 总块数
QUANT_USE_NZ = bool(int(os.environ.get('QUANT_USE_NZ', 1)))  # 量化是否使用nz：1=启用，0=禁用

EXAMPLE_PROMPT = {
     "Qwen-Image": {
        "prompt":
            '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition''',
        "negative_prompt":
            " ",
    },
    "Qwen-Image-2512": {
        "prompt":
           '''A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.''',
        "negative_prompt":
            " ",
    },
    "Qwen-Image-Edit": {
        "prompt":
            "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color",
        "negative_prompt":
            " ",
        "image":
            "./examples/yarn-art-pikachu.png",
    },
    "Qwen-Image-Edit-2509": {
        "prompt":
            "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而庄重。",
        "negative_prompt":
            " ",
        "image":
            "./examples/girl.PNG examples/boy.PNG",
    },
    "Qwen-Image-Edit-2511": {
        "prompt":
            "根据这图1中女性和图2中的男性，生成一组结婚照，并遵循以下描述：新郎穿着红色的中式马褂，新娘穿着精致的秀禾服，头戴金色凤冠。他们并肩站立在古老的朱红色宫墙前，背景是雕花的木窗。光线明亮柔和，构图对称，氛围喜庆而庄重。",
        "negative_prompt":
            " ",
        "image":
            "./examples/girl.PNG examples/boy.PNG",
    },
    "Qwen-Image-Layered": {
        "prompt":
            None,
        "negative_prompt":
            " ",        
        "image":
            "./examples/1.png",
    },
}

def load_images(image_path_str: str, color_format: str = "RGB") -> List[Image.Image]:
    """
    安全加载图片，支持多路径分割，带完整异常捕获
    :param image_path_str: 图片路径字符串，多路径用逗号分隔
    :param color_format: 图片颜色格式（RGB/RGBA）
    :return: 加载后的图片列表
    :raises: ValueError/FileNotFoundError/RuntimeError
    """
    if not image_path_str or not image_path_str.strip():
        raise ValueError("图片路径不能为空，请检查输入参数--image")
    
    # 分割并过滤空路径
    image_paths = [p.strip() for p in image_path_str.split(" ") if p.strip()]
    if not image_paths:
        raise ValueError(f"图片路径分割后为空，原始输入：{image_path_str}")
    
    images = []
    for idx, path in enumerate(image_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"第{idx+1}张图片文件不存在：{path}")
        if not os.path.isfile(path):
            raise RuntimeError(f"第{idx+1}张图片路径不是文件：{path}")
        try:
            img = Image.open(path).convert(color_format)
            images.append(img)
            logging.info(f"成功加载图片[{idx+1}/{len(image_paths)}]：{path}")
        except Exception as e:
            raise RuntimeError(f"第{idx+1}张图片加载失败：{path}，错误信息：{str(e)}") from e
    return images


def init_pipeline(
    task: str,
    ckpt_dir: str,
    transformer: QwenImageTransformer2DModel,
    vae: Optional[AutoencoderKLQwenImage] = None,
    torch_dtype: torch.dtype = torch.bfloat16
) -> Any:
    """
    提取公共Pipeline初始化逻辑，减少代码冗余，便于扩展新任务
    :param task: 任务类型
    :param ckpt_dir: 模型权重目录
    :param transformer: 初始化后的transformer模型
    :param vae: 初始化后的vae模型（仅Layered任务需要）
    :param torch_dtype: 数据类型
    :return: 对应任务的Pipeline实例
    """
    common_kwargs = {
        "pretrained_model_name_or_path": ckpt_dir,
        "transformer": transformer,
        "torch_dtype": torch_dtype,
        "device_map": None,  # 禁用自动设备映射，适配国产硬件
        "low_cpu_mem_usage": True,  # 低CPU内存模式，避免加载时内存溢出
    }
    # Layered任务需要传入VAE
    if vae is not None:
        common_kwargs["vae"] = vae

    if task in ["Qwen-Image", "Qwen-Image-2512"]:
        return QwenImagePipeline.from_pretrained(**common_kwargs)
    elif task == "Qwen-Image-Edit":
        return QwenImageEditPipeline.from_pretrained(**common_kwargs)
    elif task in ["Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511"]:
        return QwenImageEditPlusPipeline.from_pretrained(**common_kwargs)
    elif task == "Qwen-Image-Layered":
        return QwenImageLayeredPipeline.from_pretrained(**common_kwargs)
    else:
        raise ValueError(f"不支持的任务类型：{task}，仅支持{SUPPORT_LIST}")


def _init_logging(rank: int = 0):
    """
    分级日志初始化，主进程打印INFO，子进程仅打印ERROR，避免分布式日志刷屏
    :param rank: 分布式进程rank
    """
    log_level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=log_level,
        format=f"[Rank {rank}][%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def _validate_args(args: argparse.Namespace):
    # 基础必选参数校验
    assert args.ckpt_dir is not None and os.path.exists(args.ckpt_dir), \
        f"模型权重目录不存在，请检查--ckpt_dir：{args.ckpt_dir}"
    assert args.task in SUPPORT_LIST, f"不支持的任务类型：{args.task}，仅支持{SUPPORT_LIST}"
    assert args.task in EXAMPLE_PROMPT, f"任务{args.task}无示例配置，请补充EXAMPLE_PROMPT"

    # 自动补全默认Prompt/图片
    if args.prompt is None:
        args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"未指定--prompt，使用{args.task}默认示例Prompt：{args.prompt}")
    if args.image is None and "image" in EXAMPLE_PROMPT[args.task]:
        args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"未指定--image，使用{args.task}默认示例图片：{args.image}")

    # 数值参数范围校验
    assert args.cfg_scale > 0, f"CFG系数必须大于0，当前值：{args.cfg_scale}"
    assert args.guidance_scale > 0, f"引导系数必须大于0，当前值：{args.guidance_scale}"
    assert args.num_images_per_prompt >= 1, f"每张Prompt生成图片数必须大于等于1，当前值：{args.num_images_per_prompt}"
    assert args.cfg_size >= 1 and args.ulysses_size >= 1, f"并行度必须为正整数，当前值：cfg_size={args.cfg_size}, ulysses_size={args.ulysses_size}"

    # 分布式并行度关联性校验
    world_size = int(os.getenv("WORLD_SIZE", 1))
    assert args.cfg_size * args.ulysses_size == world_size, \
        f"并行度乘积必须等于分布式进程数（WORLD_SIZE={world_size}），当前：cfg_size={args.cfg_size} * ulysses_size={args.ulysses_size} = {args.cfg_size*args.ulysses_size}"

    # 单卡环境并行度校验
    if world_size == 1:
        assert args.cfg_size == 1 and args.ulysses_size == 1, \
            "单卡环境不支持并行配置，请将--cfg_size和--ulysses_size设为1"

    # Layered任务专属参数校验
    if args.task == "Qwen-Image-Layered":
        assert args.layers in range(1, 9), f"Layered任务层数必须为1-8之间的整数，当前值：{args.layers}"
        assert args.resolution in [640, 1024], f"Layered任务分辨率仅支持640/1024，当前值：{args.resolution}"
        assert args.color_format == "RGBA", f"Layered任务颜色格式仅支持RGBA，当前值：{args.color_format}"
    
    # 量化路径校验
    if args.quant_dit_path and not os.path.exists(args.quant_dit_path):
        raise FileNotFoundError(f"量化模型路径不存在：{args.quant_dit_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image系列模型推理加速生成脚本，适配NPU")
    # 核心任务配置
    parser.add_argument("--task", type=str, default="Qwen-Image", choices=SUPPORT_LIST, help="推理任务类型")
    parser.add_argument("--ckpt_dir", type=str, default="Qwen/Qwen-Image", help="模型权重根目录（本地路径或Hub名称）")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="模型数据类型")
    parser.add_argument("--prompt", type=str, default=None, help="文本提示词，未指定则使用对应任务的示例Prompt")
    parser.add_argument("--negative_prompt", type=str, default=None, help="负向提示词，用于Classifier-Free Guidance")
    parser.add_argument("--image", type=str, default=None, help="输入图片路径，多图用逗号分隔，编辑/分层任务必填")

    # 生成配置
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证结果可复现")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="True classifier-free guidance scale specific to Qwen-Image.")
    parser.add_argument("--guidance_scale",type=float, default=1.0, help="指导蒸馏模型系数，大于1.0时生效")
    parser.add_argument("--height", type=int, default=1024, help="生成图片高度")
    parser.add_argument("--width", type=int, default=1024, help="生成图片宽度")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="每张Prompt生成的图片数量")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="扩散模型去噪步")
    parser.add_argument("--output_file", type=str, default="text_to_image.png", help="生成图片保存路径（PNG格式）")

    # Layered任务专属配置
    parser.add_argument("--layers", type=int, default=4, help="Layered任务图片分解层数（1-8）")
    parser.add_argument("--resolution", type=int, default=640, choices=[640, 1024], help="Layered任务分辨率桶（仅支持640/1024）")
    parser.add_argument("--color_format", type=str, default="RGB",choices=["RGB", "RGBA"], help="图片颜色格式，Layered任务需要设置为RGBA")

    # 分布式并行配置
    parser.add_argument("--cfg_size", type=int, default=1, choices=[1, 2], help="DiT的CFG并行度（仅支持1/2）")
    parser.add_argument("--ulysses_size", type=int, default=1, help="DiT的Ulysses并行度")

    # Vae配置
    parser.add_argument("--vae_tiling", action="store_true", default=False, help="启用VAE分块推理，降低显存占用")
    parser.add_argument("--vae_slicing", action="store_true", default=False, help="启用VAE切片推理，降低显存占用")

    # 设备配置
    parser.add_argument("--device", type=str, default="npu", choices=["npu"], help="运行设备")
    parser.add_argument("--device_id", type=int, default=0, help="单卡环境设备ID，分布式环境忽略")

    # 量化配置
    parser.add_argument("--quant_dit_path", type=str, help="量化模型路径，指定则启用模型量化")    

    # 扩展配置
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径（仅Edit-2509/2511任务有效）")

    args = parser.parse_args()
    rank = int(os.getenv("RANK", 0))  # 获取rank（分布式/单卡）
    _init_logging(rank)  # 先初始化日志，再校验参数
    _validate_args(args)
    return args

def generate(args: argparse.Namespace):
    """主推理函数，整合所有逻辑：设备初始化→模型加载→优化配置→推理→保存"""
    # 分布式环境变量初始化
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    logging.info(f"开始初始化推理环境，Rank：{rank}/{world_size-1}，本地Rank：{local_rank}")

    device_idx = local_rank if int(os.getenv("WORLD_SIZE", 1)) > 1 else args.device_id
    device = torch.device(f"npu:{device_idx}")
    torch.npu.set_device(device)
    logging.info(f"初始化NPU设备成功，设备ID：{device_idx}")

    # 分布式进程组初始化
    if world_size > 1:
        try:
            dist.init_process_group(
                backend="hccl" if args.device == "npu" else "nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size
            )
            logging.info(f"分布式进程组初始化成功，后端：{dist.get_backend()}")
        except Exception as e:
            logging.error(f"分布式进程组初始化失败", exc_info=True)
            raise SystemExit(1) from e

    # 并行配置初始化（CFG/Ulysses）
    if args.cfg_size > 1 or args.ulysses_size > 1:
        try:
            parallel_config = ParallelConfig(
                sp_degree=args.ulysses_size,
                ulysses_degree=args.ulysses_size,
                use_cfg_parallel=(args.cfg_size == 2),
                world_size=world_size,
            )
            init_parallel_env(parallel_config)
            logging.info(f"并行环境初始化成功，配置：{parallel_config.__dict__}")
        except Exception as e:
            logging.error(f"并行环境初始化失败", exc_info=True)
            raise SystemExit(1) from e

    torch_dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
    logging.info(f"模型数据类型：{torch_dtype}，运行设备：{device}")

    # 加载Transformer模型（带低内存优化）
    transformer = QwenImageTransformer2DModel.from_pretrained(
        os.path.join(args.ckpt_dir, 'transformer'),
        torch_dtype=torch_dtype,
        device_map=None,  # 禁用自动设备映射，昇腾环境下默认加载到CPU
        low_cpu_mem_usage=True,  # 启用CPU低内存模式，避免加载时CPU内存溢出
    )

    if args.quant_dit_path:
        logging.info(f"开始模型量化，量化路径：{args.quant_dit_path}")
        try:
            quant_type, quant_config_desc_path = validate_quant_files(args.quant_dit_path)
            logging.info(f"量化文件校验通过，类型：{quant_type}，配置文件：{quant_config_desc_path}")
            
            quantize(
                model=transformer,
                quant_des_path=quant_config_desc_path,
                use_nz=QUANT_USE_NZ,
            )
            logging.info("Transformer模型量化成功")
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logging.error(f"模型量化失败", exc_info=True)
            raise SystemExit(1) from e

    # 单独加载VAE模型（仅Layered任务需要）
    vae = None
    if args.task == "Qwen-Image-Layered":
        logging.info(f"开始加载VAE模型，路径：{os.path.join(args.ckpt_dir, 'vae')}")
        vae = AutoencoderKLQwenImage.from_pretrained(
            os.path.join(args.ckpt_dir, 'vae'),
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )

    # 初始化Pipeline
    logging.info(f"开始初始化{args.task}任务Pipeline")
    pipeline = init_pipeline(
        task=args.task,
        ckpt_dir=args.ckpt_dir,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch_dtype
    )
    # Pipeline移至目标设备
    pipeline = pipeline.to(device)

    # 加载LoRA权重
    if args.lora_path:
        logging.info(f"开始加载LoRA权重，路径：{args.lora_path}")
        pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=args.lora_path)
        pipeline.fuse_lora(safe_fuse=True)
        
    # VAE优化配置（避免显存溢出）
    if args.vae_tiling:
        pipeline.vae.use_tiling = True
        logging.info("启用VAE分块推理（vae_tiling）")
    if args.vae_slicing:
        pipeline.vae.use_slicing = True
        logging.info("启用VAE切片推理（vae_slicing）")

    # 推理进度条配置
    pipeline.set_progress_bar_config(disable=None) 

    # 缓存配置（基于环境变量，灵活控制）
    if COND_CACHE or UNCOND_CACHE:
        cache_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=CACHE_BLOCKS_COUNT,
            steps_count=args.num_inference_steps,
            step_start=CACHE_STEP_START,
            step_interval=CACHE_STEP_INTERVAL,
            step_end=CACHE_STEP_END,
            block_start=CACHE_BLOCK_START,
            block_end=CACHE_BLOCK_END
        )
        pipeline.transformer.cache_cond = CacheAgent(cache_config) if COND_CACHE else None
        pipeline.transformer.cache_uncond = CacheAgent(cache_config) if UNCOND_CACHE else None
        logging.info(f"启用缓存配置，COND_CACHE={COND_CACHE}, UNCOND_CACHE={UNCOND_CACHE}，配置：{cache_config.__dict__}")

    # Ulysses并行化改造Transformer
    if args.ulysses_size > 1:
        parallelize_qwen_image_transformer(pipeline)

    # 构造推理输入参数（分预热和正式，预热3步解决首次算子编译耗时
    inputs_warm_up: Dict[str, Any] = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "true_cfg_scale": args.cfg_scale,
        "guidance_scale": args.guidance_scale,
        "generator": torch.Generator(device="cpu").manual_seed(args.seed),
        "num_images_per_prompt": args.num_images_per_prompt,
        "num_inference_steps": 3,  # 预热步数，固定3步
        "width": args.width,
        "height": args.height,
    }

    # 编辑/分层任务添加图片参数
    if "image" in EXAMPLE_PROMPT[args.task]:
        input_images = load_images(args.image, args.color_format)
        inputs_warm_up["image"] = input_images
        inputs_warm_up.pop("width")
        inputs_warm_up.pop("height")

    # Layered任务添加专属参数
    if args.task == "Qwen-Image-Layered":
        inputs_warm_up.update({
            "layers": args.layers,
            "resolution": args.resolution,
            "cfg_normalize": True,
            "use_en_prompt": True,
        })

    # 正式推理参数（复制预热参数并修改步数）
    inputs = inputs_warm_up.copy()
    inputs["num_inference_steps"] = args.num_inference_steps

    # 模型预热
    logging.info("开始模型预热（3步推理），解决首次算子编译耗时")
    with torch.inference_mode():
        output_warm = pipeline(**inputs_warm_up)

    # 分布式屏障，等待所有进程预热完成
    if dist.is_initialized():
        dist.barrier()

    # 正式推理（统计端到端耗时）
    # 设备同步，保证耗时统计准确
    torch.npu.synchronize()
    start_time = time.time()

    with torch.inference_mode():
        output = pipeline(**inputs)
    
     # 设备同步，结束耗时统计
    torch.npu.synchronize()
    end_time = time.time()
    infer_time = end_time - start_time
    logging.info(f"正式推理完成，端到端耗时：{infer_time:.4f}秒")

    # 仅主进程（rank=0）保存图片，避免分布式多进程重复写入/覆盖
    if rank == 0:
        logging.info(f"开始保存生成图片，输出路径：{args.output_file}")
        try:
            # 处理输出路径：自动创建目录，支持多图片命名
            output_path = os.path.abspath(args.output_file)
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logging.info(f"创建输出目录成功：{output_dir}")
            # 提取文件名和后缀
            output_name, output_ext = os.path.splitext(os.path.basename(output_path))
            output_ext = output_ext if output_ext else ".png"

            # 保存图片（区分Layered任务和普通任务）
            output_images = output.images
            if args.task == "Qwen-Image-Layered":
                for i, layer_images in enumerate(output_images):
                    logging.info(f"保存第{i+1}组图片，共{len(layer_images)}层")
                    for j, img in enumerate(layer_images):
                        save_path = os.path.join(output_dir, f"{output_name}_{i}_layer{j}{output_ext}")
                        img.save(save_path)
                        logging.info(f"层图片保存成功：{save_path}")
            else:
                for i, img in enumerate(output_images):
                    save_path = os.path.join(output_dir, f"{output_name}_{i}{output_ext}")
                    img.save(save_path)
                    logging.info(f"生成图片保存成功：{save_path}")
        except Exception as e:
            logging.error(f"图片保存失败", exc_info=True)
            raise SystemExit(1) from e

    # 资源释放（关键：避免显存泄漏/碎片，尤其多次推理时）
    logging.info("开始释放推理资源，清理显存/内存")
    # 删除大变量，释放内存
    del output, output_warm, pipeline, transformer
    if vae is not None:
        del vae
    # 强制垃圾回收
    gc.collect()
    # 清空设备缓存
    torch.npu.empty_cache()
    logging.info("资源释放完成")

   # 销毁分布式进程组（仅分布式场景）
    if world_size > 1:
        dist.destroy_process_group()
        logging.info("分布式进程组销毁成功")

    logging.info(f"Rank {rank} 推理任务全部完成")

if __name__ == "__main__":
    args = _parse_args()
    generate(args)