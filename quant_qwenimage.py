import argparse
import os
import torch
import torch_npu
import torch.nn as nn
import fnmatch
from typing import List, Union, Tuple

from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

def parse_quant_args():
    """解析Qwen-Image量化的命令行参数（适配DiffusionPipeline）"""
    parser = argparse.ArgumentParser(description="Qwen-Image model quantization script (no calibration data required)")

    # model arguments
    core_group = parser.add_argument_group(title="核心模型参数")
    core_group.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory containing original model checkpoints"
    )
    core_group.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="NPU device ID to use (default: 0)"
    )

    # Quantization-specific arguments
    quant_group = parser.add_argument_group(title="量化参数")
    quant_group.add_argument(
        "--quant_save_dir",
        type=str,
        default="./quant_dit_w8a8_dynamic",
        help="Directory to save quantized weights (default: ./quant_dit_w8a8_dynamic)"
    )
    quant_group.add_argument(
        "--quant_mode",
        type=str,
        default="w8a8",
        choices=["w8a8", "w8a16"],
        help="Quantization mode: w8a8 (8bit weight + 8bit activation) or w8a16 (8bit weight + 16bit activation)"
    )
    quant_group.add_argument(
        "--is_dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic quantization (activation parameters generated dynamically)"
    )
    quant_group.add_argument(
        "--w_sym",
        action="store_true",
        default=False,
        help="权重使用对称量化"
    )
    quant_group.add_argument(
        "--act_method",
        type=int,
        default=3,
        help="激活量化方法（1=min-max，2=histogram，3=auto-mixed，推荐3）"
    )
    quant_group.add_argument(
        "--disable_quant_layers",
        type=str,
        nargs="*",
        default=['*txt_mlp.net.2*', '*img_mod.1*', '*txt_mod.1*', '*time_text_embed*'],
        help="额外排除量化的层（前缀匹配）"
    )

    # 参数校验
    args = parser.parse_args()
    if not os.path.exists(args.model_path) and not args.model_path.startswith("qwen/"):
        raise FileNotFoundError(f"模型路径不存在：{args.model_path}（本地路径或用'qwen/Qwen-Image'从HF下载）")

    return args


def init_npu_env(args):
    """初始化昇腾NPU环境，避免算子编译问题"""
    try:
        torch_npu.npu.set_compile_mode(jit_compile=False)  # 禁用JIT编译
        torch.npu.config.allow_internal_format = False     # 禁用内部格式优化（保证量化兼容性）
        torch.npu.set_device(args.device_id)
        print("NPU环境初始化成功")
    except Exception as e:
        raise RuntimeError(f"NPU环境初始化失败：{str(e)}")

def load_qwen_image_dit(args):
    """加载Qwen-Image 的 transformer"""
    try:
        print(f"从 {args.model_path} 加载Qwen-Image模型...")
        core_model = QwenImageTransformer2DModel.from_pretrained(
            os.path.join(args.model_path, 'transformer'),
            torch_dtype=torch.bfloat16,
            device_map=None,  # 禁用自动设备映射，昇腾环境下默认加载到CPU
        )

        device = torch.device(f"npu:{args.device_id}")
        core_model = core_model.to(device) 
        core_model.eval()
        print(f"模型加载成功，核心模块已转移到 NPU:{args.device_id}")
        return core_model
    except Exception as e:
        raise RuntimeError(f"加载Qwen-Image模型失败：{str(e)}")


def get_qwen_image_quant_config(
    quant_mode: str,
    is_dynamic: bool,
    w_sym: bool,
    act_method: int,
    disable_names: list = None,
    dev_type: str = "npu",
    dev_id: int = None,** kwargs
) -> QuantConfig:
    """Generate quantization configuration based on command-line arguments.

    Args:
        quant_mode: Quantization mode, choices are "w8a8" (8-bit weight + 8-bit activation)
                    or "w8a16" (8-bit weight + 16-bit activation).
        is_dynamic: Whether to enable dynamic quantization (activation params computed on-the-fly).
        w_sym: Whether to use symmetric quantization for weights.
        act_method: Activation quantization method (Label-Free scenarios):
                    1 = min-max quantization,
                    2 = histogram quantization,
                    3 = auto-mixed quantization (recommended for LLM large models).
                    Note: 3 is not supported when low-bit sparse quantization is enabled.
        disable_names: List of layer names to exclude from quantization (optional).
        dev_type: Device type (default: "npu" for Ascend devices).
        dev_id: Device ID to use (e.g., NPU device number). 
        **kwargs: Additional keyword arguments for QuantConfig initialization.

    Returns:
        QuantConfig: Initialized quantization configuration object.

    Raises:
        ValueError: If quant_mode is not in ["w8a8", "w8a16"].
        ValueError: If act_method is not in [1, 2, 3].
    """
    # Validate activation quantization method
    if act_method not in [1, 2, 3]:
        raise ValueError(f"Unsupported act_method {act_method}. Supported: 1 (min-max), 2 (histogram), 3 (auto-mixed)")

    # Parse quant_mode to determine bit widths
    if quant_mode == "w8a8":
        w_bit = 8
        a_bit = 8
    elif quant_mode == "w8a16":
        w_bit = 8
        a_bit = 16  # 16-bit activation = no quantization (original precision)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}. Choose 'w8a8' or 'w8a16'")

    # Initialize quantization configuration
    quant_config = QuantConfig(
        w_bit=w_bit,
        a_bit=a_bit,
        disable_names=disable_names,
        dev_type=dev_type,
        dev_id=dev_id,
        act_method=act_method,
        pr=1.0,
        w_sym=w_sym,
        mm_tensor=False,
        is_dynamic=is_dynamic,** kwargs
    )
    
    return quant_config


def get_disable_layer_names(model: nn.Module,
                            layer_include: Union[List[str], Tuple[str], str],
                            layer_exclude: Union[List[str], Tuple[str], str]) -> List[str]:
    """
    Get the names of layers to be disabled based on inclusion and exclusion patterns using fnmatch.

    Args:
        model: The neural network module
        layer_include: Patterns for layers to include. Can be a string, list or tuple of strings.
        layer_exclude: Patterns for layers to exclude. Can be a string, list or tuple of strings.

    Returns:
        List of layer names that should be disabled for quantization.
    """
    # Convert single string patterns to list for uniform processing
    if isinstance(layer_include, str):
        layer_include = [layer_include]
    if isinstance(layer_exclude, str):
        layer_exclude = [layer_exclude]

    all_layer_names = []
    quant_layer_names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            all_layer_names.append(name)

        # Check inclusion patterns
        if layer_include and not any(fnmatch.fnmatch(name, pattern) for pattern in layer_include):
            continue
        # Check exclusion patterns
        if layer_exclude and any(fnmatch.fnmatch(name, pattern) for pattern in layer_exclude):
            continue

        quant_layer_names.add(name)

    disable_layer_names = [name for name in all_layer_names if name not in quant_layer_names]
    return disable_layer_names


def quant(args):
    try:
        # 1. 初始化NPU环境
        init_npu_env(args)
        # 2. 加载Qwen-Imag模型的dit部分
        model = load_qwen_image_dit(args)
        # 3. 生成量化配置
        disable_quant_layers = get_disable_layer_names(
            model,
            layer_include='*',
            layer_exclude=args.disable_quant_layers,
        )
        quant_config = get_qwen_image_quant_config(
            quant_mode=args.quant_mode,
            is_dynamic=args.is_dynamic,
            w_sym=args.w_sym,
            act_method=args.act_method,
            disable_names=disable_quant_layers,
            dev_id=args.device_id
        )
        print(f"量化配置生成完成：{args.quant_mode} 模式，排除层：{quant_config.disable_names}")


        # 4. 执行校准，不需要校准数据的场景不需要传calib_data
        calibrator = Calibrator(model=model, cfg=quant_config, disable_level='L0')  # disable_level: 自动回退n个linear
        calibrator.run()  # 执行PTQ量化校准

        # 5. 保存量化权重
        os.makedirs(args.quant_save_dir, exist_ok=True)
        calibrator.save(args.quant_save_dir, save_type=["safe_tensor"])

        # 6. 完成提示
        print("\nQwen-Image量化全流程完成！")
    except Exception as e:
        print(f"量化失败：{str(e)}", flush=True)
        exit(1)


if __name__ == "__main__":
    args = parse_quant_args()
    quant(args)