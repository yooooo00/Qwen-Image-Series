import os
import re

def validate_quant_files(quant_dit_path):
    """
    校验量化文件夹（仅含1个权重+1个配置文件）的合法性
    :param quant_dit_path: 量化权重文件夹路径
    :raises FileNotFoundError: 文件夹/文件不存在
    :raises ValueError: 量化类型不匹配/文件类型异常
    """
    # 1. 基础校验：文件夹存在
    quant_dir = os.path.abspath(quant_dit_path)
    if not os.path.isdir(quant_dir):
        raise FileNotFoundError(f"量化文件夹不存在：{quant_dir}")
    
    # 2. 获取文件夹下的所有文件（过滤子文件夹）
    all_files = [f for f in os.listdir(quant_dir) if os.path.isfile(os.path.join(quant_dir, f))]
    
    # 3. 筛选权重文件和配置文件（各找1个）
    weight_file = None
    config_file = None
    quant_type_pattern = re.compile(r"(w8a8|w8a16)")  # 提取w8a8/w8a16
    
    for file_name in all_files:
        file_path = os.path.join(quant_dir, file_name)
        # 匹配权重文件（.safetensors后缀）
        if file_name.endswith(".safetensors") and "quant_model_weight" in file_name:
            weight_file = file_path
            # 提取权重文件的量化类型
            weight_quant_match = quant_type_pattern.search(file_name)
            weight_quant_type = weight_quant_match.group(1) if weight_quant_match else None
        # 匹配配置文件（.json后缀）
        elif file_name.endswith(".json") and "quant_model_description" in file_name:
            config_file = file_path
            # 提取配置文件的量化类型
            config_quant_match = quant_type_pattern.search(file_name)
            config_quant_type = config_quant_match.group(1) if config_quant_match else None
    
    # 4. 校验文件是否存在
    if not weight_file:
        raise FileNotFoundError(f"在 {quant_dir} 中未找到量化权重文件（需包含quant_model_weight且后缀为.safetensors）")
    if not config_file:
        raise FileNotFoundError(f"在 {quant_dir} 中未找到量化配置文件（需包含quant_model_description且后缀为.json）")
    
    # 5. 校验量化类型是否匹配
    if not (weight_quant_type and config_quant_type):
        raise ValueError("权重/配置文件名中未找到w8a8或w8a16标识")
    if weight_quant_type != config_quant_type:
        raise ValueError(f"量化类型不匹配：权重文件是{weight_quant_type}，配置文件是{config_quant_type}")
    
    # 返回校验通过的量化类型和配置文件路径（供后续使用）
    return weight_quant_type, config_file


import functools
import torch

def warp_blend(vae, tile_overlap_factor: float = 0.25):
    """
    适配AutoencoderKLQwenImage的tile混合函数替换
    
    Args:
        vae: AutoencoderKLQwenImage实例
        tile_overlap_factor: tile重叠因子（默认0.25，即25%重叠）
    """
    # 优化后的垂直混合函数
    @functools.wraps(vae.__class__.blend_v)
    def new_blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        # 张量维度：B, C, T, H, W → H是dim3，W是dim4
        blend_extent_new = min(a.shape[3], b.shape[3], blend_extent)
        if blend_extent_new == blend_extent:
            if blend_extent_new == self.blend_extent_enc:
                weights = self.blend_weights_enc
            else:
                weights = self.blend_weights_dec
        else:
            weights = torch.linspace(0, 1, blend_extent_new, device=a.device, dtype=a.dtype)
        
        # 切片：a的底部重叠区域（H维度）、b的顶部重叠区域（H维度）
        a_slice = a[:, :, :, -blend_extent_new:, :]
        b_slice = b[:, :, :, :blend_extent_new, :]
        
        # 广播权重：(N,) → (1,1,1,N,1)
        blended = a_slice * (1 - weights[None, None, None, :, None]) + b_slice * weights[None, None, None, :, None]
        b[:, :, :, :blend_extent_new, :] = blended
        return b

    # 优化后的水平混合函数
    @functools.wraps(vae.__class__.blend_h)
    def new_blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent_new = min(a.shape[4], b.shape[4], blend_extent)
        if blend_extent_new == blend_extent:
            if blend_extent_new == self.blend_extent_enc:
                weights = self.blend_weights_enc
            else:
                weights = self.blend_weights_dec
        else:
            weights = torch.linspace(0, 1, blend_extent_new, device=a.device, dtype=a.dtype)
        
        # 切片：a的右侧重叠区域（W维度）、b的左侧重叠区域（W维度）
        a_slice = a[:, :, :, :, -blend_extent_new:]
        b_slice = b[:, :, :, :, :blend_extent_new]
        
        # 广播权重：(N,) → (1,1,1,1,N)
        blended = a_slice * (1 - weights[None, None, None, None, :]) + b_slice * weights[None, None, None, None, :]
        b[:, :, :, :, :blend_extent_new] = blended
        return b

    # 绑定方法到VAE实例
    vae.blend_v = new_blend_v.__get__(vae, vae.__class__)
    vae.blend_h = new_blend_h.__get__(vae, vae.__class__)

    # 适配AutoencoderKLQwenImage的属性，计算混合范围
    vae.blend_extent_enc = int(vae.tile_sample_min_height // vae.spatial_compression_ratio * tile_overlap_factor)
    vae.blend_extent_dec = int(vae.tile_sample_min_height * tile_overlap_factor)
    
    # 预生成并缓存权重（确保VAE已移到正确设备）
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    vae.blend_weights_enc = torch.linspace(0, 1, vae.blend_extent_enc, device=device, dtype=dtype)
    vae.blend_weights_dec = torch.linspace(0, 1, vae.blend_extent_dec, device=device, dtype=dtype)