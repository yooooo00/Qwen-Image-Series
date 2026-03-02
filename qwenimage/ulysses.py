import torch
import torch_npu
import numpy as np
import functools
import os
from math import prod
from typing import Optional, Tuple, Union, List, Dict, Any


from diffusers import DiffusionPipeline
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from mindiesd import attention_forward

from qwenimage.transformer_qwenimage import (  
    QwenDoubleStreamAttnProcessor2_0,
    apply_rotary_emb_qwen,
    compute_text_seq_len_from_mask
)

from .distributed.parallel_mgr import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group
)

from .distributed.all_to_all import SeqAllToAll4D

# Transformer并行化函数（适配Qwen-Image结构）
def parallelize_qwen_image_transformer(pipe: DiffusionPipeline):
    """
    并行化Qwen-Image的Transformer，添加CFG和USP支持
    严格保留Qwen的双流处理逻辑
    """
    transformer = pipe.transformer

    @functools.wraps(transformer.__class__.forward)
    def parallel_forward(
        self,
        hidden_states: torch.Tensor,  # 图像流输入 (B, S_img, D_in)
        encoder_hidden_states: torch.Tensor,  # 文本流输入 (B, S_txt, D_joint)
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,  # Qwen专属：(frame, H, W)
        txt_seq_lens: Optional[List[int]] = None,  # Qwen专属：文本长度列表
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
        use_cache: bool = False,     
        if_cond: bool = True,    
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        if txt_seq_lens is not None:
            deprecate(
                "txt_seq_lens",
                "0.39.0",
                "Passing `txt_seq_lens` is deprecated and will be removed in version 0.39.0. "
                "Please use `encoder_hidden_states_mask` instead. "
                "The mask-based approach is more flexible and supports variable-length sequences.",
                standard_warn=False,
            )
        if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.img_in(hidden_states) 

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        # Use the encoder_hidden_states sequence length for RoPE computation and normalize mask
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)


        sp_world_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        txt_pad_len = 0
        # 原始图像序列长度
        img_seq_len = hidden_states.shape[1]   # 6889
        # 计算需要填充的长度（使其能被sp_world_size整除）
        img_pad_len = (sp_world_size - (img_seq_len % sp_world_size)) % sp_world_size

        if sp_world_size > 1:
            hidden_states = torch.chunk(hidden_states, sp_world_size, dim=1)[sp_rank]
            if modulate_index is not None:
                # print("ljf 对 modulate_index 进行 切分")
                modulate_index = torch.chunk(modulate_index, sp_world_size, dim=1)[sp_rank]

        # Construct joint attention mask once to avoid reconstructing in every block
        # This eliminates 60 GPU syncs during training while maintaining torch.compile compatibility
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            # Build joint mask: [text_mask, all_ones_for_image]
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask
      
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,  # Don't pass encoder_hidden_states_mask (using attention_mask instead)
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                    img_pad_len=img_pad_len,
                    # txt_pad_len=txt_pad_len
                )

            else:
                if not use_cache:  
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead)
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                        modulate_index=modulate_index,
                        img_pad_len=img_pad_len,
                        # txt_pad_len=txt_pad_len
                    )
                else:
                    if if_cond:
                        hidden_states, encoder_hidden_states = self.cache_cond.apply(
                            block,
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead)
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=attention_kwargs,
                            modulate_index=modulate_index,
                            img_pad_len=img_pad_len,
                            # txt_pad_len=txt_pad_len
                        )
                    else:
                        hidden_states, encoder_hidden_states = self.cache_uncond.apply(
                            block,
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=encoder_hidden_states_mask,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            joint_attention_kwargs=attention_kwargs,
                            modulate_index=modulate_index,
                            img_pad_len=img_pad_len,
                            # txt_pad_len=txt_pad_len
                        )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        
        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)  
        output = self.proj_out(hidden_states)  

        # 第一次 通信前 padding
        if img_pad_len > 0:  
            output_original_len = output.shape[1]  * sp_world_size - img_pad_len
            if sp_rank == sp_world_size - 1:

                output_original_len = (output.shape[1] + img_pad_len) * sp_world_size - img_pad_len

                output = torch.nn.functional.pad(
                    output, 
                    (0, 0, 0, img_pad_len, 0, 0)  # (左,右,上,下,前,后)，仅填充seq_len维度（dim=1）
                )

        output = output if sp_world_size <= 1 else get_sp_group().all_gather(output, dim=1)

        # 第一次 通信完 切除掉 padding
        if img_pad_len > 0:
            output = output[:, :output_original_len, :].contiguous()

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


    parallel_forward = parallel_forward.__get__(transformer)
    # 替换forward方法
    transformer.forward = parallel_forward

    # 替换注意力处理器为并行版本 
    for block in transformer.transformer_blocks:
        block.attn.processor = xFuserQwenDoubleStreamAttnProcessor()


# 并行注意力处理器（继承自Qwen原生处理器）
class xFuserQwenDoubleStreamAttnProcessor(QwenDoubleStreamAttnProcessor2_0):
    """
    继承Qwen原生双流注意力处理器，添加USP并行支持
    保持所有原生逻辑，仅在注意力计算环节引入并行化
    """
    def __init__(self):
        super().__init__()  # 调用Qwen原生处理器初始化
        self.sp_world_size = get_sequence_parallel_world_size()
        self.sp_rank = get_sequence_parallel_rank()
        self.ulysses_pg = get_sp_group().ulysses_group
        self.scatter_idx = 2
        self.gather_idx = 1

        self.algo = int(os.getenv('ALGO', 0))  
        head_dim = 128
        self.scale=(head_dim ** -0.5)

        self.fa_alltoall_overlap = int(os.getenv('OVERLAP', 0))
        self.rope_fuse = bool(int(os.environ.get('ROPE_FUSE', 0)))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # 图像流 (B, S_img/P, D)
        encoder_hidden_states: torch.FloatTensor,  # 文本流 (B, S_txt, D)
        encoder_hidden_states_mask: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (img_freqs, txt_freqs)
        img_pad_len = None,
        # txt_pad_len = None
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        txt_seq_len = encoder_hidden_states.shape[1]   
        
        img_original_len = hidden_states.shape[1]  * self.sp_world_size
        # 第一次 通信前 padding
        if img_pad_len > 0:
            img_original_len = hidden_states.shape[1]  * self.sp_world_size - img_pad_len
            if self.sp_rank == self.sp_world_size - 1:
            
                img_original_len = (hidden_states.shape[1] + img_pad_len) * self.sp_world_size - img_pad_len
                hidden_states = torch.nn.functional.pad(
                    hidden_states, 
                    (0, 0, 0, img_pad_len, 0, 0)  # (左,右,上,下,前,后)，仅填充seq_len维度（dim=1）
                )

        if self.fa_alltoall_overlap == True:
            qkv_sycn = False
            async_op = True
        else:
            qkv_sycn = True
            async_op = False
     
        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_query = SeqAllToAll4D.apply(
            self.ulysses_pg, img_query, self.scatter_idx, self.gather_idx, qkv_sycn
            # [B, S_image/ulysses_size, H, D] -->   [B, S_image, H/ulysses_size, D]
        )
        img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = SeqAllToAll4D.apply(
            self.ulysses_pg, img_key, self.scatter_idx, self.gather_idx, qkv_sycn
        )
        img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        img_value = SeqAllToAll4D.apply(
            self.ulysses_pg, img_value, self.scatter_idx, self.gather_idx, qkv_sycn
        )
        # 文本流QKV
        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))  # (B, S_txt, H, D_head)
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if self.fa_alltoall_overlap == True:
            img_query = img_query()
            img_key = img_key()

        # 第一次 通信完 切除掉 padding
        if img_pad_len > 0:
            img_query = img_query[:, :img_original_len, :, :].contiguous()
            img_key = img_key[:, :img_original_len, :, :].contiguous()     

        # QK归一化
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            # 缓存img_freqs的cos/sin预处理结果
            if self.rope_fuse:     
                img_cos, img_sin = self._preprocess_rope_cos_sin(img_freqs, hidden_states.device)  
                txt_cos, txt_sin = self._preprocess_rope_cos_sin(txt_freqs, hidden_states.device)

            # 调用apply_rotary_emb_qwen时传入预处理后的cos/sin
            img_query = apply_rotary_emb_qwen(
                img_query, img_freqs, use_real=False, 
                preprocessed_cos_sin=(img_cos, img_sin) if self.rope_fuse else None 
            )
            img_key = apply_rotary_emb_qwen(
                img_key, img_freqs, use_real=False,
                preprocessed_cos_sin=(img_cos, img_sin) if self.rope_fuse else None
            )

            txt_query = apply_rotary_emb_qwen(
                txt_query, txt_freqs, use_real=False,
                preprocessed_cos_sin=(txt_cos, txt_sin) if self.rope_fuse else None
            )
            txt_key = apply_rotary_emb_qwen(
                txt_key, txt_freqs, use_real=False,
                preprocessed_cos_sin=(txt_cos, txt_sin) if self.rope_fuse else None
            )

        if self.fa_alltoall_overlap == True:
            img_value = img_value()

        # 第一次 通信完 切除掉 padding
        if img_pad_len > 0:
            img_value = img_value[:, :img_original_len, :, :].contiguous()

        txt_query = torch.chunk(txt_query, self.sp_world_size, dim=2)[self.sp_rank]  # [B, S_text, H, D] -->  [B, S_text, H/ulysses_size, D]
        txt_key = torch.chunk(txt_key, self.sp_world_size, dim=2)[self.sp_rank]
        txt_value = torch.chunk(txt_value, self.sp_world_size, dim=2)[self.sp_rank]

        joint_query = torch.cat([txt_query, img_query], dim=1)  # (B, S_txt  + S_img, H/ulysses_size, D_head)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        if self.algo == 0:
            out = dispatch_attention_fn(
                joint_query,
                joint_key,
                joint_value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        elif self.algo == 1:
            out = attention_forward(
                joint_query, 
                joint_key, 
                joint_value,
                opt_mode="manual", 
                op_type="fused_attn_score", 
                layout="BSND"
            )
        elif self.algo == 3:
            joint_query = joint_query * self.scale
            out = attention_forward(
                joint_query, 
                joint_key, 
                joint_value,
                opt_mode="manual",
                op_type="ascend_laser_attention", 
                layout="BNSD")

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        txt_seq_len = txt_query.shape[1]

        text_out = context_layer[:, :txt_seq_len, :, :].contiguous()  # 强制连续
        image_out = context_layer[:, txt_seq_len:, :, :].contiguous()

        # 第二次 通信前 padding
        if img_pad_len > 0:
            image_out = torch.nn.functional.pad(
                image_out, 
                (0, 0, 0, 0, 0, img_pad_len, 0, 0)  # (左,右,上,下,前,后)，仅填充seq_len维度（dim=1）
            )

        img_attn_output = SeqAllToAll4D.apply(
            self.ulysses_pg, image_out, self.gather_idx, self.scatter_idx, True
            # [B,  S_image, H/ulysses_size, D] -->  [B, S_image/ulysses_size, H, D]
        ).flatten(2, 3).to(img_query.dtype)

        # 第二次 通信后 切除掉 padding
        if img_pad_len > 0:
            if self.sp_rank == self.sp_world_size - 1:
                img_attn_output_len = img_attn_output.shape[1]
                img_attn_output = img_attn_output[:, :img_attn_output_len-img_pad_len, :].contiguous()

        txt_attn_output = get_sp_group().all_gather(text_out, dim=2, async_op=async_op)  # (B, S_txt  , H/ulysses_size, D_head)  -->  (B, S_txt  , H, D_head)

        # 输出投影
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        if self.fa_alltoall_overlap == True:
            txt_attn_output = txt_attn_output()
        txt_attn_output = txt_attn_output.flatten(2, 3).to(img_query.dtype)
        txt_attn_output = attn.to_add_out(txt_attn_output)
      
        return img_attn_output, txt_attn_output