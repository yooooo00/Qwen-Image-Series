export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
# 0: 普通,  1: FA, 2: LA
export ALGO=1
export OVERLAP=1

# 算子优化
export ROPE_FUSE=1
export ADALN_FUSE=1

# 算法优化
export COND_CACHE=1
export UNCOND_CACHE=1

# Qwen-Image-Edit-2511   2卡：ulysses2
model_path="/home//weights/Qwen-Image-Edit-2511"
export ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master-port 29508 generate.py \
    --task Qwen-Image-Edit-2511 \
    --ckpt_dir ${model_path} \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --image "./examples/yarn-art-pikachu.png" \
    --seed 2918787680 \
    --cfg_scale 4.0 \
    --num_images_per_prompt 1 \
    --num_inference_steps 40 \
    --output_file "./pikachu_2511_ulysses2_FA_Cache.png" \
    --vae_tiling \
    --vae_slicing \
    --ulysses_size 2


# Qwen-Image-Edit-2509  单卡
model_path="/home/weights/Qwen-Image-Edit-2509"
export ASCEND_RT_VISIBLE_DEVICES=2
torchrun --nproc_per_node=1 --master-port 29509 generate.py \
    --task Qwen-Image-Edit-2509 \
    --ckpt_dir ${model_path} \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --image "./examples/yarn-art-pikachu.png" \
    --seed 2918787680 \
    --cfg_scale 4.0 \
    --num_images_per_prompt 1 \
    --num_inference_steps 40 \
    --output_file "./pikachu_2509_1190x2032.png" \
    --vae_tiling \
    --vae_slicing \
    --width 1190 \
    --height 2032


# Qwen-Image-Edit
model_path="/mnt/share/weights/Qwen-Image-Edit"
export ASCEND_RT_VISIBLE_DEVICES=9
torchrun --nproc_per_node=1 --master-port 29510 generate_优化.py \
    --task Qwen-Image-Edit \
    --ckpt_dir ${model_path} \
    --prompt "Make Pikachu hold a sign that says 'Qwen Edit is awesome', yarn art style, detailed, vibrant color" \
    --negative_prompt " " \
    --image "./examples/yarn-art-pikachu.png" \
    --seed 0 \
    --cfg_scale 4.0 \
    --num_images_per_prompt 1 \
    --num_inference_steps 50 \
    --output_file "./pikachu_2507_1190x2032.png" \
    --vae_tiling \
    --width 1190 \
    --height 2032

# Qwen-Image/Qwen-Image-2512  2卡：cfg2
model_path="/home/g00887675/logan/weights/Qwen-Image-2512"
export ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master-port 29508 generate.py \
    --task Qwen-Image \
    --ckpt_dir ${model_path} \
    --seed 42 \
    --cfg_scale 4.0 \
    --num_inference_steps 50 \
    --height 1024 \
    --width 1024 \
    --output_file "./output/text_to_image.png" \
    --cfg_size 2 \
    --num_images_per_prompt 2 \
    --vae_tiling \
    # --quant_dit_path ./quant_w8a8_withoutData_use_disable_quant_layers \

    
# 当为2时有bug， 第一个有五张，第二个有三张
# Qwen-Image-Layered
model_path="/home/g00887675/logan/weights/Qwen-Image-Layered"
export ASCEND_RT_VISIBLE_DEVICES=9
torchrun --nproc_per_node=1 --master-port 29508 generate_优化.py \
    --task Qwen-Image-Layered \
    --ckpt_dir ${model_path} \
    --image ./examples/1.png \
    --seed 777 \
    --num_images_per_prompt 2 \
    --num_inference_steps 50 \
    --cfg_scale 4.0 \
    --layers 2 \
    --resolution 640 \
    --color_format "RGBA" \
    --output_file "./image_layered" \
    --vae_tiling \