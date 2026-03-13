# Qwen-Image-Series

本 README 只保留实际使用所需内容，不包含环境准备步骤（使用镜像环境）。

## 权重链接
### Base 模型
- Qwen-Image  
Hugging Face: https://huggingface.co/Qwen/Qwen-Image  
ModelScope: https://modelscope.cn/models/Qwen/Qwen-Image
- Qwen-Image-Edit-2511  
Hugging Face: https://huggingface.co/Qwen/Qwen-Image-Edit-2511  
ModelScope: https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511

### Lightning LoRA
- Qwen-Image-Lightning  
Hugging Face: https://huggingface.co/lightx2v/Qwen-Image-Lightning  
ModelScope: https://www.modelscope.cn/models/lightx2v/Qwen-Image-Lightning
- Qwen-Image-Edit-2511-Lightning  
Hugging Face: https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning  
ModelScope: https://www.modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning

## 四种单图推理脚本
说明：先按需修改脚本内 `MODEL_PATH`、`lora_path`、输入图路径等，再执行。

1. Qwen-Image-Lightning单卡  
脚本: `run_1card.sh`
```bash
bash run_1card.sh
```

2. Qwen-Image-Lightning双卡（当前最佳）（仅export OVERLAP=1/0 差别——其他脚本可以在实际环境上调整该值测试性能）   
脚本: `run_image_2card_current_best.sh`
```bash
bash run_image_2card_current_best.sh
```

3. Qwen-Image-Edit-2511-Lightning单卡  
脚本: `run_edit_1card.sh`
```bash
bash run_edit_1card.sh
```

4. Qwen-Image-Edit-2511-Lightning双卡  
脚本: `run_edit_2card.sh`
```bash
bash run_edit_2card.sh
```

## 四种服务化拉起方式
默认端口统一为 `8000`。

1. Qwen-Image-Lightning单卡服务  
脚本: `run_server_image_1card_current_best.sh`
```bash
bash run_server_image_1card_current_best.sh
```

2. Qwen-Image-Lightning双卡服务  
脚本: `run_server_image_2card_current_best.sh`
```bash
bash run_server_image_2card_current_best.sh
```

3. Qwen-Image-Edit-2511-Lightning单卡服务  
脚本: `run_server_edit_1card_current_best.sh`
```bash
bash run_server_edit_1card_current_best.sh
```

4. Qwen-Image-Edit-2511-Lightning双卡服务  
脚本: `run_server_edit_2card_current_best.sh`
```bash
bash run_server_edit_2card_current_best.sh
```

## 服务接口说明
### 1) Qwen-Image-Lightning
- 路径: `POST /qwenimage`
- 默认 URL: `http://127.0.0.1:8000/qwenimage`

示例：
```bash
curl -X POST "http://127.0.0.1:8000/qwenimage" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"A cute cat drinking coffee, cinematic, 4k\",\"width\":1024,\"height\":1024,\"steps\":8}"
```

### 2) Qwen-Image-Edit-2511-Lightning
- 路径: `POST /qwenimage_edit`（兼容 `POST /qwenimage`）
- 默认 URL: `http://127.0.0.1:8000/qwenimage_edit`
- 请求需包含 `prompt` + 图像输入（`image/images/image_path/image_paths` 之一）

示例（服务端本地路径）：
```bash
curl -X POST "http://127.0.0.1:8000/qwenimage_edit" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Make Pikachu hold a sign says Qwen Edit is awesome\",\"image_path\":\"examples/yarn-art-pikachu.png\",\"width\":1024,\"height\":1024,\"steps\":8}"
```

## 批量测试脚本
### Qwen-Image-Lightning批量
- 脚本: `batch_generate_via_api.py`
- 输入: `prompts.txt`（每行一个 prompt）
```bash
python batch_generate_via_api.py --prompts_file prompts.txt
```

### Qwen-Image-Edit-2511-Lightning批量
- 脚本: `batch_edit_via_api.py`
- 输入: `jobs.json`（JSON 数组，每条任务至少包含 `prompt` + 图像输入）
- 示例文件: `edit_jobs_example.json`
```bash
python batch_edit_via_api.py --jobs_file edit_jobs_example.json
```

## 备注
- 服务返回结果中的 `result` 是 base64 编码图片。
- 批量测试脚本会保存生成图片并输出 `results.json` 记录耗时与返回信息。
