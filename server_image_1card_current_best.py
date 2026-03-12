import argparse
import base64
import gc
import io
import json
import logging
import os
import random
import threading
import time
from typing import Any, Dict

import torch
import torch_npu  # noqa: F401
from flask import Flask, request
from mindiesd import CacheAgent, CacheConfig

from generate import EXAMPLE_PROMPT, _load_and_fuse_lora, init_pipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel


app = Flask(__name__)
request_semaphore = threading.Semaphore(1)
SERVICE = None


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _pil_to_base64_png(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class QwenImageSingleCardService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        _init_logging()
        self._validate_args()
        self.device = self._init_device()
        self.pipeline = self._load_pipeline()
        self._maybe_enable_cache()
        self._warmup()

    def _validate_args(self) -> None:
        if self.args.task not in ("Qwen-Image", "Qwen-Image-2512"):
            raise ValueError("Only text-to-image tasks are supported in this service.")
        if not os.path.exists(self.args.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint path not found: {self.args.ckpt_dir}")
        if self.args.lora_path and not os.path.exists(self.args.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {self.args.lora_path}")

    def _init_device(self) -> torch.device:
        device = torch.device(f"npu:{self.args.device_id}")
        torch.npu.set_device(device)
        logging.info(f"Single-card service initialized on {device}")
        return device

    def _load_pipeline(self):
        torch_dtype = torch.bfloat16 if self.args.torch_dtype == "bfloat16" else torch.float32
        transformer = QwenImageTransformer2DModel.from_pretrained(
            os.path.join(self.args.ckpt_dir, "transformer"),
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )

        pipeline = init_pipeline(
            task=self.args.task,
            ckpt_dir=self.args.ckpt_dir,
            transformer=transformer,
            vae=None,
            torch_dtype=torch_dtype,
        )
        if self.args.lora_path:
            _load_and_fuse_lora(pipeline, self.args.lora_path)

        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def _maybe_enable_cache(self) -> None:
        cond_cache = bool(int(os.environ.get("COND_CACHE", 0)))
        uncond_cache = bool(int(os.environ.get("UNCOND_CACHE", 0)))
        if not cond_cache and not uncond_cache:
            return

        cache_config = CacheConfig(
            method="dit_block_cache",
            blocks_count=int(os.environ.get("CACHE_BLOCKS_COUNT", 60)),
            steps_count=self.args.num_inference_steps,
            step_start=int(os.environ.get("CACHE_STEP_START", 10)),
            step_interval=int(os.environ.get("CACHE_STEP_INTERVAL", 3)),
            step_end=int(os.environ.get("CACHE_STEP_END", 35)),
            block_start=int(os.environ.get("CACHE_BLOCK_START", 10)),
            block_end=int(os.environ.get("CACHE_BLOCK_END", 50)),
        )
        self.pipeline.transformer.cache_cond = CacheAgent(cache_config) if cond_cache else None
        self.pipeline.transformer.cache_uncond = CacheAgent(cache_config) if uncond_cache else None
        logging.info(
            f"Cache enabled: COND_CACHE={cond_cache}, UNCOND_CACHE={uncond_cache}, "
            f"config={cache_config.__dict__}"
        )

    def _normalize_payload(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        if "prompt" not in raw or not str(raw["prompt"]).strip():
            raise ValueError("Missing required field: prompt")

        width = int(raw.get("width", self.args.width))
        height = int(raw.get("height", self.args.height))
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")

        steps = int(raw.get("steps", self.args.num_inference_steps))
        if steps <= 0:
            raise ValueError("steps must be a positive integer")

        seed = raw.get("seed")
        if seed is None:
            seed = random.randint(1, 2**31 - 1)
        seed = int(seed)

        cfg_scale = float(raw.get("cfg_scale", self.args.cfg_scale))
        guidance_scale = float(raw.get("guidance_scale", self.args.guidance_scale))
        if cfg_scale <= 0 or guidance_scale <= 0:
            raise ValueError("cfg_scale and guidance_scale must be > 0")

        infer_width = max((width // 16) * 16, 16)
        infer_height = max((height // 16) * 16, 16)

        return {
            "prompt": str(raw["prompt"]),
            "negative_prompt": raw.get("negative_prompt", self.args.negative_prompt),
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "guidance_scale": guidance_scale,
            "infer_width": infer_width,
            "infer_height": infer_height,
            "target_width": width,
            "target_height": height,
        }

    def _build_infer_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": payload["prompt"],
            "negative_prompt": payload["negative_prompt"],
            "true_cfg_scale": payload["cfg_scale"],
            "guidance_scale": payload["guidance_scale"],
            "generator": torch.Generator(device="cpu").manual_seed(payload["seed"]),
            "num_images_per_prompt": 1,
            "num_inference_steps": payload["steps"],
            "width": payload["infer_width"],
            "height": payload["infer_height"],
            "output_type": "pil",
        }

    def _warmup(self) -> None:
        warmup_prompt = self.args.warmup_prompt or EXAMPLE_PROMPT[self.args.task]["prompt"]
        payload = self._normalize_payload(
            {
                "prompt": warmup_prompt,
                "steps": self.args.warmup_steps,
                "width": self.args.width,
                "height": self.args.height,
                "seed": self.args.seed,
                "cfg_scale": self.args.cfg_scale,
                "guidance_scale": self.args.guidance_scale,
                "negative_prompt": self.args.negative_prompt,
            }
        )
        with torch.inference_mode():
            output = self.pipeline(**self._build_infer_inputs(payload))
        del output
        gc.collect()
        torch.npu.empty_cache()
        logging.info("Warm-up finished")

    def infer_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        torch.npu.synchronize()
        start = time.time()
        with torch.inference_mode():
            output = self.pipeline(**self._build_infer_inputs(payload))
        torch.npu.synchronize()
        infer_time = time.time() - start

        image = output.images[0]
        target_size = (payload["target_width"], payload["target_height"])
        if image.size != target_size:
            image = image.resize(target_size)
        image_b64 = _pil_to_base64_png(image)

        del output
        gc.collect()
        return {"image_b64": image_b64, "infer_time": infer_time}


@app.route("/health", methods=["GET", "POST"])
def health():
    return {"message": "health", "code": 20090, "result": "Healthy service"}


@app.route("/qwenimage", methods=["POST"])
def infer():
    global SERVICE
    response = {"code": 20090, "result": None, "message": "", "performance": {}}

    if SERVICE is None:
        response["code"] = 500
        response["message"] = "Service is not initialized"
        return json.dumps(response, ensure_ascii=False), 500, {"content-type": "application/json"}

    if not request_semaphore.acquire(blocking=False):
        response["code"] = 111
        response["message"] = "Too many requests"
        return json.dumps(response, ensure_ascii=False), 200, {"content-type": "application/json"}

    try:
        payload = SERVICE._normalize_payload(request.get_json(force=True, silent=False) or {})
        result = SERVICE.infer_once(payload)

        response["code"] = 201
        response["result"] = result["image_b64"]
        response["message"] = "success"
        response["performance"] = {
            "infer_time_s": round(result["infer_time"], 4),
            "seed": payload["seed"],
            "steps": payload["steps"],
            "infer_width": payload["infer_width"],
            "infer_height": payload["infer_height"],
            "target_width": payload["target_width"],
            "target_height": payload["target_height"],
        }
        return json.dumps(response, ensure_ascii=False), 200, {"content-type": "application/json"}
    except Exception as exc:
        logging.error(f"Inference failed: {exc}", exc_info=True)
        response["code"] = 402
        response["message"] = "failed"
        return json.dumps(response, ensure_ascii=False), 200, {"content-type": "application/json"}
    finally:
        request_semaphore.release()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image 1-card service (best current settings)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, default="Qwen-Image", choices=["Qwen-Image", "Qwen-Image-2512"])
    parser.add_argument("--ckpt_dir", type=str, default="/root/work/filestorage/cyy/Qwen-Image")
    parser.add_argument("--lora_path", type=str, default="/root/work/filestorage/cyy/Qwen-Image-Lightning/lora")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--warmup_prompt", type=str, default=None)
    parser.add_argument("--device_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    SERVICE = QwenImageSingleCardService(args)
    app.run(host=args.host, port=args.port, threaded=True)
