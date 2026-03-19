import argparse
import base64
import gc
import io
import json
import logging
import math
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from flask import Flask, request
from mindiesd import CacheAgent, CacheConfig
from PIL import Image

from generate import (
    EXAMPLE_PROMPT,
    _load_and_fuse_lora,
    _should_decode_on_this_rank,
    init_pipeline,
)
from qwenimage.distributed.parallel_mgr import ParallelConfig, init_parallel_env
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.ulysses import parallelize_qwen_image_transformer


EDIT_TASKS = ("Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511")

app = Flask(__name__)
request_semaphore = threading.Semaphore(1)
SERVICE = None


def _init_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format=f"[Rank {rank}][%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _pil_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _decode_base64_image(image_str: str, color_format: str = "RGB") -> Image.Image:
    if image_str.startswith("data:") and "," in image_str:
        image_str = image_str.split(",", 1)[1]
    img_bytes = base64.b64decode(image_str)
    return Image.open(io.BytesIO(img_bytes)).convert(color_format)


def _load_image_entry(entry: str, color_format: str = "RGB") -> Image.Image:
    if os.path.exists(entry):
        return Image.open(entry).convert(color_format)
    return _decode_base64_image(entry, color_format=color_format)


def _is_oom_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _close_payload_images(payload: Optional[Dict[str, Any]]) -> None:
    if not payload:
        return
    images = payload.get("images")
    if not isinstance(images, list):
        return
    for image in images:
        if isinstance(image, Image.Image):
            try:
                image.close()
            except Exception:
                pass
    payload["images"] = []


def _resolve_infer_size(
    width: int,
    height: int,
    max_infer_area: int,
    max_infer_width: int,
    max_infer_height: int,
    multiple: int = 16,
) -> tuple[int, int, bool]:
    infer_width = max((width // multiple) * multiple, multiple)
    infer_height = max((height // multiple) * multiple, multiple)
    size_limited = False

    max_blocks = max_infer_area // (multiple * multiple) if max_infer_area > 0 else 0
    block_width = infer_width // multiple
    block_height = infer_height // multiple

    if max_blocks > 0 and block_width * block_height > max_blocks:
        target_ratio = width / height
        while block_width * block_height > max_blocks:
            reduce_width_error = math.inf
            reduce_height_error = math.inf
            if block_width > 1:
                reduce_width_error = abs(((block_width - 1) / block_height) - target_ratio)
            if block_height > 1:
                reduce_height_error = abs((block_width / (block_height - 1)) - target_ratio)

            if reduce_width_error <= reduce_height_error:
                block_width -= 1
            else:
                block_height -= 1

        infer_width = block_width * multiple
        infer_height = block_height * multiple
        size_limited = True

    if max_infer_width <= 0 and max_infer_height <= 0:
        return infer_width, infer_height, size_limited

    max_width = max_infer_width if max_infer_width > 0 else infer_width
    max_height = max_infer_height if max_infer_height > 0 else infer_height

    if infer_width <= max_width and infer_height <= max_height:
        return infer_width, infer_height, size_limited

    scale = min(max_width / width, max_height / height)
    scaled_width = max(int(math.floor(width * scale)), multiple)
    scaled_height = max(int(math.floor(height * scale)), multiple)
    infer_width = max((scaled_width // multiple) * multiple, multiple)
    infer_height = max((scaled_height // multiple) * multiple, multiple)
    infer_width = min(infer_width, max_width)
    infer_height = min(infer_height, max_height)
    infer_width = max((infer_width // multiple) * multiple, multiple)
    infer_height = max((infer_height // multiple) * multiple, multiple)
    return infer_width, infer_height, True


def _parse_image_entries(raw: Dict[str, Any]) -> List[str]:
    if "images" in raw:
        images = raw["images"]
        if isinstance(images, str):
            return [images]
        if isinstance(images, list):
            return [str(x) for x in images if str(x).strip()]
        raise ValueError("images must be a string or list of strings")

    if "image" in raw:
        return [str(raw["image"])]

    if "image_paths" in raw:
        image_paths = raw["image_paths"]
        if isinstance(image_paths, str):
            return [x for x in image_paths.split(" ") if x.strip()]
        if isinstance(image_paths, list):
            return [str(x) for x in image_paths if str(x).strip()]
        raise ValueError("image_paths must be a string or list of strings")

    if "image_path" in raw:
        return [str(raw["image_path"])]

    return []


class QwenEditService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = int(os.getenv("RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.command_lock = threading.Lock()
        _init_logging(self.rank)

        self._validate_args()
        self.device = self._init_device()
        self._init_dist_and_parallel()
        self.pipeline = self._load_pipeline()
        self.decode_on_this_rank = _should_decode_on_this_rank(
            self.args.task, self.rank, self.world_size
        )

        self._maybe_enable_cache()
        self._warmup()

    def _validate_args(self) -> None:
        if self.args.task not in EDIT_TASKS:
            raise ValueError(f"Only edit tasks are supported. got={self.args.task}")
        if not os.path.exists(self.args.ckpt_dir):
            raise FileNotFoundError(f"Checkpoint path not found: {self.args.ckpt_dir}")
        if self.args.lora_path and not os.path.exists(self.args.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {self.args.lora_path}")
        if self.args.cfg_size * self.args.ulysses_size != self.world_size:
            raise ValueError(
                "cfg_size * ulysses_size must equal WORLD_SIZE. "
                f"Got cfg_size={self.args.cfg_size}, ulysses_size={self.args.ulysses_size}, "
                f"WORLD_SIZE={self.world_size}"
            )
        if self.args.max_infer_area is not None and self.args.max_infer_area < 0:
            raise ValueError("max_infer_area must be >= 0")
        if self.args.max_infer_area not in (None, 0) and self.args.max_infer_area < 16 * 16:
            raise ValueError("max_infer_area must be 0 or >= 256")
        for name in ("max_infer_width", "max_infer_height"):
            value = getattr(self.args, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")
            if value not in (None, 0) and value < 16:
                raise ValueError(f"{name} must be 0 or >= 16")

    def _init_device(self) -> torch.device:
        device_idx = self.local_rank if self.world_size > 1 else self.args.device_id
        device = torch.device(f"npu:{device_idx}")
        torch.npu.set_device(device)
        logging.info(
            f"Initializing on rank={self.rank}, world_size={self.world_size}, device={device}"
        )
        return device

    def _init_dist_and_parallel(self) -> None:
        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="hccl",
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
            )
            logging.info("Distributed process group initialized")

        if self.args.cfg_size > 1 or self.args.ulysses_size > 1:
            parallel_config = ParallelConfig(
                sp_degree=self.args.ulysses_size,
                ulysses_degree=self.args.ulysses_size,
                use_cfg_parallel=(self.args.cfg_size == 2),
                world_size=self.world_size,
            )
            init_parallel_env(parallel_config)
            logging.info(f"Parallel config initialized: {parallel_config.__dict__}")

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
        if self.args.ulysses_size > 1:
            parallelize_qwen_image_transformer(pipeline)
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

    def _load_images_from_entries(self, entries: List[str]) -> List[Image.Image]:
        if not entries:
            raise ValueError("Missing required image input. Use image/images/image_path/image_paths.")
        images = []
        for idx, entry in enumerate(entries, start=1):
            try:
                images.append(_load_image_entry(entry, color_format="RGB"))
            except Exception as exc:
                raise ValueError(f"Invalid image entry at index {idx}: {exc}") from exc
        return images

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

        infer_width, infer_height, size_limited = _resolve_infer_size(
            width,
            height,
            self.args.max_infer_area,
            self.args.max_infer_width,
            self.args.max_infer_height,
        )
        image_entries = _parse_image_entries(raw)
        images = self._load_images_from_entries(image_entries)
        if size_limited and self.rank == 0:
            logging.info(
                f"Request size limited from {width}x{height} to {infer_width}x{infer_height} "
                f"(max_infer_area={self.args.max_infer_area}, "
                f"max_infer_width={self.args.max_infer_width}, max_infer_height={self.args.max_infer_height})"
            )

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
            "size_limited": size_limited,
            "images": images,
        }

    def _build_infer_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "prompt": payload["prompt"],
            "negative_prompt": payload["negative_prompt"],
            "image": payload["images"],
            "true_cfg_scale": payload["cfg_scale"],
            "guidance_scale": payload["guidance_scale"],
            "generator": torch.Generator(device="cpu").manual_seed(payload["seed"]),
            "num_images_per_prompt": 1,
            "num_inference_steps": payload["steps"],
            "width": payload["infer_width"],
            "height": payload["infer_height"],
            "output_type": "pil" if self.decode_on_this_rank else "latent",
        }

    def _warmup(self) -> None:
        warmup_prompt = self.args.warmup_prompt or EXAMPLE_PROMPT[self.args.task]["prompt"]
        warmup_entries: List[str] = []

        if self.args.warmup_image_paths:
            warmup_entries = [x for x in self.args.warmup_image_paths.split(" ") if x.strip()]
        else:
            example_images = EXAMPLE_PROMPT[self.args.task].get("image")
            if example_images:
                candidates = [x for x in str(example_images).split(" ") if x.strip()]
                if candidates and all(os.path.exists(x) for x in candidates):
                    warmup_entries = candidates

        if not warmup_entries:
            logging.warning("Warm-up skipped because no warm-up images are available")
            return

        payload = self._normalize_payload(
            {
                "prompt": warmup_prompt,
                "images": warmup_entries,
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

        if dist.is_initialized():
            dist.barrier()
        logging.info("Warm-up finished")

    def infer_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self._build_infer_inputs(payload)
        output = None
        image = None

        try:
            torch.npu.synchronize()
            start = time.time()
            with torch.inference_mode():
                output = self.pipeline(**inputs)
            torch.npu.synchronize()
            infer_time = time.time() - start

            result_image_b64: Optional[str] = None
            if self.rank == 0:
                image = output.images[0]
                target_size = (payload["target_width"], payload["target_height"])
                if image.size != target_size:
                    image = image.resize(target_size)
                result_image_b64 = _pil_to_base64_png(image)

            return {"image_b64": result_image_b64, "infer_time": infer_time}
        except Exception as exc:
            if _is_oom_error(exc):
                logging.warning("OOM during edit inference, clearing request memory before re-raising")
            raise
        finally:
            if image is not None:
                del image
            if output is not None:
                del output
            del inputs
            _close_payload_images(payload)
            gc.collect()
            try:
                torch.npu.synchronize()
            except Exception:
                pass
            torch.npu.empty_cache()

    def broadcast_command_locked(self, cmd: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.world_size <= 1:
            return
        obj = [{"cmd": cmd, "payload": payload}]
        dist.broadcast_object_list(obj, src=0)

    def recv_command(self) -> Dict[str, Any]:
        if self.world_size <= 1:
            return {"cmd": "idle", "payload": None}
        obj = [None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
            logging.info("Destroyed distributed process group")


class Rank0Heartbeat:
    def __init__(self, service: QwenEditService, interval_s: int):
        self.service = service
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="rank0-heartbeat", daemon=True)

    def start(self) -> None:
        if self.service.rank == 0 and self.service.world_size > 1 and self.interval_s > 0:
            logging.info(f"Heartbeat enabled, interval={self.interval_s}s")
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=self.interval_s + 2)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            try:
                with self.service.command_lock:
                    self.service.broadcast_command_locked("idle", None)
            except Exception as exc:
                if self.stop_event.is_set():
                    return
                logging.error(f"Heartbeat broadcast failed: {exc}", exc_info=True)
                return


@app.route("/health", methods=["GET", "POST"])
def health():
    return {"message": "health", "code": 20090, "result": "Healthy service"}


def _handle_infer_request():
    global SERVICE
    response = {"code": 20090, "result": None, "message": "", "performance": {}}
    payload = None

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
        with SERVICE.command_lock:
            if SERVICE.world_size > 1:
                SERVICE.broadcast_command_locked("infer", payload)
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
            "size_limited": payload["size_limited"],
            "num_input_images": len(payload["images"]),
        }
        return json.dumps(response, ensure_ascii=False), 200, {"content-type": "application/json"}
    except Exception as exc:
        logging.error(f"Inference failed: {exc}", exc_info=True)
        if _is_oom_error(exc) and SERVICE is not None:
            _close_payload_images(payload)
            gc.collect()
            torch.npu.empty_cache()
        response["code"] = 402
        response["message"] = "failed"
        return json.dumps(response, ensure_ascii=False), 200, {"content-type": "application/json"}
    finally:
        request_semaphore.release()


@app.route("/qwenimage_edit", methods=["POST"])
def infer_edit():
    return _handle_infer_request()


@app.route("/qwenimage", methods=["POST"])
def infer_compat():
    return _handle_infer_request()


def _worker_loop(service: QwenEditService) -> None:
    while True:
        cmd = service.recv_command()
        command = cmd.get("cmd")
        payload = cmd.get("payload")

        if command == "stop":
            logging.info("Worker got stop command")
            break
        if command != "infer":
            continue

        try:
            service.infer_once(payload)
        except Exception as exc:
            if _is_oom_error(exc):
                _close_payload_images(payload)
                gc.collect()
                torch.npu.empty_cache()
            logging.error(f"Worker inference failed: {exc}", exc_info=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit 2-card service (best current settings)")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, default="Qwen-Image-Edit-2511", choices=list(EDIT_TASKS))
    parser.add_argument("--ckpt_dir", type=str, default="/root/work/filestorage/cyy/Qwen-Image-Edit-2511")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/root/work/filestorage/cyy/Qwen-Image-Edit-2511-Lightning/lora",
    )
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max_infer_area", type=int, default=0)
    parser.add_argument("--max_infer_width", type=int, default=0)
    parser.add_argument("--max_infer_height", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--warmup_prompt", type=str, default=None)
    parser.add_argument("--warmup_image_paths", type=str, default=None)

    parser.add_argument("--cfg_size", type=int, default=1, choices=[1, 2])
    parser.add_argument("--ulysses_size", type=int, default=2)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--heartbeat_interval_s", type=int, default=30)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    SERVICE = QwenEditService(args)
    heartbeat = None
    if SERVICE.rank == 0 and SERVICE.world_size > 1:
        heartbeat = Rank0Heartbeat(SERVICE, args.heartbeat_interval_s)
        heartbeat.start()

    try:
        if SERVICE.rank == 0:
            app.run(host=args.host, port=args.port, threaded=True)
        else:
            _worker_loop(SERVICE)
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        if SERVICE is not None and SERVICE.rank == 0 and SERVICE.world_size > 1 and dist.is_initialized():
            try:
                with SERVICE.command_lock:
                    SERVICE.broadcast_command_locked("stop", None)
            except Exception:
                pass
        if SERVICE is not None:
            SERVICE.close()
