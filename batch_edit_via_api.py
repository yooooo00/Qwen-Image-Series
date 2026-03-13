import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned[:max_len] or "prompt").rstrip("_")


def _post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode("utf-8")
        return json.loads(text)


def _decode_image_b64(image_b64: str) -> bytes:
    if "," in image_b64 and image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    return base64.b64decode(image_b64)


def _encode_image_file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_jobs(jobs_file: str) -> List[Dict[str, Any]]:
    with open(jobs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("jobs_file must be a JSON array.")
    jobs: List[Dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Job #{idx} must be a JSON object.")
        jobs.append(item)
    return jobs


def _extract_path_list(job: Dict[str, Any]) -> List[str]:
    if "image_paths" in job:
        paths = job["image_paths"]
        if isinstance(paths, str):
            return [x for x in paths.split(" ") if x.strip()]
        if isinstance(paths, list):
            return [str(x) for x in paths if str(x).strip()]
        raise ValueError("image_paths must be string or list")
    if "image_path" in job:
        return [str(job["image_path"])]
    return []


def _build_payload(
    job: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("job.prompt is required")

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "width": int(job.get("width", args.width)),
        "height": int(job.get("height", args.height)),
        "steps": int(job.get("steps", args.steps)),
        "cfg_scale": float(job.get("cfg_scale", args.cfg_scale)),
        "guidance_scale": float(job.get("guidance_scale", args.guidance_scale)),
    }

    seed = job.get("seed", args.seed)
    if seed is not None:
        payload["seed"] = int(seed)

    negative_prompt = job.get("negative_prompt", args.negative_prompt)
    if negative_prompt is not None:
        payload["negative_prompt"] = str(negative_prompt)

    if "images" in job or "image" in job:
        if "images" in job:
            payload["images"] = job["images"]
        else:
            payload["image"] = job["image"]
        return payload

    image_paths = _extract_path_list(job)
    if not image_paths:
        raise ValueError("job requires one of: image/images/image_path/image_paths")

    if args.send_path_directly:
        payload["image_paths"] = image_paths
        return payload

    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Image file(s) not found: {missing}")
    payload["images"] = [_encode_image_file_to_b64(p) for p in image_paths]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch test Qwen edit service via /qwenimage_edit and save outputs."
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:8000/qwenimage_edit",
        help="Edit API endpoint",
    )
    parser.add_argument("--jobs_file", type=str, required=True, help="JSON file with job list")
    parser.add_argument("--output_dir", type=str, default="qwenimage_edit_output")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed")
    parser.add_argument("--send_path_directly", action="store_true", default=False)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    jobs = _load_jobs(args.jobs_file)
    if not jobs:
        raise ValueError(f"No jobs found in: {args.jobs_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_meta: List[Dict[str, Any]] = []
    total = len(jobs)
    print(f"Total jobs: {total}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")

    for idx, job in enumerate(jobs, start=1):
        prompt = str(job.get("prompt", "")).strip()
        try:
            payload = _build_payload(job, args)
            start = time.time()
            resp = _post_json(args.api_url, payload, args.timeout)
            elapsed = time.time() - start

            code = resp.get("code")
            image_b64 = resp.get("result")
            if code != 201 or not image_b64:
                raise RuntimeError(f"API returned code={code}, message={resp.get('message')}")

            img_bytes = _decode_image_b64(image_b64)
            short_name = _sanitize_filename(prompt or f"job_{idx}")
            filename = f"{idx:03d}_{short_name}.png"
            output_path = os.path.join(args.output_dir, filename)
            with open(output_path, "wb") as f:
                f.write(img_bytes)

            perf = resp.get("performance") or {}
            print(
                f"[{idx}/{total}] saved: {filename} | "
                f"api_infer={perf.get('infer_time_s', 'NA')}s | wall={elapsed:.2f}s"
            )
            results_meta.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "file": filename,
                    "payload_summary": {
                        "width": payload.get("width"),
                        "height": payload.get("height"),
                        "steps": payload.get("steps"),
                        "num_images": len(payload.get("images", []))
                        if "images" in payload
                        else (1 if "image" in payload else len(payload.get("image_paths", []))),
                    },
                    "response": resp,
                }
            )
        except urllib.error.URLError as e:
            print(f"[{idx}/{total}] request failed: {e}")
            results_meta.append({"index": idx, "prompt": prompt, "error": str(e)})
        except Exception as e:
            print(f"[{idx}/{total}] failed: {e}")
            results_meta.append({"index": idx, "prompt": prompt, "error": str(e)})

        if args.sleep > 0 and idx < total:
            time.sleep(args.sleep)

    meta_path = os.path.join(args.output_dir, "results.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results_meta, f, ensure_ascii=False, indent=2)
    print(f"Done. Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
