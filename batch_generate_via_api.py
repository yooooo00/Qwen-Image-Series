import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import List


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned[:max_len] or "prompt").rstrip("_")


def _load_prompts(prompts_file: str) -> List[str]:
    with open(prompts_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if prompts_file.lower().endswith(".json"):
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON prompts file must be a list of strings.")
        return [str(x).strip() for x in data if str(x).strip()]

    return [line.strip() for line in content.splitlines() if line.strip()]


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
    # Supports both pure base64 and data URI format.
    if "," in image_b64 and image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    return base64.b64decode(image_b64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch generate images via /qwenimage API and save to qwenimage_output."
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://127.0.0.1:26004/qwenimage",
        help="Qwen image API endpoint",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Prompt list file (.txt: one prompt per line, or .json: string list).",
    )
    parser.add_argument("--output_dir", type=str, default="qwenimage_output")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional fixed seed for all prompts.")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds per request.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    args = parser.parse_args()

    prompts = _load_prompts(args.prompts_file)
    if not prompts:
        raise ValueError(f"No prompts found in: {args.prompts_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_meta = []

    total = len(prompts)
    print(f"Total prompts: {total}")
    print(f"Output dir: {os.path.abspath(args.output_dir)}")

    for idx, prompt in enumerate(prompts, start=1):
        payload = {
            "prompt": prompt,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "guidance_scale": args.guidance_scale,
        }
        if args.negative_prompt is not None:
            payload["negative_prompt"] = args.negative_prompt
        if args.seed is not None:
            payload["seed"] = args.seed

        try:
            start = time.time()
            resp = _post_json(args.api_url, payload, args.timeout)
            elapsed = time.time() - start

            code = resp.get("code")
            image_b64 = resp.get("result")
            if code != 201 or not image_b64:
                raise RuntimeError(f"API returned code={code}, message={resp.get('message')}")

            img_bytes = _decode_image_b64(image_b64)
            short_name = _sanitize_filename(prompt)
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
