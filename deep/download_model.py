import os
import sys
import urllib.request

MODEL_URLS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
]


def main() -> int:
    out_dir = os.path.join("deep", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inswapper_128.onnx")

    if os.path.exists(out_path):
        print(f"model already exists: {os.path.abspath(out_path)}")
        return 0

    last_error = "unknown"
    for url in MODEL_URLS:
        try:
            print(f"downloading: {url}")
            urllib.request.urlretrieve(url, out_path)
            print(f"saved: {os.path.abspath(out_path)}")
            return 0
        except Exception as e:
            last_error = str(e)
            print(f"failed url: {url} ({e})")

    raise RuntimeError(f"all model urls failed: {last_error}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"failed: {e}", file=sys.stderr)
        raise SystemExit(1)
