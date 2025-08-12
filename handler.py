import os
import io
import json
import base64
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen, Request

import runpod

# ------------------------------------------------------------
# Environment hardening for Runpod Serverless
# ------------------------------------------------------------
# Serverless network volume is always mounted here:
PERSIST_ROOT = Path("/runpod-volume")

# Herd every cache to the persistent volume so cold starts don't redownload.
os.environ.setdefault("MODEL_DIR", str(PERSIST_ROOT / "SEEDVR2"))
os.environ.setdefault("HF_HOME", str(PERSIST_ROOT / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PERSIST_ROOT / "hf"))
os.environ.setdefault("XDG_CACHE_HOME", str(PERSIST_ROOT / ".cache"))
os.environ.setdefault("TORCH_HOME", str(PERSIST_ROOT / "torch"))

# Make CUDA device selection predictable for child processes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# If their VRAM probe is fragile, let it fail-soft (some forks respect this).
os.environ.setdefault("SEEDVR_SKIP_VRAM_QUERY", "1")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
COMFY_ROOT = Path("/workspace/seedvr2_upscaler")
REPO_DIR   = COMFY_ROOT / "ComfyUI-SeedVR2_VideoUpscaler"
MODEL_DIR  = Path(os.environ["MODEL_DIR"])
HF_HOME    = Path(os.environ["HF_HOME"])
XDG_CACHE  = Path(os.environ["XDG_CACHE_HOME"])
TORCH_HOME = Path(os.environ["TORCH_HOME"])

# Persist outputs too; change to /workspace/outputs if you want ephemeral.
OUT_ROOT   = PERSIST_ROOT / "outputs"

FFMPEG     = "ffmpeg"

# ------------------------------------------------------------
# Model filenames (adjust if upstream renames)
# ------------------------------------------------------------
MODEL_MAP = {
    "3b-fp16": "seedvr2_ema_3b_fp16.safetensors",
    "3b-fp8":  "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "7b-fp16": "seedvr2_ema_7b_fp16.safetensors",
    "7b-fp8":  "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
}

# ------------------------------------------------------------
# Repo bootstrap
# ------------------------------------------------------------
def _ensure_repo():
    """Clone the ComfyUI-SeedVR2_VideoUpscaler repo, install deps, and force its model dir to the persistent volume."""
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not REPO_DIR.exists():
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
            str(REPO_DIR)
        ])

    # Install repo requirements (idempotent).
    req_file = REPO_DIR / "requirements.txt"
    if req_file.exists():
        subprocess.check_call(["pip", "install", "-r", str(req_file)])

    # Ensure persistent caches exist.
    for p in [MODEL_DIR, HF_HOME, XDG_CACHE, TORCH_HOME]:
        p.mkdir(parents=True, exist_ok=True)

    # Belt-and-suspenders: force repo's default model folder to point at our persistent MODEL_DIR
    repo_models_dir = REPO_DIR / "models" / "SEEDVR2"
    try:
        if repo_models_dir.exists() and not repo_models_dir.is_symlink():
            # Move any stray downloads into the volume, then replace with a symlink.
            for f in repo_models_dir.glob("*"):
                target = MODEL_DIR / f.name
                if not target.exists():
                    f.replace(target)
            shutil.rmtree(repo_models_dir)
        if not repo_models_dir.exists():
            repo_models_dir.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(MODEL_DIR, repo_models_dir, target_is_directory=True)
    except Exception as e:
        # Worst case, we just rely on --model_dir flag.
        print(f"[warn] model symlink setup failed: {e}")

# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def _fetch_image_to_path(inp, dst_path: Path):
    """inp may be http(s) URL, data URL, or raw base64 string."""
    if isinstance(inp, str) and inp.strip().startswith("http"):
        req = Request(inp, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=60) as r, open(dst_path, "wb") as f:
            f.write(r.read())
        return
    if isinstance(inp, str) and inp.strip().startswith("data:image"):
        b64 = inp.split(",", 1)[1]
        data = base64.b64decode(b64)
        dst_path.write_bytes(data)
        return
    # Assume raw base64
    try:
        data = base64.b64decode(inp)
        dst_path.write_bytes(data)
    except Exception:
        raise ValueError("image must be http(s) URL, data URL, or base64-encoded bytes")

def _image_to_stub_video(image_path: Path, video_path: Path, fps: int = 8, frames: int = 8):
    """Create a short mp4 repeating the single image, to satisfy the video-only CLI."""
    tmp_dir = image_path.parent / "frames_tmp"
    tmp_dir.mkdir(exist_ok=True)
    for i in range(frames):
        shutil.copy(image_path, tmp_dir / f"frame_{i:04d}.png")
    cmd = [
        FFMPEG, "-y", "-r", str(fps),
        "-i", str(tmp_dir / "frame_%04d.png"),
        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "12",
        str(video_path),
    ]
    subprocess.check_call(cmd)
    shutil.rmtree(tmp_dir, ignore_errors=True)

def _run_seedvr(video_path: Path, model_key: str, short_edge: int, batch_size: int, preserve_vram: bool):
    out_dir = OUT_ROOT / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    model_file = MODEL_MAP.get(model_key, MODEL_MAP["3b-fp8"])
    cli = REPO_DIR / "inference_cli.py"

    cmd = [
        "python", str(cli),
        "--video_path", str(video_path),
        "--resolution", str(short_edge),
        "--batch_size", str(batch_size),
        "--model", model_file,
        "--model_dir", str(MODEL_DIR),
        "--output", str(out_dir),
        "--output_format", "png",
    ]
    if preserve_vram:
        cmd.append("--preserve_vram")

    # Helpful logging
    print(f"[info] launching SeedVR2 CLI with: {' '.join(cmd)}")
    print(f"[info] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[info] MODEL_DIR={MODEL_DIR}")

    subprocess.check_call(cmd)

    # Grab first produced PNG
    pngs = sorted(out_dir.glob("*.png"))
    if not pngs:
        pngs = sorted(out_dir.rglob("*.png"))
    if not pngs:
        raise RuntimeError("No PNG output from SeedVR2 CLI.")
    return pngs[0]

# ------------------------------------------------------------
# Runpod handler
# ------------------------------------------------------------
def handler(event):
    """
    Input JSON:
    {
      "image": "<url | data:image/... | base64>",   # required
      "short_edge": 1072,                           # optional; default 1072
      "model": "3b-fp8",                            # one of: 3b-fp8,3b-fp16,7b-fp8,7b-fp16
      "batch_size": 1,                              # optional
      "preserve_vram": true,                        # optional
      "return_base64": true                         # optional; if false, returns path on disk
    }
    """
    try:
        _ensure_repo()

        payload = event.get("input", {}) if isinstance(event, dict) else {}
        img_input = payload.get("image")
        if not img_input:
            return {"error": "missing 'image' in input."}

        short_edge   = int(payload.get("short_edge", 1072))
        model_key    = str(payload.get("model", "3b-fp8")).lower()
        batch_size   = int(payload.get("batch_size", 1))
        preserve_vram = bool(payload.get("preserve_vram", True))
        as_b64       = bool(payload.get("return_base64", True))

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            img_path = td / "input.png"
            _fetch_image_to_path(img_input, img_path)

            # Build a tiny stub video so we can reuse the video CLI
            vid_path = td / "stub.mp4"
            _image_to_stub_video(img_path, vid_path, fps=8, frames=max(5, batch_size))

            out_png = _run_seedvr(vid_path, model_key, short_edge, batch_size, preserve_vram)

            if as_b64:
                data = out_png.read_bytes()
                return {
                    "status": "success",
                    "model": model_key,
                    "short_edge": short_edge,
                    "output_image_base64": base64.b64encode(data).decode("utf-8")
                }
            else:
                return {
                    "status": "success",
                    "model": model_key,
                    "short_edge": short_edge,
                    "output_path": str(out_png)
                }

    except subprocess.CalledProcessError as e:
        return {"error": f"process failed: {e}", "returncode": e.returncode}
    except Exception as e:
        # Extra hint for common CUDA visibility issue
        msg = str(e)
        if "CUDA" in msg or "torch.cuda" in msg:
            msg += " | Hint: ensure this endpoint uses a GPU template with NVIDIA runtime."
        return {"error": msg}

# Kick off the Runpod serverless handler
runpod.serverless.start({"handler": handler})
