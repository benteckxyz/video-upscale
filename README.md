# AI Video Upscaler (Real-ESRGAN)

Upscale videos to 2x or 4x resolution using Real-ESRGAN AI models.
For example, upscale 1080p → 4K, or 720p → 1440p.

## Features

- **4x / 2x Upscaling** using Real-ESRGAN models
- **Anime-optimized model** for animation content
- **Audio preservation** — original audio is kept
- **FP16 precision** for fast CUDA inference
- **Tiling support** for low VRAM GPUs
- **Custom target resolution** override

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (recommended)
- FFmpeg in system PATH

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Upscale to 4x (e.g., 1080p -> 4K)
python upscale.py -i input.mp4 -o output_4k.mp4

# Upscale to 2x
python upscale.py -i input.mp4 -o output.mp4 --model general-x2

# Anime content
python upscale.py -i anime.mp4 -o anime_4k.mp4 --model anime-x4

# Scan folder and upscale all videos (saves to "upscaled" subfolder)
python upscale.py -f /path/to/videos
python upscale.py --folder /path/to/videos --model general-x2

# Low VRAM (use tiling)
python upscale.py -i input.mp4 -o output.mp4 --tile 256

# Force 4K resolution
python upscale.py -i input.mp4 -o output.mp4 --target-res 3840x2160

# List models
python upscale.py --list-models
```

## Available Models

| Model | Scale | Best For |
|---|---|---|
| `general-x4` | 4x | Real-world video (default) |
| `general-x2` | 2x | Gentle upscale |
| `anime-x4` | 4x | Anime / animation |

## Arguments

| Arg | Default | Description |
|---|---|---|
| `-i, --input` | required | Input video path |
| `-o, --output` | auto | Output video path |
| `-f, --folder` | — | Scan folder and upscale all videos |
| `--model` | `general-x4` | Model name |
| `--tile` | `0` (auto) | Tile size for low VRAM |
| `--target-res` | — | Force output resolution (e.g., `3840x2160`) |
| `--quality` | `18` | CRF quality (lower = better) |
| `--no-fp16` | — | Disable FP16 |
