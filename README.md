# AI Media Upscaler (Real-ESRGAN)

Upscale images and videos to 2x or 4x resolution using Real-ESRGAN AI models.
For example, upscale 1080p → 4K, or enhance a photo to 4x its original size.

## Features

- **Image & Video Upscaling** — supports both images and videos
- **4x / 2x Upscaling** using Real-ESRGAN models
- **GFPGAN Face Enhancement** — restore natural skin texture (fixes plastic look)
- **Anime-optimized model** for animation content
- **Folder batch mode** — scan a folder and upscale all media
- **Audio preservation** — original audio is kept for videos
- **FP16 precision** for fast CUDA inference
- **Tiling support** for low VRAM GPUs
- **Custom target resolution** override
- **Auto-detect** input type (image vs video)

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (recommended)
- FFmpeg in system PATH (for video processing)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Upscale a single image
python upscale.py -i photo.jpg -o photo_4k.png

# Upscale with face enhancement (natural skin texture)
python upscale.py -i photo.jpg -o photo_4k.png --face-enhance

# Upscale video to 4x (e.g., 1080p -> 4K)
python upscale.py -i input.mp4 -o output_4k.mp4

# Upscale video with face enhancement
python upscale.py -i input.mp4 -o output_4k.mp4 --face-enhance

# Upscale to 2x
python upscale.py -i input.mp4 -o output.mp4 --model general-x2

# Anime content
python upscale.py -i anime.mp4 -o anime_4k.mp4 --model anime-x4

# Scan folder and upscale all images & videos (saves to "upscaled" subfolder)
python upscale.py -f /path/to/media
python upscale.py --folder /path/to/media --face-enhance

# Low VRAM (use tiling)
python upscale.py -i input.mp4 -o output.mp4 --tile 256

# Force 4K resolution
python upscale.py -i photo.jpg -o photo_4k.png --target-res 3840x2160

# List models
python upscale.py --list-models
```

## Available Models

| Model | Scale | Best For |
|---|---|---|
| `general-x4` | 4x | Real-world content (default) |
| `general-x2` | 2x | Gentle upscale |
| `anime-x4` | 4x | Anime / animation |

## Supported Formats

| Type | Extensions |
|---|---|
| **Images** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif` |
| **Videos** | `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.ts` |

## Arguments

| Arg | Default | Description |
|---|---|---|
| `-i, --input` | required | Input image or video path |
| `-o, --output` | auto | Output path |
| `-f, --folder` | — | Scan folder and upscale all media |
| `--model` | `general-x4` | Model name |
| `--face-enhance` | off | Enhance faces with GFPGAN (natural skin) |
| `--tile` | `0` (auto) | Tile size for low VRAM |
| `--target-res` | — | Force output resolution (e.g., `3840x2160`) |
| `--quality` | `18` | CRF quality for video (lower = better) |
| `--no-fp16` | — | Disable FP16 |

## Benchmark

| Task | Input | Output | Time | GPU |
|---|---|---|---|---|
| Video upscale (8s clip) | 720p | 4K (3840×2160) | ~8 min | RTX 4070 Ti |

**Test system:** Intel i7-14700K, 32GB RAM, NVIDIA RTX 4070 Ti, FP16 enabled.
