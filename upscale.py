"""
AI Media Upscaler using Real-ESRGAN
Upscale images and videos to 2x or 4x resolution (e.g., 1080p -> 4K)
Supports face enhancement via GFPGAN.
"""

import argparse
import os
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import subprocess
import tempfile
import shutil
import glob


# ─── Model Configurations ───────────────────────────────────────────────────
MODELS = {
    "general-x4": {
        "description": "Real-ESRGAN x4 - General purpose (best quality for real-world content)",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "model_init": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
    },
    "general-x2": {
        "description": "Real-ESRGAN x2 - General purpose 2x upscale",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "scale": 2,
        "model_init": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
    },
    "anime-x4": {
        "description": "Real-ESRGAN Anime x4 - Optimized for animation/anime content",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4,
        "model_init": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        "is_anime": True,
    },
}

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def get_video_info(video_path):
    """Get video metadata using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }


def extract_audio(video_path, audio_path):
    """Extract audio from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "copy", audio_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        # Try with aac encoding as fallback
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "aac", "-b:a", "192k", audio_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False


def merge_audio(video_path, audio_path, output_path):
    """Merge audio back into the upscaled video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "copy",
        "-shortest",
        output_path
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def setup_upscaler(model_name, gpu_id=0, fp16=True, tile=0, face_enhance=False):
    """Initialize the Real-ESRGAN upscaler and optionally GFPGAN face enhancer."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    config = MODELS[model_name]
    model = config["model_init"]()

    # Determine model path (auto-download if needed)
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(model_dir, exist_ok=True)

    upsampler = RealESRGANer(
        scale=config["scale"],
        model_path=config["url"],  # Will auto-download
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=fp16 and torch.cuda.is_available(),
        gpu_id=gpu_id if torch.cuda.is_available() else None,
    )

    # Optional GFPGAN face enhancement
    face_enhancer = None
    if face_enhance:
        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=config["scale"],
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        print("  Face enhance: GFPGAN v1.3 loaded")

    return upsampler, face_enhancer, config["scale"]


def is_image_file(filepath):
    """Check if a file is an image based on extension."""
    return os.path.splitext(filepath)[1].lower() in IMAGE_EXTENSIONS


def is_video_file(filepath):
    """Check if a file is a video based on extension."""
    return os.path.splitext(filepath)[1].lower() in VIDEO_EXTENSIONS


def upscale_image(input_path, output_path, model_name="general-x4",
                  gpu_id=0, fp16=True, tile=0, target_res=None,
                  face_enhance=False):
    """
    Upscale a single image using Real-ESRGAN.

    Args:
        input_path: Path to input image
        output_path: Path to output image
        model_name: Model to use
        gpu_id: GPU device id
        fp16: Use half precision
        tile: Tile size (0=auto)
        target_res: Target resolution as "WxH"
        face_enhance: Enable GFPGAN face enhancement
    """
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Cannot open image: {input_path}")

    h, w = img.shape[:2]

    print(f"\n{'='*60}")
    print(f"  AI Image Upscaler (Real-ESRGAN)")
    print(f"{'='*60}")
    print(f"  Input:       {input_path}")
    print(f"  Resolution:  {w}x{h}")
    print(f"  Model:       {model_name} ({MODELS[model_name]['description']})")
    print(f"  Face enhance: {'GFPGAN' if face_enhance else 'Off'}")
    print(f"  FP16:        {fp16}")
    print(f"  GPU:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Setup upscaler
    print(f"\n  Loading model...")
    upsampler, face_enhancer, scale = setup_upscaler(
        model_name, gpu_id, fp16, tile, face_enhance=face_enhance
    )

    out_w = w * scale
    out_h = h * scale

    final_w, final_h = out_w, out_h
    if target_res:
        parts = target_res.lower().split("x")
        final_w, final_h = int(parts[0]), int(parts[1])

    print(f"  Output:      {final_w}x{final_h}")
    print(f"{'='*60}\n")

    # Upscale
    print("  Upscaling...")
    try:
        if face_enhancer is not None:
            _, _, output = face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = upsampler.enhance(img, outscale=scale)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ⚠ GPU OOM. Trying with tile=256...")
            torch.cuda.empty_cache()
            upsampler.tile_size = 256
            if face_enhancer is not None:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=scale)
        else:
            raise

    # Resize to target if specified
    if target_res and (output.shape[1] != final_w or output.shape[0] != final_h):
        output = cv2.resize(output, (final_w, final_h), interpolation=cv2.INTER_LANCZOS4)

    # Save
    cv2.imwrite(output_path, output)

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  ✅ Upscale complete!")
    print(f"  Input:  {w}x{h} ({input_size:.2f} MB)")
    print(f"  Output: {output.shape[1]}x{output.shape[0]} ({output_size:.2f} MB)")
    print(f"  Saved:  {output_path}")
    print(f"{'='*60}\n")


def upscale_video(input_path, output_path, model_name="general-x4", 
                  gpu_id=0, fp16=True, tile=0, target_res=None,
                  face_enhance=False, output_quality=23):
    """
    Upscale a video using Real-ESRGAN.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video  
        model_name: Model to use (general-x4, general-x2, anime-x4)
        gpu_id: GPU device id
        fp16: Use half precision for faster inference
        tile: Tile size for processing (0=auto, use >0 for low VRAM)
        target_res: Target resolution as "WxH" (e.g., "3840x2160" for 4K)
        face_enhance: Enable face enhancement with GFPGAN
        output_quality: CRF value for output (lower = better quality, 18-28 range)
    """
    # Get video info
    info = get_video_info(input_path)
    print(f"\n{'='*60}")
    print(f"  AI Video Upscaler (Real-ESRGAN)")
    print(f"{'='*60}")
    print(f"  Input:       {input_path}")
    print(f"  Resolution:  {info['width']}x{info['height']}")
    print(f"  FPS:         {info['fps']}")
    print(f"  Frames:      {info['total_frames']}")
    print(f"  Model:       {model_name} ({MODELS[model_name]['description']})")
    print(f"  Face enhance: {'GFPGAN' if face_enhance else 'Off'}")
    print(f"  FP16:        {fp16}")
    print(f"  GPU:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Setup upscaler
    print(f"\n  Loading model...")
    upsampler, face_enhancer, scale = setup_upscaler(
        model_name, gpu_id, fp16, tile, face_enhance=face_enhance
    )

    out_width = info["width"] * scale
    out_height = info["height"] * scale

    # If target resolution specified, we'll resize after upscaling
    final_width = out_width
    final_height = out_height
    if target_res:
        parts = target_res.lower().split("x")
        final_width, final_height = int(parts[0]), int(parts[1])

    print(f"  Output:      {final_width}x{final_height}")
    print(f"{'='*60}\n")

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="upscale_")
    temp_audio = os.path.join(temp_dir, "audio.m4a")
    temp_video_no_audio = os.path.join(temp_dir, "video_no_audio.mp4")

    try:
        # Step 1: Extract audio
        print("  [1/4] Extracting audio...")
        has_audio = extract_audio(input_path, temp_audio)
        if has_audio:
            print("         Audio extracted successfully.")
        else:
            print("         No audio track found or extraction failed.")

        # Step 2: Process frames
        print("  [2/4] Upscaling frames...")
        cap = cv2.VideoCapture(input_path)

        # Setup video writer with ffmpeg for better quality
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            temp_video_no_audio, fourcc, info["fps"],
            (final_width, final_height)
        )

        pbar = tqdm(total=info["total_frames"], desc="         Upscaling", unit="frame")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Upscale frame
            try:
                if face_enhancer is not None:
                    _, _, output = face_enhancer.enhance(
                        frame, has_aligned=False, only_center_face=False, paste_back=True
                    )
                else:
                    output, _ = upsampler.enhance(frame, outscale=scale)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  ⚠ GPU OOM at frame {frame_idx}. Trying with tile=256...")
                    torch.cuda.empty_cache()
                    upsampler.tile_size = 256
                    if face_enhancer is not None:
                        _, _, output = face_enhancer.enhance(
                            frame, has_aligned=False, only_center_face=False, paste_back=True
                        )
                    else:
                        output, _ = upsampler.enhance(frame, outscale=scale)
                else:
                    raise

            # Resize to target resolution if specified
            if target_res and (output.shape[1] != final_width or output.shape[0] != final_height):
                output = cv2.resize(output, (final_width, final_height), interpolation=cv2.INTER_LANCZOS4)

            writer.write(output)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        # Step 3: Re-encode with ffmpeg for better compression
        print("  [3/4] Encoding output video (H.264)...")
        if has_audio:
            # Merge with audio
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video_no_audio,
                "-i", temp_audio,
                "-c:v", "libx264",
                "-crf", str(output_quality),
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                output_path
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video_no_audio,
                "-c:v", "libx264",
                "-crf", str(output_quality),
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                output_path
            ]

        subprocess.run(cmd, capture_output=True, check=True)

        # Step 4: Report
        out_info = get_video_info(output_path)
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        output_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"\n  [4/4] Done!")
        print(f"{'='*60}")
        print(f"  ✅ Upscale complete!")
        print(f"  Input:  {info['width']}x{info['height']} ({input_size:.1f} MB)")
        print(f"  Output: {out_info['width']}x{out_info['height']} ({output_size:.1f} MB)")
        print(f"  Saved:  {output_path}")
        print(f"{'='*60}\n")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def scan_and_upscale(folder_path, model_name="general-x4", gpu_id=0,
                     fp16=True, tile=0, target_res=None, output_quality=18,
                     face_enhance=False):
    """
    Scan a folder for video and image files and upscale them all.
    Outputs are saved to a 'upscaled' subfolder.
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    all_extensions = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

    # Find all media files
    media_files = sorted([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower() in all_extensions
    ])

    video_files = [f for f in media_files if is_video_file(os.path.join(folder_path, f))]
    image_files = [f for f in media_files if is_image_file(os.path.join(folder_path, f))]

    if not media_files:
        print(f"No media files found in: {folder_path}")
        print(f"Supported video formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        print(f"Supported image formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
        return

    # Create output subfolder
    output_dir = os.path.join(folder_path, "upscaled")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Folder Scan Mode")
    print(f"{'='*60}")
    print(f"  Folder:    {folder_path}")
    print(f"  Videos:    {len(video_files)} found")
    print(f"  Images:    {len(image_files)} found")
    print(f"  Output:    {output_dir}")
    print(f"  Model:     {model_name}")
    print(f"{'='*60}\n")

    results = []
    for idx, filename in enumerate(media_files, 1):
        input_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_dir, filename)

        # For images, change output to .png for best quality
        if is_image_file(input_path):
            base = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base}.png")

        # Skip if already upscaled
        if os.path.exists(output_path):
            print(f"  [{idx}/{len(media_files)}] Skipping (already exists): {filename}")
            results.append((filename, "skipped"))
            continue

        print(f"\n  [{idx}/{len(media_files)}] Processing: {filename}")
        try:
            if is_image_file(input_path):
                upscale_image(
                    input_path=input_path,
                    output_path=output_path,
                    model_name=model_name,
                    gpu_id=gpu_id,
                    fp16=fp16,
                    tile=tile,
                    target_res=target_res,
                    face_enhance=face_enhance,
                )
            else:
                upscale_video(
                    input_path=input_path,
                    output_path=output_path,
                    model_name=model_name,
                    gpu_id=gpu_id,
                    fp16=fp16,
                    tile=tile,
                    target_res=target_res,
                    output_quality=output_quality,
                    face_enhance=face_enhance,
                )
            results.append((filename, "success"))
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            results.append((filename, f"failed: {e}"))

    # Summary
    print(f"\n{'='*60}")
    print(f"  Batch Upscale Summary")
    print(f"{'='*60}")
    success = sum(1 for _, s in results if s == "success")
    skipped = sum(1 for _, s in results if s == "skipped")
    failed = sum(1 for _, s in results if s.startswith("failed"))
    print(f"  ✅ Success:  {success}")
    print(f"  ⏭  Skipped:  {skipped}")
    print(f"  ❌ Failed:   {failed}")
    print(f"  Output dir:  {output_dir}")
    print(f"{'='*60}\n")


def list_models():
    """Print available models."""
    print("\nAvailable Models:")
    print("-" * 60)
    for name, config in MODELS.items():
        print(f"  {name:15s} | {config['scale']}x | {config['description']}")
    print("-" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Media Upscaler - Upscale images and videos using Real-ESRGAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upscale a single image
  python upscale.py -i photo.jpg -o photo_4k.png

  # Upscale with face enhancement (natural skin)
  python upscale.py -i photo.jpg -o photo_4k.png --face-enhance

  # Upscale video to 4x resolution (e.g., 1080p -> 4K)
  python upscale.py -i input.mp4 -o output_4k.mp4

  # Upscale video with face enhancement
  python upscale.py -i input.mp4 -o output_4k.mp4 --face-enhance

  # Upscale to 2x resolution
  python upscale.py -i input.mp4 -o output.mp4 --model general-x2

  # Upscale anime/animation video
  python upscale.py -i anime.mp4 -o anime_4k.mp4 --model anime-x4

  # Scan folder and upscale all images & videos
  python upscale.py -f /path/to/media
  python upscale.py --folder /path/to/media --face-enhance

  # Low VRAM mode (use tiling)
  python upscale.py -i input.mp4 -o output.mp4 --tile 256

  # Force specific target resolution
  python upscale.py -i photo.jpg -o photo_4k.png --target-res 3840x2160

  # List available models
  python upscale.py --list-models
        """
    )

    parser.add_argument("-i", "--input", help="Input image or video path")
    parser.add_argument("-o", "--output", help="Output path")
    parser.add_argument("-f", "--folder", help="Scan folder and upscale all images & videos (saves to 'upscaled' subfolder)")
    parser.add_argument("--model", default="general-x4",
                        choices=list(MODELS.keys()),
                        help="Model to use (default: general-x4)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for faster inference (default: True)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 (use FP32)")
    parser.add_argument("--tile", type=int, default=0,
                        help="Tile size (0=auto, use 256/512 for low VRAM GPUs)")
    parser.add_argument("--target-res", type=str, default=None,
                        help="Target output resolution, e.g., '3840x2160' for 4K")
    parser.add_argument("--quality", type=int, default=18,
                        help="Output quality CRF (lower=better, default: 18)")
    parser.add_argument("--face-enhance", action="store_true",
                        help="Enhance faces using GFPGAN (fixes plastic/smooth skin)")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models")

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    # Folder scan mode
    if args.folder:
        fp16 = args.fp16 and not args.no_fp16
        scan_and_upscale(
            folder_path=args.folder,
            model_name=args.model,
            gpu_id=args.gpu_id,
            fp16=fp16,
            tile=args.tile,
            target_res=args.target_res,
            output_quality=args.quality,
            face_enhance=args.face_enhance,
        )
        return

    if not args.input:
        parser.print_help()
        return

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    fp16 = args.fp16 and not args.no_fp16

    # Auto-detect image vs video
    if is_image_file(args.input):
        if not args.output:
            base, _ = os.path.splitext(args.input)
            args.output = f"{base}_upscaled.png"

        upscale_image(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            gpu_id=args.gpu_id,
            fp16=fp16,
            tile=args.tile,
            target_res=args.target_res,
            face_enhance=args.face_enhance,
        )
    else:
        if not args.output:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_upscaled{ext}"

        upscale_video(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            gpu_id=args.gpu_id,
            fp16=fp16,
            tile=args.tile,
            target_res=args.target_res,
            output_quality=args.quality,
            face_enhance=args.face_enhance,
        )


if __name__ == "__main__":
    main()
