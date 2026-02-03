"""
Final Censorship Application Script

This script takes the original image, SAM3 segment data, and the list of
segment IDs to censor, then applies the censorship (blur/pixelate/black box).

Usage:
    python apply_censorship.py <original_image> <segments_json> <censor_decision_json> [--method blur|pixelate|blackbox]
"""
import json
import sys
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import numpy as np


def apply_blur(image: Image.Image, mask: np.ndarray, strength: int = 30) -> Image.Image:
    """Apply gaussian blur to masked region"""
    # Create blurred version
    blurred = image.filter(ImageFilter.GaussianBlur(radius=strength))
    
    # Convert to numpy for masking
    img_array = np.array(image)
    blur_array = np.array(blurred)
    
    # Apply mask
    mask_3d = np.stack([mask] * 3, axis=-1) if len(img_array.shape) == 3 else mask
    result = np.where(mask_3d, blur_array, img_array)
    
    return Image.fromarray(result.astype(np.uint8))


def apply_pixelate(image: Image.Image, mask: np.ndarray, block_size: int = 15) -> Image.Image:
    """Apply pixelation to masked region"""
    img_array = np.array(image)
    result = img_array.copy()
    
    # Get bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return image
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Pixelate the region
    for y in range(rmin, rmax + 1, block_size):
        for x in range(cmin, cmax + 1, block_size):
            # Get block bounds
            y_end = min(y + block_size, rmax + 1)
            x_end = min(x + block_size, cmax + 1)
            
            # Check if any pixel in block is in mask
            block_mask = mask[y:y_end, x:x_end]
            if block_mask.any():
                # Average color of block
                block = img_array[y:y_end, x:x_end]
                avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
                
                # Apply to masked pixels only
                for by in range(y, y_end):
                    for bx in range(x, x_end):
                        if mask[by, bx]:
                            result[by, bx] = avg_color
    
    return Image.fromarray(result)


def apply_blackbox(image: Image.Image, mask: np.ndarray, color: tuple = (0, 0, 0)) -> Image.Image:
    """Apply solid color box over masked region"""
    img_array = np.array(image)
    
    mask_3d = np.stack([mask] * 3, axis=-1)
    result = np.where(mask_3d, np.array(color), img_array)
    
    return Image.fromarray(result.astype(np.uint8))


def apply_censorship(
    original_image_path: str,
    segments_json_path: str,
    censor_decision_json_path: str,
    method: str = "blur",
    output_path: str = None
) -> str:
    """
    Apply censorship to an image based on SAM3 segments and censor decisions.
    
    Args:
        original_image_path: Path to original image
        segments_json_path: Path to SAM3 segments JSON WITH masks (*_segments_full.json)
        censor_decision_json_path: Path to censor decision JSON (with segment IDs)
        method: "blur", "pixelate", or "blackbox"
        output_path: Optional output path (default: <original>_censored.jpg)
    
    Returns:
        Path to the censored image
    """
    # Load original image
    image = Image.open(original_image_path).convert("RGB")
    
    # Load segments (need the full data with masks)
    with open(segments_json_path, "r") as f:
        segments_data = json.load(f)
    
    # Handle different JSON structures
    if "segments" in segments_data:
        segments = segments_data["segments"]
    else:
        segments = segments_data
    
    # Load censor decisions
    with open(censor_decision_json_path, "r") as f:
        censor_data = json.load(f)
    
    segments_to_censor = set(censor_data.get("segments_to_censor", []))
    
    if not segments_to_censor:
        print("No segments to censor!")
        return original_image_path
    
    print(f"Applying {method} censorship to {len(segments_to_censor)} segments...")
    
    # Apply censorship to each selected segment
    result_image = image
    
    for seg in segments:
        seg_id = seg["id"]
        if seg_id not in segments_to_censor:
            continue
        
        print(f"  Censoring segment [{seg_id}] {seg['label']}...")
        
        # Get mask
        mask = np.array(seg["mask"], dtype=bool)
        
        # Apply censorship method
        if method == "blur":
            result_image = apply_blur(result_image, mask, strength=30)
        elif method == "pixelate":
            result_image = apply_pixelate(result_image, mask, block_size=15)
        elif method == "blackbox":
            result_image = apply_blackbox(result_image, mask, color=(0, 0, 0))
        else:
            print(f"Unknown method: {method}, using blur")
            result_image = apply_blur(result_image, mask)
    
    # Save result
    if output_path is None:
        orig_path = Path(original_image_path)
        output_path = orig_path.parent / f"{orig_path.stem}_censored.jpg"
    
    result_image.save(output_path, quality=95)
    print(f"\n✅ Censored image saved: {output_path}")
    
    return str(output_path)


def main():
    """CLI interface"""
    if len(sys.argv) < 4:
        print("Usage: python apply_censorship.py <original_image> <segments_full_json> <censor_decision_json> [--method blur|pixelate|blackbox]")
        print("\nMethods:")
        print("  blur     - Gaussian blur (default)")
        print("  pixelate - Mosaic/pixelation effect")
        print("  blackbox - Solid black box")
        print("\nExample:")
        print("  python apply_censorship.py photo.jpg photo_segments_full.json photo_censor_decision.json --method blur")
        print("\nNote: Use *_segments_full.json (with masks), not *_segments.json (metadata only)")
        sys.exit(1)
    
    original_image = sys.argv[1]
    segments_json = sys.argv[2]
    censor_decision_json = sys.argv[3]
    
    # Parse method
    method = "blur"
    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]
    
    # Validate files
    for f in [original_image, segments_json, censor_decision_json]:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)
    
    print("=" * 60)
    print("Censorship Application")
    print("=" * 60)
    print(f"Image: {original_image}")
    print(f"Method: {method}")
    print("=" * 60)
    
    output = apply_censorship(
        original_image,
        segments_json,
        censor_decision_json,
        method=method
    )
    
    print(f"\n🎉 Done! Censored image: {output}")


if __name__ == "__main__":
    main()