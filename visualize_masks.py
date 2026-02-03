"""
Helper script to visualize segmentation masks from the SAM3 endpoint with timing
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import base64
from pathlib import Path
import time


def segment_and_visualize(
    image_path: str,
    endpoint_url: str,
    categories: list = None,
    output_path: str = "output.png"
):
    """
    Complete workflow: segment image and visualize results with detailed timing
    
    Args:
        image_path: Path to image
        endpoint_url: Your Modal endpoint URL
        categories: Optional custom categories to search for
        output_path: Where to save visualization
    """
    print(f"🖼️  Loading image: {image_path}")
    
    # Read and encode image
    start_time = time.time()
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    encode_time = time.time() - start_time
    print(f"⏱️  Image encoding took: {encode_time:.2f}s")
    
    # Prepare request
    payload = {"image": image_base64}
    if categories:
        payload["categories"] = categories
    
    # Send request
    print(f"\n📤 Sending request to endpoint...")
    request_start = time.time()
    response = requests.post(endpoint_url, json=payload)
    request_time = time.time() - request_start
    print(f"⏱️  Request + Processing took: {request_time:.2f}s")
    
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")
    
    # Parse response
    parse_start = time.time()
    result = response.json()
    parse_time = time.time() - parse_start
    print(f"⏱️  Response parsing took: {parse_time:.2f}s")
    
    print(f"\n✅ Segmentation complete!")
    print(f"   Total segments: {result['total_segments']}")
    print(f"   Image shape: {result['image_shape']}")
    
    # Show detected objects
    print(f"\n📋 Detected objects:")
    for seg in result['metadata']:
        print(f"   - {seg['label']}: confidence={seg['confidence']:.2f}, area={seg['area']}")
    
    # Save visualization
    save_start = time.time()
    vis_image_data = base64.b64decode(result['visualization_image'])
    with open(output_path, "wb") as f:
        f.write(vis_image_data)
    save_time = time.time() - save_start
    print(f"\n📸 Visualization saved to: {output_path}")
    print(f"⏱️  Saving visualization took: {save_time:.2f}s")
    
    # Save metadata
    import json
    metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.json"
    metadata_light = [{k: v for k, v in seg.items() if k != 'mask'} 
                     for seg in result['metadata']]
    with open(metadata_path, "w") as f:
        json.dump({
            "total_segments": result['total_segments'],
            "image_shape": result['image_shape'],
            "segments": metadata_light
        }, f, indent=2)
    print(f"📄 Metadata saved to: {metadata_path}")
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"⏱️  TOTAL TIME: {total_time:.2f}s")
    print(f"{'='*60}")
    print(f"   Breakdown:")
    print(f"   - Image encoding:      {encode_time:6.2f}s ({encode_time/total_time*100:5.1f}%)")
    print(f"   - Request/Processing:  {request_time:6.2f}s ({request_time/total_time*100:5.1f}%)")
    print(f"   - Response parsing:    {parse_time:6.2f}s ({parse_time/total_time*100:5.1f}%)")
    print(f"   - Saving outputs:      {save_time:6.2f}s ({save_time/total_time*100:5.1f}%)")
    print(f"{'='*60}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Segment and visualize images with SAM3")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--endpoint", required=True, help="Modal endpoint URL")
    parser.add_argument("--categories", nargs="+", help="Optional custom categories to search for")
    parser.add_argument("--output", default="output.png", help="Output path")
    
    args = parser.parse_args()
    
    # Run segmentation and visualization with timing
    segment_and_visualize(
        image_path=args.image,
        endpoint_url=args.endpoint,
        categories=args.categories,
        output_path=args.output
    )
    
    print(f"\n✅ Done! Check {args.output}")
