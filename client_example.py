"""
Example client for calling the SAM2 segmentation endpoint
"""
import requests
import base64
from pathlib import Path
import json


def segment_image(image_path: str, endpoint_url: str, point_coords=None, point_labels=None, mode="point"):
    """
    Send an image to the Modal endpoint for segmentation
    
    Args:
        image_path: Path to the image file
        endpoint_url: Your Modal endpoint URL
        point_coords: List of [x, y] coordinates (optional)
        point_labels: List of labels (1=foreground, 0=background) (optional)
        mode: "point" for point-based or "auto" for automatic segmentation
    """
    # Read and encode image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Prepare request payload
    payload = {
        "image": image_base64,
        "mode": mode,
    }
    
    if point_coords:
        payload["point_coords"] = point_coords
    if point_labels:
        payload["point_labels"] = point_labels
    
    # Send request
    response = requests.post(endpoint_url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


# Example usage
if __name__ == "__main__":
    # Replace with your Modal endpoint URL after deployment
    ENDPOINT_URL = "https://your-username--sam2-segmentation-segment-endpoint.modal.run"
    
    # Example 1: Point-based segmentation
    result = segment_image(
        image_path="test_image.jpg",
        endpoint_url=ENDPOINT_URL,
        point_coords=[[100, 100]],  # Click at position (100, 100)
        point_labels=[1],  # Foreground point
        mode="point"
    )
    print("Point-based segmentation result:")
    print(f"Score: {result['score']}")
    print(f"Image shape: {result['image_shape']}")
    
    # Example 2: Automatic segmentation (segment everything)
    result_auto = segment_image(
        image_path="test_image.jpg",
        endpoint_url=ENDPOINT_URL,
        mode="auto"
    )
    print(f"\nAutomatic segmentation found {result_auto['count']} objects")
