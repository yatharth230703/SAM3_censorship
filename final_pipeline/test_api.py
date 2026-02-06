#!/usr/bin/env python3
"""
Test the deployed censorship API
Edit IMAGE_PATH and PROMPT below to test with your image
"""
import requests
import base64
from pathlib import Path

# ============ EDIT THESE ============
IMAGE_PATH = "img3.png"
PROMPT = "Blur the whole body of the woman with the shortest hair"
# ====================================

API_URL = "https://vasub0723--censorship-pipeline-complete-fastapi-app.modal.run"
METHOD = "blur"

def censor_image(image_path, prompt, method="blur"):
    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    print(f"Sending request to API...")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Method: {method}")
    print()
    
    # Call API
    response = requests.post(
        f"{API_URL}/censor",
        json={
            "image": image_base64,
            "prompt": prompt,
            "method": method
        },
        timeout=300
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    
    print(f"✅ Success!")
    print(f"Categories: {result['categories_selected']}")
    print(f"Segments found: {result['segments_found']}")
    print(f"Segments censored: {result['segments_censored']}")
    print(f"Time: {result['processing_time_seconds']}s")
    
    # Save results
    stem = Path(image_path).stem
    
    censored_path = f"{stem}_censored.jpg"
    with open(censored_path, "wb") as f:
        f.write(base64.b64decode(result['censored_image']))
    print(f"\n📸 Censored: {censored_path}")
    
    vis_path = f"{stem}_visualization.jpg"
    with open(vis_path, "wb") as f:
        f.write(base64.b64decode(result['visualization_image']))
    print(f"📸 Visualization: {vis_path}")
    
    return result

if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        print(f"Error: Image not found: {IMAGE_PATH}")
        print(f"Edit IMAGE_PATH in the script to point to your image")
        exit(1)
    
    censor_image(IMAGE_PATH, PROMPT, METHOD)



