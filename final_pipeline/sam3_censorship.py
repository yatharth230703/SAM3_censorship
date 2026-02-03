"""
SAM3 Censorship Engine - Modal Deployment
Optimized endpoint for AI-driven selective censorship

This endpoint accepts a list of specific categories to segment,
making it fast for targeted censorship workflows.
"""
from pathlib import Path
import modal

MODEL_TYPE = "facebook/sam3"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "transformers>=4.48.0",
        "pillow>=10.0.0",
        "numpy>=1.26.0",
        "huggingface_hub>=0.25.0",
        "accelerate>=0.34.0",
        "fastapi[standard]>=0.115.0",
    )
)

app = modal.App("sam3-censorship-engine", image=image)

cache_vol = modal.Volume.from_name("sam3-hf-cache", create_if_missing=True)
cache_dir = "/cache"

# SAM3 Official Category Registry - comprehensive list of supported categories
SAM3_CATEGORY_REGISTRY = [
    # People & Body Parts
    "person", "man", "woman", "child", "baby", "face", "head", "eye", "nose", "mouth",
    "ear", "hair", "hand", "finger", "arm", "leg", "foot", "body", "torso", "neck",
    
    # Clothing & Accessories
    "shirt", "t-shirt", "sweater", "hoodie", "jacket", "coat", "suit", "dress", "skirt",
    "pants", "jeans", "shorts", "underwear", "bra", "bikini", "swimsuit", "shoe", "sneaker",
    "boot", "sandal", "sock", "hat", "cap", "helmet", "glasses", "sunglasses", "watch",
    "jewelry", "necklace", "ring", "earring", "bracelet", "tie", "scarf", "glove", "belt",
    "bag", "purse", "handbag", "backpack", "wallet", "umbrella",
    
    # Vehicles & Transportation
    "car", "truck", "bus", "van", "motorcycle", "bicycle", "scooter", "airplane", "helicopter",
    "boat", "ship", "train", "subway", "tram", "taxi", "ambulance", "fire truck", "police car",
    "wheel", "tire", "license plate", "windshield", "mirror", "headlight",
    
    # Animals
    "dog", "cat", "bird", "horse", "cow", "sheep", "goat", "pig", "chicken", "duck",
    "fish", "shark", "whale", "dolphin", "elephant", "lion", "tiger", "bear", "deer",
    "rabbit", "mouse", "rat", "squirrel", "monkey", "gorilla", "zebra", "giraffe",
    "snake", "lizard", "frog", "turtle", "insect", "butterfly", "bee", "spider",
    
    # Furniture & Home
    "chair", "armchair", "sofa", "couch", "bed", "mattress", "pillow", "blanket", "table",
    "desk", "shelf", "cabinet", "drawer", "wardrobe", "closet", "door", "window", "curtain",
    "carpet", "rug", "lamp", "chandelier", "mirror", "clock", "painting", "picture frame",
    "vase", "plant pot", "fireplace", "stairs", "railing",
    
    # Electronics & Technology
    "tv", "television", "monitor", "screen", "computer", "laptop", "keyboard", "mouse",
    "phone", "smartphone", "tablet", "camera", "webcam", "microphone", "speaker", "headphones",
    "remote", "controller", "game console", "printer", "router", "cable", "charger",
    
    # Kitchen & Food
    "refrigerator", "oven", "stove", "microwave", "dishwasher", "sink", "faucet", "counter",
    "plate", "bowl", "cup", "mug", "glass", "bottle", "can", "jar", "fork", "knife", "spoon",
    "chopsticks", "pan", "pot", "kettle", "toaster", "blender", "cutting board",
    "apple", "banana", "orange", "grape", "strawberry", "watermelon", "lemon", "tomato",
    "carrot", "broccoli", "lettuce", "onion", "potato", "bread", "sandwich", "burger",
    "pizza", "pasta", "rice", "soup", "salad", "cake", "cookie", "donut", "ice cream",
    "chocolate", "candy", "cheese", "egg", "meat", "steak", "chicken", "fish", "sushi",
    
    # Outdoor & Nature
    "tree", "bush", "flower", "grass", "leaf", "branch", "trunk", "plant", "garden",
    "forest", "mountain", "hill", "rock", "stone", "sand", "beach", "ocean", "sea",
    "river", "lake", "pond", "waterfall", "sky", "cloud", "sun", "moon", "star",
    "rainbow", "rain", "snow", "ice", "fire", "smoke",
    
    # Buildings & Structures
    "building", "house", "apartment", "skyscraper", "tower", "church", "temple", "mosque",
    "castle", "bridge", "tunnel", "road", "street", "sidewalk", "crosswalk", "parking lot",
    "fence", "wall", "gate", "roof", "chimney", "balcony", "porch", "garage", "shed",
    "pool", "fountain", "statue", "monument", "sign", "billboard", "traffic light",
    "street light", "bench", "trash can", "mailbox", "fire hydrant",
    
    # Sports & Recreation
    "ball", "football", "soccer ball", "basketball", "baseball", "tennis ball", "golf ball",
    "volleyball", "rugby ball", "bat", "racket", "club", "stick", "net", "goal",
    "skateboard", "surfboard", "snowboard", "ski", "sled", "bicycle", "treadmill",
    "dumbbell", "barbell", "yoga mat", "helmet", "knee pad", "glove",
    
    # Office & School
    "book", "notebook", "paper", "pen", "pencil", "marker", "eraser", "ruler", "scissors",
    "stapler", "tape", "folder", "binder", "envelope", "stamp", "calculator", "globe",
    "whiteboard", "blackboard", "projector", "desk", "chair", "backpack", "lunchbox",
    
    # Medical & Safety
    "mask", "face mask", "surgical mask", "bandage", "syringe", "pill", "medicine bottle",
    "stethoscope", "thermometer", "wheelchair", "crutch", "first aid kit", "fire extinguisher",
    "safety vest", "hard hat", "goggles", "gloves",
    
    # Music & Entertainment
    "guitar", "piano", "keyboard", "drum", "violin", "flute", "trumpet", "microphone",
    "speaker", "headphones", "record", "cd", "dvd", "movie", "poster", "ticket",
    
    # Weapons & Sensitive Items (for censorship purposes)
    "gun", "pistol", "rifle", "knife", "sword", "weapon",
    
    # Documents & Text
    "document", "letter", "card", "id card", "credit card", "passport", "ticket",
    "receipt", "menu", "newspaper", "magazine", "poster", "label", "tag", "barcode", "qr code",
    
    # Miscellaneous
    "box", "container", "basket", "bucket", "rope", "chain", "wire", "pipe", "hose",
    "tool", "hammer", "screwdriver", "wrench", "saw", "drill", "ladder", "broom", "mop",
    "sponge", "towel", "tissue", "soap", "toothbrush", "toothpaste", "comb", "brush",
    "toy", "doll", "teddy bear", "lego", "puzzle", "balloon", "candle", "gift", "ribbon",
]


@app.cls(
    image=image.env({"HF_HUB_CACHE": cache_dir}),
    volumes={cache_dir: cache_vol},
    gpu="H200",
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class SAM3CensorshipEngine:
    @modal.enter()
    def load_model(self):
        """Load SAM3 models optimized for censorship workflow"""
        import torch
        from transformers import Sam3Model, Sam3Processor
        from accelerate import Accelerator
        
        print("Loading SAM3 Censorship Engine...")
        
        self.device = Accelerator().device
        print(f"Using device: {self.device}")
        
        # Load in FP16 for speed
        self.model = Sam3Model.from_pretrained(
            MODEL_TYPE,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.processor = Sam3Processor.from_pretrained(MODEL_TYPE)
        self.model.eval()
        
        print("SAM3 Censorship Engine ready!")
    
    @modal.method()
    def segment_for_censorship(self, image_bytes: bytes, target_categories: list,
                                confidence_threshold: float = 0.3):
        """
        Segment specific categories for censorship.
        
        This is the FAST endpoint - only processes categories you specify.
        
        Args:
            image_bytes: Image as bytes
            target_categories: List of specific categories to find (e.g., ["face", "license plate"])
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            {
                "segments": [
                    {
                        "id": 0,
                        "label": "face",
                        "confidence": 0.95,
                        "bbox": [x1, y1, x2, y2],
                        "area": 1234,
                        "center": [cx, cy],
                        "mask": [[0,1,1,...], ...]  # Binary mask
                    },
                    ...
                ],
                "visualization_image": "base64...",  # Image with segments highlighted
                "original_image": "base64...",  # Original image passed through
                "image_shape": [height, width],
                "categories_searched": ["face", "license plate"],
                "processing_time_seconds": 3.2
            }
        """
        import torch
        import numpy as np
        from PIL import Image, ImageDraw
        import io
        import base64
        import time
        
        start_time = time.time()
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        target_size = [list(image.size[::-1])]
        
        print(f"Processing image {image.size} for categories: {target_categories}")
        
        all_segments = []
        
        # Process only the requested categories (FAST!)
        for i, category in enumerate(target_categories):
            print(f"  [{i+1}/{len(target_categories)}] Segmenting '{category}'...")
            
            try:
                inputs = self.processor(
                    images=image,
                    text=category,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
                
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=confidence_threshold,
                    mask_threshold=0.5,
                    target_sizes=target_size
                )[0]
                
                masks = results.get("masks", [])
                boxes = results.get("boxes", torch.tensor([]))
                scores = results.get("scores", torch.tensor([]))
                
                for j in range(len(masks)):
                    mask = masks[j]
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                    
                    box = boxes[j] if j < len(boxes) else torch.zeros(4)
                    box_np = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)
                    
                    score = scores[j] if j < len(scores) else torch.tensor(0.0)
                    score_val = float(score.cpu().numpy() if hasattr(score, 'cpu') else score)
                    
                    area = int(mask_np.sum())
                    
                    if area > 50:
                        bbox_list = box_np.tolist()
                        if len(bbox_list) == 4:
                            bbox_xyxy = [int(b) for b in bbox_list]
                        else:
                            rows = np.any(mask_np, axis=1)
                            cols = np.any(mask_np, axis=0)
                            if rows.any() and cols.any():
                                rmin, rmax = np.where(rows)[0][[0, -1]]
                                cmin, cmax = np.where(cols)[0][[0, -1]]
                                bbox_xyxy = [int(cmin), int(rmin), int(cmax), int(rmax)]
                            else:
                                bbox_xyxy = [0, 0, 0, 0]
                        
                        all_segments.append({
                            "label": category,
                            "confidence": score_val,
                            "mask": mask_np,
                            "bbox": bbox_xyxy,
                            "area": area,
                        })
                        print(f"    Found {category} with confidence {score_val:.3f}")
                        
            except Exception as e:
                print(f"  Warning: Failed on '{category}': {str(e)[:50]}")
                continue
        
        # Filter segments with confidence >= 80% (0.8)
        all_segments = [seg for seg in all_segments if seg["confidence"] >= 0.8]
        
        # Sort by confidence and deduplicate
        all_segments.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Calculate centroids for all segments
        for seg in all_segments:
            bbox = seg["bbox"]
            seg["centroid"] = [
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ]
        
        unique_segments = []
        for seg in all_segments:
            is_duplicate = False
            for existing in unique_segments:
                # Check if segments are approximately the same size
                area_ratio = seg["area"] / max(existing["area"], 1)
                similar_size = 0.8 < area_ratio < 1.25  # Within 25% size difference
                
                if similar_size:
                    # Check IoU (overlap)
                    intersection = np.logical_and(seg["mask"], existing["mask"]).sum()
                    union = np.logical_or(seg["mask"], existing["mask"]).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    # Check centroid distance
                    dx = seg["centroid"][0] - existing["centroid"][0]
                    dy = seg["centroid"][1] - existing["centroid"][1]
                    centroid_distance = np.sqrt(dx**2 + dy**2)
                    
                    # Calculate relative distance (as percentage of bbox diagonal)
                    bbox_width = seg["bbox"][2] - seg["bbox"][0]
                    bbox_height = seg["bbox"][3] - seg["bbox"][1]
                    bbox_diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
                    relative_distance = centroid_distance / max(bbox_diagonal, 1)
                    
                    # Mark as duplicate only if: similar size + high overlap + very close centroids
                    if iou > 0.7 and relative_distance < 0.15:  # Centroids within 15% of bbox diagonal
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_segments.append(seg)
        
        # Build response
        segments_output = []
        
        # Create visualization
        vis_image = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        np.random.seed(42)
        colors = [
            tuple(np.random.randint(100, 255, 3).tolist() + [150])
            for _ in range(max(len(unique_segments), 1))
        ]
        
        for idx, seg in enumerate(unique_segments):
            mask_np = seg["mask"]
            bbox_xyxy = seg["bbox"]
            
            segment_data = {
                "id": idx,
                "label": seg["label"],
                "confidence": seg["confidence"],
                "bbox": bbox_xyxy,
                "area": seg["area"],
                "center": [
                    int((bbox_xyxy[0] + bbox_xyxy[2]) / 2),
                    int((bbox_xyxy[1] + bbox_xyxy[3]) / 2)
                ],
                "mask": mask_np.tolist(),
            }
            segments_output.append(segment_data)
            
            # Draw on visualization
            mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
            colored_mask = Image.new("RGBA", image.size, colors[idx])
            overlay.paste(colored_mask, (0, 0), mask_image)
            draw.rectangle(bbox_xyxy, outline=colors[idx][:3] + (255,), width=3)
            label_text = f"[{idx}] {seg['label']} ({seg['confidence']:.2f})"
            draw.text((bbox_xyxy[0], max(0, bbox_xyxy[1] - 18)), label_text, fill=(255, 255, 255, 255))
        
        vis_image = Image.alpha_composite(vis_image, overlay).convert("RGB")
        
        # Encode images
        vis_buffer = io.BytesIO()
        vis_image.save(vis_buffer, format="JPEG", quality=95)
        vis_base64 = base64.b64encode(vis_buffer.getvalue()).decode()
        
        orig_buffer = io.BytesIO()
        image.save(orig_buffer, format="JPEG", quality=95)
        orig_base64 = base64.b64encode(orig_buffer.getvalue()).decode()
        
        processing_time = time.time() - start_time
        print(f"Completed! Found {len(segments_output)} segments in {processing_time:.2f}s")
        
        return {
            "segments": segments_output,
            "visualization_image": vis_base64,
            "original_image": orig_base64,
            "image_shape": [image.height, image.width],
            "categories_searched": target_categories,
            "processing_time_seconds": round(processing_time, 2)
        }
    
    @modal.method()
    def get_category_registry(self):
        """Return the full list of supported SAM3 categories"""
        return {
            "categories": SAM3_CATEGORY_REGISTRY,
            "total_count": len(SAM3_CATEGORY_REGISTRY)
        }


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """FastAPI endpoints for censorship workflow"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional
    import base64
    
    web_app = FastAPI(
        title="SAM3 Censorship Engine",
        description="Fast, targeted image segmentation for censorship workflows"
    )
    
    class SegmentRequest(BaseModel):
        image: str  # Base64 encoded
        categories: List[str]  # Categories to segment
        confidence_threshold: float = 0.3
    
    @web_app.post("/segment")
    async def segment_endpoint(request: SegmentRequest):
        """
        Segment specific categories in an image.
        
        Send only the categories you need - this is the fast path!
        """
        try:
            image_bytes = base64.b64decode(request.image)
            engine = SAM3CensorshipEngine()
            result = engine.segment_for_censorship.remote(
                image_bytes,
                target_categories=request.categories,
                confidence_threshold=request.confidence_threshold
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.get("/categories")
    async def get_categories():
        """Get the full list of supported SAM3 categories"""
        return {
            "categories": SAM3_CATEGORY_REGISTRY,
            "total_count": len(SAM3_CATEGORY_REGISTRY)
        }
    
    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "engine": "SAM3 Censorship Engine"}
    
    return web_app


@app.local_entrypoint()
def test(image_path: str = "test_image.jpg", categories: str = "person,face"):
    """
    Test the censorship endpoint
    
    Usage:
        modal run sam3_censorship.py --image-path photo.jpg --categories "face,license plate,person"
    """
    from pathlib import Path
    import base64
    import time
    import json
    
    test_image = Path(image_path)
    if not test_image.exists():
        print(f"Image not found: {test_image}")
        return
    
    with open(test_image, "rb") as f:
        image_bytes = f.read()
    
    # Parse categories
    category_list = [c.strip() for c in categories.split(",")]
    
    print("=" * 60)
    print("SAM3 Censorship Engine Test")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Categories: {category_list}")
    print("=" * 60)
    
    engine = SAM3CensorshipEngine()
    
    start_time = time.time()
    result = engine.segment_for_censorship.remote(
        image_bytes,
        target_categories=category_list
    )
    total_time = time.time() - start_time
    
    print(f"\n✅ Segmentation complete!")
    print(f"Found {len(result['segments'])} segments")
    print(f"Processing time: {result['processing_time_seconds']}s")
    
    print(f"\nSegments found:")
    for seg in result['segments']:
        print(f"  [{seg['id']}] {seg['label']}: confidence={seg['confidence']:.3f}, "
              f"bbox={seg['bbox']}, area={seg['area']}")
    
    # Save outputs
    vis_data = base64.b64decode(result['visualization_image'])
    vis_path = test_image.parent / f"{test_image.stem}_censorship_preview.jpg"
    with open(vis_path, "wb") as f:
        f.write(vis_data)
    print(f"\n📸 Visualization: {vis_path}")
    
    # Save metadata WITHOUT masks (for AI agents to read)
    meta_path = test_image.parent / f"{test_image.stem}_segments.json"
    segments_light = [{k: v for k, v in s.items() if k != 'mask'} for s in result['segments']]
    with open(meta_path, "w") as f:
        json.dump({
            "segments": segments_light,
            "image_shape": result['image_shape'],
            "categories_searched": result['categories_searched'],
            "processing_time_seconds": result['processing_time_seconds']
        }, f, indent=2)
    print(f"📄 Segments metadata: {meta_path}")
    
    # Save full segments WITH masks (for censorship application only)
    full_path = test_image.parent / f"{test_image.stem}_segments_full.json"
    with open(full_path, "w") as f:
        json.dump({
            "segments": result['segments'],
            "image_shape": result['image_shape'],
            "categories_searched": result['categories_searched'],
            "processing_time_seconds": result['processing_time_seconds']
        }, f, indent=2)
    print(f"📄 Full segments (with masks): {full_path}")
    
    print(f"\n⏱  TOTAL TIME: {total_time:.2f}s")