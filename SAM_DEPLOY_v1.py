"""
Deploy Facebook's Segment Anything Model 3 (SAM3) on Modal
for image segmentation with GPU acceleration.

SAM3 supports:
- Text-based concept segmentation (NEW!)
- Point/box-based segmentation (like SAM2)
- Automatic mask generation
"""
from pathlib import Path
import modal

# Model configuration
MODEL_TYPE = "facebook/sam3"

# Define the image with all dependencies
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
        "fastapi[standard]>=0.115.0",  # Required for web endpoints
    )
)

app = modal.App("sam3-segmentation", image=image)

# Create volumes for caching
cache_vol = modal.Volume.from_name("sam3-hf-cache", create_if_missing=True)
cache_dir = "/cache"


@app.cls(
    image=image.env({"HF_HUB_CACHE": cache_dir}),
    volumes={cache_dir: cache_vol},
    gpu="H200",  # Using H200 GPU for maximum performance
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],  # Add HF token
)
class SAM3Segmenter:
    @modal.enter()
    def load_model(self):
        """Download and initialize SAM3 models on container startup"""
        import torch
        from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor
        from accelerate import Accelerator
        
        print(f"Loading SAM3 models: {MODEL_TYPE}")
        
        # Get device
        self.device = Accelerator().device
        print(f"Using device: {self.device}")
        
        # Load SAM3 model for text-based concept segmentation
        print("Loading Sam3Model (text-based segmentation)...")
        self.model = Sam3Model.from_pretrained(MODEL_TYPE).to(self.device)
        self.processor = Sam3Processor.from_pretrained(MODEL_TYPE)
        
        # Load SAM3 Tracker model for point/box-based segmentation
        print("Loading Sam3TrackerModel (point/box-based segmentation)...")
        self.tracker_model = Sam3TrackerModel.from_pretrained(MODEL_TYPE).to(self.device)
        self.tracker_processor = Sam3TrackerProcessor.from_pretrained(MODEL_TYPE)
        
        print("SAM3 models loaded successfully!")
    
    @modal.method()
    def segment_with_text(self, image_bytes: bytes, text_prompt: str, threshold: float = 0.5, mask_threshold: float = 0.5):
        """
        Segment an image using text prompts (NEW in SAM3!)
        
        Args:
            image_bytes: Image as bytes
            text_prompt: Text description of what to segment (e.g., "ear", "laptop", "person")
            threshold: Confidence threshold for detections (default: 0.5)
            mask_threshold: Threshold for mask binarization (default: 0.5)
        
        Returns:
            Dictionary with segmentation masks, boxes, and scores
        """
        import torch
        import numpy as np
        from PIL import Image
        import io
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process with text prompt
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        return {
            "masks": [mask.cpu().numpy().tolist() for mask in results["masks"]],
            "boxes": results["boxes"].cpu().numpy().tolist(),
            "scores": results["scores"].cpu().numpy().tolist(),
            "count": len(results["masks"]),
            "image_shape": list(image.size[::-1]),  # [height, width]
        }
    
    @modal.method()
    def segment_with_points(self, image_bytes: bytes, point_coords: list = None, point_labels: list = None, box_coords: list = None):
        """
        Segment an image using point or box prompts (like SAM2)
        
        Args:
            image_bytes: Image as bytes
            point_coords: List of [x, y] coordinates for prompts (optional)
            point_labels: List of labels (1 for foreground, 0 for background) (optional)
            box_coords: Bounding box as [x_min, y_min, x_max, y_max] (optional)
        
        Returns:
            Dictionary with segmentation masks and metadata
        """
        import torch
        import numpy as np
        from PIL import Image
        import io
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Prepare inputs
        input_points = None
        input_labels = None
        input_boxes = None
        
        if point_coords is not None:
            # Format: [[[[x, y]]]] for single point
            input_points = [[[[x, y] for x, y in point_coords]]]
            input_labels = [[[label for label in (point_labels if point_labels else [1] * len(point_coords))]]]
        
        if box_coords is not None:
            # Format: [[[x_min, y_min, x_max, y_max]]]
            input_boxes = [[box_coords]]
        
        # If no prompts provided, use center point as default
        if input_points is None and input_boxes is None:
            w, h = image.size
            input_points = [[[[w // 2, h // 2]]]]
            input_labels = [[[1]]]
        
        # Process inputs
        inputs = self.tracker_processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.tracker_model(**inputs)
        
        # Post-process masks
        masks = self.tracker_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]
        
        # Get scores
        scores = outputs.iou_scores.cpu().numpy()
        
        # Return the best mask (highest score)
        best_mask_idx = scores.argmax()
        
        return {
            "mask": masks[0, best_mask_idx].numpy().tolist(),
            "score": float(scores[0, best_mask_idx]),
            "all_scores": scores[0].tolist(),
            "image_shape": list(image.size[::-1]),  # [height, width]
        }
    
    @modal.method()
    def segment_and_label_everything(self, image_bytes: bytes, custom_categories: list = None):
        """
        Segment everything in the image using text-based prompts for semantic labels.
        
        This method uses SAM3's text-based segmentation to find objects by category,
        giving you meaningful labels like "person", "dog", "car" instead of just "object".
        
        Args:
            image_bytes: Image as bytes
            custom_categories: Optional list of categories to search for. 
                              If None, uses a default comprehensive list.
        
        Returns:
            Dictionary with:
            - metadata: List of segments with semantic labels, masks, boxes, scores
            - visualization_image: Base64 encoded image with colored segments
        """
        import torch
        import numpy as np
        from PIL import Image, ImageDraw
        import io
        import base64
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        
        print(f"Processing image of size: {image.size}")
        
        # Categories to search for - these are semantic labels SAM3 can recognize
        categories = custom_categories or [
            # People and body parts
            "person", "face", "hand", "arm", "leg", "foot", "hair",
            # Clothing
            "shirt", "sweater", "jacket", "coat", "pants", "jeans", "shorts", 
            "dress", "skirt", "shoe", "sneaker", "boot", "hat", "cap", "glasses",
            # Animals
            "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", 
            "bear", "zebra", "giraffe", "fish",
            # Vehicles
            "car", "truck", "bus", "motorcycle", "bicycle", "airplane", 
            "boat", "train",
            # Furniture
            "chair", "couch", "sofa", "bed", "table", "desk", "bench",
            # Electronics
            "tv", "laptop", "computer", "phone", "keyboard", "mouse", "remote",
            # Kitchen items
            "bottle", "cup", "mug", "plate", "bowl", "fork", "knife", "spoon",
            # Food
            "apple", "banana", "orange", "pizza", "sandwich", "cake", "donut",
            # Outdoor/Nature
            "tree", "plant", "flower", "grass", "sky", "cloud", "water", 
            "mountain", "rock", "sand", "road", "sidewalk",
            # Buildings/Structures
            "building", "house", "window", "door", "wall", "fence", "bridge",
            # Sports equipment
            "ball", "baseball bat", "tennis racket", "skateboard", "surfboard",
            # Other common objects
            "bag", "backpack", "umbrella", "book", "clock", "vase", 
            "scissors", "teddy bear", "toothbrush",
        ]
        
        print(f"Searching for {len(categories)} object categories...")
        
        # Store all detected segments
        all_segments = []
        
        # Process each category using text-based segmentation
        for i, category in enumerate(categories):
            if i % 20 == 0:
                print(f"  Processing category {i+1}/{len(categories)}: {category}...")
            
            try:
                # Use the text-based model to find this category
                inputs = self.processor(
                    images=image, 
                    text=category, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-process with lower threshold to catch more objects
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.3,  # Lower threshold to catch more
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]
                
                # Add detected objects for this category
                for j, (mask, box, score) in enumerate(zip(
                    results.get("masks", []), 
                    results.get("boxes", torch.tensor([])), 
                    results.get("scores", torch.tensor([]))
                )):
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                    box_np = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)
                    score_val = float(score.cpu().numpy() if hasattr(score, 'cpu') else score)
                    
                    # Calculate area
                    area = int(mask_np.sum())
                    
                    if area > 50:  # Filter very tiny detections
                        all_segments.append({
                            "label": category,
                            "confidence": score_val,
                            "mask": mask_np,
                            "bbox": box_np.tolist(),
                            "area": area,
                        })
                        
            except Exception as e:
                # Some categories might not work, skip them
                print(f"    Warning: Could not process '{category}': {str(e)[:50]}")
                continue
        
        print(f"Found {len(all_segments)} total detections before deduplication...")
        
        # Sort by confidence (highest first)
        all_segments.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Remove duplicate/overlapping detections (keep highest confidence)
        unique_segments = []
        for seg in all_segments:
            is_duplicate = False
            for existing in unique_segments:
                # Check IoU
                intersection = np.logical_and(seg["mask"], existing["mask"]).sum()
                union = np.logical_or(seg["mask"], existing["mask"]).sum()
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.7:  # High overlap = duplicate
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_segments.append(seg)
        
        print(f"Found {len(unique_segments)} unique segments after deduplication")
        
        # Prepare metadata (without numpy arrays for JSON serialization)
        metadata = []
        
        # Create visualization
        vis_image = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", vis_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Generate colors
        np.random.seed(42)
        colors = [
            tuple(np.random.randint(50, 255, 3).tolist() + [128])
            for _ in range(max(len(unique_segments), 1))
        ]
        
        for idx, seg in enumerate(unique_segments):
            mask_np = seg["mask"]
            bbox = seg["bbox"]
            
            # Ensure bbox is a list of 4 values [x_min, y_min, x_max, y_max]
            if len(bbox) == 4:
                bbox_xyxy = [int(b) for b in bbox]
            else:
                # Calculate from mask if bbox is wrong format
                rows = np.any(mask_np, axis=1)
                cols = np.any(mask_np, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bbox_xyxy = [int(cmin), int(rmin), int(cmax), int(rmax)]
                else:
                    bbox_xyxy = [0, 0, 0, 0]
            
            segment_data = {
                "id": idx,
                "label": seg["label"],
                "confidence": seg["confidence"],
                "bbox": bbox_xyxy,
                "area": seg["area"],
                "mask": mask_np.tolist(),
                "center": [
                    int((bbox_xyxy[0] + bbox_xyxy[2]) / 2),
                    int((bbox_xyxy[1] + bbox_xyxy[3]) / 2)
                ]
            }
            metadata.append(segment_data)
            
            # Draw visualization
            mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))
            colored_mask = Image.new("RGBA", image.size, colors[idx])
            overlay.paste(colored_mask, (0, 0), mask_image)
            
            # Draw bounding box
            draw.rectangle(bbox_xyxy, outline=colors[idx][:3] + (255,), width=2)
            
            # Draw label with confidence
            label_text = f"{seg['label']} ({seg['confidence']:.2f})"
            draw.text(
                (bbox_xyxy[0], max(0, bbox_xyxy[1] - 15)), 
                label_text, 
                fill=(255, 255, 255, 255)
            )
        
        # Composite visualization
        vis_image = Image.alpha_composite(vis_image, overlay)
        vis_image = vis_image.convert("RGB")
        
        # Convert to base64
        buffer = io.BytesIO()
        vis_image.save(buffer, format="JPEG", quality=95)
        vis_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        print(f"Completed! Generated {len(metadata)} labeled segments")
        
        return {
            "metadata": metadata,
            "visualization_image": vis_base64,
            "total_segments": len(metadata),
            "image_shape": [image.height, image.width]
        }


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """
    FastAPI app for image segmentation endpoint
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import base64
    
    web_app = FastAPI()
    
    class ImageRequest(BaseModel):
        image: str  # Base64 encoded image
        categories: list = None  # Optional custom categories
    
    @web_app.post("/segment")
    async def segment_endpoint(request: ImageRequest):
        """
        Endpoint for automatic image segmentation with semantic labels
        
        Returns:
        {
            "metadata": [...],
            "visualization_image": "base64_encoded_colored_image",
            "total_segments": 15,
            "image_shape": [height, width]
        }
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(request.image)
            
            segmenter = SAM3Segmenter()
            result = segmenter.segment_and_label_everything.remote(
                image_bytes, 
                custom_categories=request.categories
            )
            
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return web_app


@app.local_entrypoint()
def test(image_path: str = "test_image.jpg"):
    """
    Test the automatic segmentation and labeling
    
    Args:
        image_path: Path to test image
    """
    from pathlib import Path
    import base64
    
    test_image = Path(image_path)
    
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        print("Please provide a valid image path")
        return
    
    with open(test_image, "rb") as f:
        image_bytes = f.read()
    
    segmenter = SAM3Segmenter()
    
    print("Running automatic segmentation and labeling...")
    result = segmenter.segment_and_label_everything.remote(image_bytes)
    
    print(f"\n✅ Segmentation complete!")
    print(f"Total segments: {result['total_segments']}")
    print(f"Image shape: {result['image_shape']}")
    
    # Show all segments
    print(f"\nDetected segments:")
    for seg in result['metadata']:
        print(f"  - {seg['label']}: confidence={seg['confidence']:.3f}, area={seg['area']}")
    
    # Save visualization image
    vis_image_data = base64.b64decode(result['visualization_image'])
    output_path = test_image.parent / f"{test_image.stem}_segmented.jpg"
    with open(output_path, "wb") as f:
        f.write(vis_image_data)
    
    print(f"\n📸 Visualization saved to: {output_path}")
    
    # Save metadata JSON
    import json
    metadata_path = test_image.parent / f"{test_image.stem}_metadata.json"
    with open(metadata_path, "w") as f:
        # Don't save full masks to keep file size reasonable
        metadata_light = [{k: v for k, v in seg.items() if k != 'mask'} for seg in result['metadata']]
        json.dump({
            "total_segments": result['total_segments'],
            "image_shape": result['image_shape'],
            "segments": metadata_light
        }, f, indent=2)
    
    print(f"📄 Metadata saved to: {metadata_path}")
    print(f"\n🎉 Done! Check the output files.")