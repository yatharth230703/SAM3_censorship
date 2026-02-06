"""
Complete Censorship Pipeline on Modal
Combines Agent 1 (Gemini), SAM3, Agent 2 (Gemini), and censorship application
Single endpoint: POST /censor - Input: image + prompt, Output: censored image
"""
import modal
import os

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
        "google-genai>=0.2.0",
    )
)

app = modal.App("censorship-pipeline-complete", image=image)
cache_vol = modal.Volume.from_name("sam3-hf-cache", create_if_missing=True)
cache_dir = "/cache"

SAM3_CATEGORIES = [
    "person", "man", "woman", "child", "baby", "face", "head", "eye", "nose", "mouth",
    "ear", "hair", "hand", "finger", "arm", "leg", "foot", "body", "torso", "neck",
    "shirt", "t-shirt", "sweater", "hoodie", "jacket", "coat", "suit", "dress", "skirt",
    "pants", "jeans", "shorts", "underwear", "bra", "bikini", "swimsuit", "shoe", "sneaker",
    "boot", "sandal", "sock", "hat", "cap", "helmet", "glasses", "sunglasses", "watch",
    "jewelry", "necklace", "ring", "earring", "bracelet", "tie", "scarf", "glove", "belt",
    "bag", "purse", "handbag", "backpack", "wallet", "umbrella",
    "car", "truck", "bus", "van", "motorcycle", "bicycle", "scooter", "airplane", "helicopter",
    "boat", "ship", "train", "license plate", "wheel", "tire", "windshield",
    "dog", "cat", "bird", "horse", "cow", "sheep", "pig", "fish", "elephant", "lion",
    "tiger", "bear", "deer", "rabbit", "snake", "turtle",
    "tv", "monitor", "screen", "computer", "laptop", "keyboard", "phone", "smartphone",
    "tablet", "camera",
    "document", "letter", "card", "id card", "credit card", "passport", "receipt",
    "newspaper", "sign", "label", "barcode", "qr code",
    "gun", "pistol", "rifle", "knife", "sword", "weapon",
    "tattoo", "logo", "brand", "text", "number", "address",
]


@app.cls(
    image=image.env({"HF_HUB_CACHE": cache_dir}),
    volumes={cache_dir: cache_vol},
    gpu="H200",
    timeout=900,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("gemini-api-key"),
    ],
)
class CensorshipPipeline:
    @modal.enter()
    def load_models(self):
        import torch
        from transformers import Sam3Model, Sam3Processor
        from accelerate import Accelerator
        from google import genai
        
        print("Loading models...")
        self.device = Accelerator().device
        self.model = Sam3Model.from_pretrained(MODEL_TYPE, torch_dtype=torch.float16).to(self.device)
        self.processor = Sam3Processor.from_pretrained(MODEL_TYPE)
        self.model.eval()
        self.gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        print("Ready!")
    
    def _agent1_select_categories(self, image_bytes: bytes, censorship_prompt: str):
        import json
        from google.genai import types
        
        system_prompt = f"""Select object categories to segment for censorship.
AVAILABLE CATEGORIES: {json.dumps(SAM3_CATEGORIES)}
RULES: Select ONLY the MINIMUM categories needed. Be PRECISE. Output ONLY valid JSON:
{{"categories": ["cat1", "cat2"], "reasoning": "why"}}"""

        user_prompt = f"CENSORSHIP: {censorship_prompt}\nSelect minimum categories from the list. Output JSON only."

        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=user_prompt),
        ])]
        
        response = self.gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1)
        )
        
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(text)
        return [c for c in result.get("categories", []) if c in SAM3_CATEGORIES]
    
    def _sam3_segment(self, image_bytes: bytes, categories: list):
        import torch
        import numpy as np
        from PIL import Image, ImageDraw
        import io
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        target_size = [list(image.size[::-1])]
        all_segments = []
        
        for category in categories:
            try:
                inputs = self.processor(images=image, text=category, return_tensors="pt").to(self.device)
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
                results = self.processor.post_process_instance_segmentation(
                    outputs, threshold=0.3, mask_threshold=0.5, target_sizes=target_size
                )[0]
                
                for j in range(len(results.get("masks", []))):
                    mask = results["masks"][j]
                    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                    score = float(results["scores"][j].cpu().numpy() if hasattr(results["scores"][j], 'cpu') else results["scores"][j])
                    
                    if mask_np.sum() > 50 and score >= 0.8:
                        box = results["boxes"][j].cpu().numpy() if hasattr(results["boxes"][j], 'cpu') else np.array(results["boxes"][j])
                        bbox = [int(b) for b in box.tolist()]
                        all_segments.append({
                            "label": category, "confidence": score, "mask": mask_np,
                            "bbox": bbox, "area": int(mask_np.sum()),
                            "centroid": [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
                        })
            except:
                continue
        
        # Deduplicate
        all_segments.sort(key=lambda x: x["confidence"], reverse=True)
        unique = []
        for seg in all_segments:
            dup = False
            for ex in unique:
                if 0.8 < seg["area"]/max(ex["area"],1) < 1.25:
                    iou = np.logical_and(seg["mask"], ex["mask"]).sum() / np.logical_or(seg["mask"], ex["mask"]).sum()
                    dist = np.sqrt((seg["centroid"][0]-ex["centroid"][0])**2 + (seg["centroid"][1]-ex["centroid"][1])**2)
                    diag = np.sqrt((seg["bbox"][2]-seg["bbox"][0])**2 + (seg["bbox"][3]-seg["bbox"][1])**2)
                    if iou > 0.7 and dist/max(diag,1) < 0.15:
                        dup = True
                        break
            if not dup:
                unique.append(seg)
        
        # Visualization
        vis = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", vis.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        np.random.seed(42)
        colors = [tuple(np.random.randint(100,255,3).tolist()+[150]) for _ in range(len(unique))]
        
        for idx, seg in enumerate(unique):
            seg["id"] = idx
            mask_img = Image.fromarray((seg["mask"]*255).astype(np.uint8))
            colored = Image.new("RGBA", image.size, colors[idx])
            overlay.paste(colored, (0,0), mask_img)
            draw.rectangle(seg["bbox"], outline=colors[idx][:3]+(255,), width=3)
            draw.text((seg["bbox"][0], max(0,seg["bbox"][1]-18)), f"[{idx}] {seg['label']}", fill=(255,255,255,255))
        
        vis = Image.alpha_composite(vis, overlay).convert("RGB")
        return unique, vis, image
    
    def _agent2_select_segments(self, vis_image_bytes: bytes, segments: list, censorship_prompt: str):
        import json
        from google.genai import types
        
        seg_info = [{"id": s["id"], "label": s["label"], "confidence": s["confidence"], 
                     "bbox": s["bbox"], "area": s["area"]} for s in segments]
        
        system_prompt = """Decide which segments to censor based on the requirement.
Output ONLY valid JSON: {"segments_to_censor": [0,2,5], "reasoning": "why"}"""

        user_prompt = f"CENSORSHIP: {censorship_prompt}\nSEGMENTS: {json.dumps(seg_info)}\nSelect segment IDs to censor. JSON only."

        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=vis_image_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=user_prompt),
        ])]
        
        response = self.gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1)
        )
        
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(text)
        valid_ids = {s["id"] for s in segments}
        return [sid for sid in result.get("segments_to_censor", []) if sid in valid_ids]
    
    def _apply_censorship(self, image, segments: list, ids: list, method: str):
        import numpy as np
        from PIL import ImageFilter, Image
        
        if not ids:
            return image
        
        result = image
        for seg in segments:
            if seg["id"] not in ids:
                continue
            mask = seg["mask"]
            
            if method == "blur":
                blurred = result.filter(ImageFilter.GaussianBlur(30))
                arr = np.array(result)
                blur_arr = np.array(blurred)
                mask_3d = np.stack([mask]*3, axis=-1)
                result = Image.fromarray(np.where(mask_3d, blur_arr, arr).astype(np.uint8))
            elif method == "pixelate":
                arr = np.array(result)
                res = arr.copy()
                rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0,-1]]
                    cmin, cmax = np.where(cols)[0][[0,-1]]
                    for y in range(rmin, rmax+1, 15):
                        for x in range(cmin, cmax+1, 15):
                            if mask[y:min(y+15,rmax+1), x:min(x+15,cmax+1)].any():
                                avg = arr[y:min(y+15,rmax+1), x:min(x+15,cmax+1)].mean(axis=(0,1)).astype(np.uint8)
                                for by in range(y, min(y+15,rmax+1)):
                                    for bx in range(x, min(x+15,cmax+1)):
                                        if mask[by,bx]:
                                            res[by,bx] = avg
                result = Image.fromarray(res)
            else:  # blackbox
                arr = np.array(result)
                mask_3d = np.stack([mask]*3, axis=-1)
                result = Image.fromarray(np.where(mask_3d, 0, arr).astype(np.uint8))
        
        return result
    
    @modal.method()
    def censor_image(self, image_bytes: bytes, censorship_prompt: str, method: str = "blur"):
        import time
        import base64
        import io
        
        start = time.time()
        
        # Agent 1
        categories = self._agent1_select_categories(image_bytes, censorship_prompt)
        if not categories:
            return {"error": "No categories selected", "censored_image": base64.b64encode(image_bytes).decode()}
        
        # SAM3
        segments, vis_image, orig_image = self._sam3_segment(image_bytes, categories)
        if not segments:
            return {"error": "No segments found", "censored_image": base64.b64encode(image_bytes).decode()}
        
        # Agent 2
        vis_buf = io.BytesIO()
        vis_image.save(vis_buf, format="JPEG", quality=95)
        ids = self._agent2_select_segments(vis_buf.getvalue(), segments, censorship_prompt)
        
        # Apply
        censored = self._apply_censorship(orig_image, segments, ids, method)
        
        # Encode
        cens_buf = io.BytesIO()
        censored.save(cens_buf, format="JPEG", quality=95)
        
        return {
            "censored_image": base64.b64encode(cens_buf.getvalue()).decode(),
            "visualization_image": base64.b64encode(vis_buf.getvalue()).decode(),
            "categories_selected": categories,
            "segments_found": len(segments),
            "segments_censored": len(ids),
            "processing_time_seconds": round(time.time()-start, 2)
        }


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import base64
    
    web_app = FastAPI(title="Censorship Pipeline")
    
    class CensorRequest(BaseModel):
        image: str
        prompt: str
        method: str = "blur"
    
    @web_app.post("/censor")
    async def censor(req: CensorRequest):
        try:
            pipeline = CensorshipPipeline()
            return pipeline.censor_image.remote(base64.b64decode(req.image), req.prompt, req.method)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    return web_app


@app.local_entrypoint()
def test(image_path: str = "img.png", prompt: str = "Blur all faces", method: str = "blur"):
    from pathlib import Path
    import base64
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    pipeline = CensorshipPipeline()
    result = pipeline.censor_image.remote(image_bytes, prompt, method)
    
    print(f"Categories: {result['categories_selected']}")
    print(f"Segments found: {result['segments_found']}, censored: {result['segments_censored']}")
    print(f"Time: {result['processing_time_seconds']}s")
    
    # Save
    stem = Path(image_path).stem
    with open(f"{stem}_censored.jpg", "wb") as f:
        f.write(base64.b64decode(result['censored_image']))
    with open(f"{stem}_visualization.jpg", "wb") as f:
        f.write(base64.b64decode(result['visualization_image']))
    
    print(f"Saved: {stem}_censored.jpg, {stem}_visualization.jpg")
