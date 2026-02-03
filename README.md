# SAM3 Image Segmentation on Modal 🎯

Deploy Facebook's Segment Anything Model 3 (SAM3) as a serverless GPU endpoint on Modal. Perfect for integrating powerful image segmentation into your fullstack web applications!

## What's NEW in SAM3? 🆕

**Text-Based Concept Segmentation!** SAM3 introduces the ability to segment objects using natural language descriptions:
- Segment "person", "laptop", "ear" - just describe what you want!
- Finds ALL instances of the concept in the image
- 75-80% of human performance on 270K+ unique concepts
- Still supports point/box prompts like SAM2

## Features

✨ **Text Prompts** - Segment using natural language (NEW!)  
🎯 **Point/Box Prompts** - Traditional click-to-segment (like SAM2)  
🤖 **Automatic Segmentation** - Segment everything in the image  
🚀 **H200 GPU** - Maximum performance with your free Modal credits  
🌐 **REST API** - Easy integration with any web framework  
📦 **Cached Models** - Fast cold starts after first deployment  
💰 **Cost Efficient** - Pay only for what you use

## Quick Start

```bash
# 1. Install Modal
pip install modal

# 2. Authenticate
modal setup

# 3. Deploy
modal deploy modal.py

# 4. Get your endpoint URL and start using it!
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## Project Structure

```
.
├── modal.py                 # Main Modal deployment script
├── client_example.py        # Python client example
├── client_example.js        # JavaScript/React client example
├── visualize_masks.py       # Visualization helper script
├── QUICKSTART.md           # Step-by-step deployment guide
├── DEPLOYMENT.md           # Detailed deployment documentation
├── MODEL_INFO.md           # Model architecture and variants
└── requirements.txt        # Python dependencies
```

## Usage Examples

### Text-Based Segmentation (NEW in SAM3!)

Segment objects using natural language:

```python
import requests
import base64

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post(ENDPOINT_URL, json={
    "image": image_b64,
    "mode": "text",
    "text": "person"  # Finds all people in the image!
})

result = response.json()
print(f"Found {result['count']} people")
print(f"Scores: {result['scores']}")
```

### Point-Based Segmentation

Click on an object to segment it:

```python
response = requests.post(ENDPOINT_URL, json={
    "image": image_b64,
    "mode": "point",
    "point_coords": [[100, 150]],  # Click coordinates
    "point_labels": [1]  # 1 = foreground
})

result = response.json()
print(f"Score: {result['score']}")
```

### Automatic Segmentation

Segment all objects in the image:

```python
response = requests.post(ENDPOINT_URL, json={
    "image": image_b64,
    "mode": "auto"
})

result = response.json()
print(f"Found {result['count']} objects")
```

### Box-Based Segmentation

Draw a bounding box around an object:

```python
response = requests.post(ENDPOINT_URL, json={
    "image": image_b64,
    "mode": "box",
    "box_coords": [50, 50, 200, 200]  # [x_min, y_min, x_max, y_max]
})
```

### JavaScript/React Integration

```javascript
async function segmentWithText(imageFile, textPrompt) {
  const base64 = await fileToBase64(imageFile);
  
  const response = await fetch(ENDPOINT_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: base64,
      mode: 'text',
      text: textPrompt  // e.g., "laptop", "person", "ear"
    })
  });
  
  return await response.json();
}
```

See [client_example.js](client_example.js) for complete React component.

## API Reference

### Endpoint

`POST https://your-username--sam3-segmentation-segment-endpoint.modal.run`

### Request Body

```typescript
{
  image: string;              // Base64 encoded image
  mode: "text" | "point" | "box" | "auto";  // default: "text"
  
  // For text mode (NEW!):
  text?: string;              // e.g., "person", "laptop", "ear"
  threshold?: number;         // Confidence threshold (default: 0.5)
  mask_threshold?: number;    // Mask binarization threshold (default: 0.5)
  
  // For point mode:
  point_coords?: number[][];  // [[x1, y1], [x2, y2], ...]
  point_labels?: number[];    // [1, 0, ...] (1=fg, 0=bg)
  
  // For box mode:
  box_coords?: number[];      // [x_min, y_min, x_max, y_max]
  
  // For auto mode:
  points_per_batch?: number;  // Default: 64
}
```

### Response

**Text Mode (NEW!):**
```typescript
{
  masks: boolean[][][];       // List of 2D boolean arrays
  boxes: number[][];          // [[x1, y1, x2, y2], ...]
  scores: number[];           // Confidence scores
  count: number;              // Number of objects found
  image_shape: [number, number];
}
```

**Point/Box Mode:**
```typescript
{
  mask: boolean[][];          // 2D boolean array
  score: number;              // Confidence score
  all_scores: number[];       // All mask scores
  image_shape: [number, number];
}
```

**Auto Mode:**
```typescript
{
  masks: Array<{
    segmentation: boolean[][];
    score: number;
  }>;
  count: number;
  image_shape: [number, number];
}
```

## Model Information

- **Model**: SAM3 (facebook/sam3)
- **Size**: ~2.4 GB (includes both text and tracker models)
- **Hugging Face**: `facebook/sam3`
- **Paper**: SAM 3: Segment Anything in Images and Videos

SAM3 includes THREE models:
1. **Sam3Model** - Text-based concept segmentation (NEW!)
2. **Sam3TrackerModel** - Point/box-based segmentation
3. **Sam3VideoModel** - Video segmentation (not included in this deployment)

See [MODEL_INFO.md](MODEL_INFO.md) for details.

## Performance

| GPU | VRAM | Inference Time | Cost/Hour |
|-----|------|----------------|-----------|
| H200 | 141 GB | ~0.5-1s | ~$4.76 |
| A100 | 80 GB | ~1-1.5s | ~$3.00 |
| A10G | 24 GB | ~1.5-2s | ~$1.10 |

Cold start: 20-40 seconds (first request only, loads both models)

## Cost Estimation

With H200 GPU:
- Per request: ~$0.002-0.003
- 100 requests: ~$0.20-0.30
- 1000 requests: ~$2.00-3.00

Your free Modal credits should cover thousands of requests!

## Web Application Integration

### React Example with Text Prompts

```jsx
function ImageSegmenter() {
  const [textPrompt, setTextPrompt] = useState("person");
  const [results, setResults] = useState(null);
  
  const handleSegment = async (imageFile) => {
    const result = await segmentWithText(imageFile, textPrompt);
    setResults(result);
    console.log(`Found ${result.count} ${textPrompt}(s)`);
  };
  
  return (
    <div>
      <input 
        type="text" 
        value={textPrompt}
        onChange={(e) => setTextPrompt(e.target.value)}
        placeholder="What to segment?"
      />
      <input type="file" onChange={(e) => handleSegment(e.target.files[0])} />
      {results && <p>Found {results.count} objects</p>}
    </div>
  );
}
```

### Next.js API Route

```typescript
// app/api/segment/route.ts
export async function POST(request: Request) {
  const { image, text } = await request.json();
  
  const response = await fetch(MODAL_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image, mode: 'text', text })
  });
  
  return Response.json(await response.json());
}
```

## Troubleshooting

### Container fails to start
- Check Modal dashboard for logs
- Try smaller GPU: change `gpu="H200"` to `gpu="A10G"`
- Verify Modal account has GPU access

### Out of memory
- Reduce image size before sending
- Use larger GPU
- Lower `points_per_batch` in auto mode

### Slow first request
- Normal! Cold start takes 20-40s (loads both models)
- Subsequent requests are fast
- Consider container keep-alive

See [DEPLOYMENT.md](DEPLOYMENT.md) for more troubleshooting tips.

## Development

### Local Testing

```bash
# Test text-based segmentation
modal run modal.py::test --image-path test.jpg --mode text --text "person"

# Test point-based segmentation
modal run modal.py::test --image-path test.jpg --mode point --x 100 --y 100

# Test automatic segmentation
modal run modal.py::test --image-path test.jpg --mode auto
```

### Monitoring

```bash
# View logs
modal app logs sam3-segmentation

# View dashboard
modal app show sam3-segmentation
```

## Files Downloaded from Hugging Face

**You don't need to manually download anything!** The script automatically downloads:

1. **Sam3Model** (~1.2 GB) - For text-based segmentation
2. **Sam3TrackerModel** (~1.2 GB) - For point/box segmentation
3. Configuration files

These are cached in a Modal volume after first deployment.

## GPU Recommendations

- **H200**: Best performance, great for high-volume production
- **A100**: Excellent balance of performance and cost
- **A10G**: Good for development and moderate usage

Change GPU in `modal.py`:
```python
gpu="H200"  # or "A100", "A10G"
```

## Resources

- 📚 [Modal Documentation](https://modal.com/docs)
- 🤗 [SAM3 on Hugging Face](https://huggingface.co/facebook/sam3)
- 💬 [Modal Discord](https://discord.gg/modal)

## License

This project uses SAM3 which is licensed under Apache 2.0.

---

Built with ❤️ using [Modal](https://modal.com) and [SAM3](https://huggingface.co/facebook/sam3)
