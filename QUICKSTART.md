# Quick Start Guide - SAM3

Get your SAM3 segmentation endpoint running in 5 minutes!

## What's NEW in SAM3?

**Text-Based Segmentation!** Just describe what you want to segment:
- "person" - finds all people
- "laptop" - finds all laptops  
- "ear" - finds all ears
- Works with 270K+ unique concepts!

## Step 1: Install Modal

```bash
pip install modal
```

## Step 2: Authenticate with Modal

```bash
modal setup
```

This will open a browser window to authenticate. Follow the prompts.

## Step 3: Deploy to Modal

```bash
modal deploy modal.py
```

Expected output:
```
✓ Created objects.
├── 🔨 Created mount /Users/you/project
├── 🔨 Created function segment_endpoint.
└── 🔨 Created class SAM3Segmenter.

✓ App deployed! 🎉

View Deployment: https://modal.com/apps/your-username/sam3-segmentation

Web endpoint: https://your-username--sam3-segmentation-segment-endpoint.modal.run
```

**Copy the web endpoint URL** - you'll need it for testing!

## Step 4: Test Your Endpoint

### Option A: Test with Text Prompts (NEW!)

```python
import requests
import base64

# Your endpoint URL from step 3
ENDPOINT_URL = "https://your-username--sam3-segmentation-segment-endpoint.modal.run"

# Read and encode your image
with open("your_image.jpg", "rb") as f:
    image_bytes = f.read()
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# Segment using text prompt
response = requests.post(ENDPOINT_URL, json={
    "image": image_base64,
    "mode": "text",
    "text": "person"  # Try: "laptop", "ear", "dog", etc.
})

result = response.json()
print(f"Found {result['count']} objects")
print(f"Scores: {result['scores']}")
```

### Option B: Test with Point Prompts

```python
# Send request with point click
response = requests.post(ENDPOINT_URL, json={
    "image": image_base64,
    "mode": "point",
    "point_coords": [[100, 100]],  # Click at (100, 100)
    "point_labels": [1]  # Foreground point
})

result = response.json()
print(f"Segmentation score: {result['score']}")
```

### Option C: Test Locally with Modal

```bash
# Test text-based segmentation
modal run modal.py::test --image-path your_image.jpg --mode text --text "person"

# Test point-based segmentation
modal run modal.py::test --image-path your_image.jpg --mode point --x 100 --y 100

# Test automatic segmentation
modal run modal.py::test --image-path your_image.jpg --mode auto
```

## Step 5: Integrate into Your Web App

### React/Next.js Example with Text Prompts

```javascript
async function segmentWithText(imageFile, textPrompt) {
  // Convert image to base64
  const reader = new FileReader();
  const base64 = await new Promise((resolve) => {
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.readAsDataURL(imageFile);
  });

  // Call your endpoint
  const response = await fetch('YOUR_ENDPOINT_URL', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: base64,
      mode: 'text',
      text: textPrompt  // e.g., "person", "laptop", "ear"
    })
  });

  const result = await response.json();
  console.log(`Found ${result.count} ${textPrompt}(s)`);
  return result;
}

// Usage in component
function ImageSegmenter() {
  const [textPrompt, setTextPrompt] = useState("person");
  
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    const result = await segmentWithText(file, textPrompt);
    console.log(`Found ${result.count} objects`);
  };
  
  return (
    <div>
      <input 
        type="text" 
        value={textPrompt}
        onChange={(e) => setTextPrompt(e.target.value)}
        placeholder="What to segment?"
      />
      <input type="file" onChange={handleImageUpload} />
    </div>
  );
}
```

## Common Use Cases

### 1. Text-Based Segmentation (NEW in SAM3!)

```json
{
  "image": "base64_string",
  "mode": "text",
  "text": "person"
}
```

### 2. Point-Based Segmentation (Click to Segment)

```json
{
  "image": "base64_string",
  "mode": "point",
  "point_coords": [[x, y]],
  "point_labels": [1]
}
```

### 3. Box-Based Segmentation (Draw Box)

```json
{
  "image": "base64_string",
  "mode": "box",
  "box_coords": [x_min, y_min, x_max, y_max]
}
```

### 4. Automatic Segmentation (Segment Everything)

```json
{
  "image": "base64_string",
  "mode": "auto",
  "points_per_batch": 64
}
```

### 5. Multi-Point Refinement

```json
{
  "image": "base64_string",
  "mode": "point",
  "point_coords": [[x1, y1], [x2, y2], [x3, y3]],
  "point_labels": [1, 1, 0]  // 1=foreground, 0=background
}
```

## Text Prompt Examples

SAM3 understands 270K+ concepts! Try these:

**People & Body Parts:**
- "person", "face", "hand", "ear", "eye", "nose"

**Objects:**
- "laptop", "phone", "cup", "bottle", "book", "pen"

**Animals:**
- "dog", "cat", "bird", "fish", "horse"

**Vehicles:**
- "car", "truck", "bicycle", "motorcycle", "bus"

**Furniture:**
- "chair", "table", "sofa", "bed", "desk"

**Food:**
- "apple", "banana", "pizza", "sandwich", "cake"

## Troubleshooting

### "Container failed to start"
- Check Modal dashboard for logs
- Verify your Modal account has GPU access
- Try a smaller GPU (change `gpu="H200"` to `gpu="A10G"` in modal.py)

### "Out of memory"
- Reduce image size before sending
- Use a larger GPU
- For auto mode, reduce `points_per_batch`

### "Slow first request"
- This is normal! Cold start takes 20-40 seconds (loads both models)
- Subsequent requests are fast (~1-2 seconds)
- Consider keeping container warm with periodic pings

### "Model download fails"
- Check internet connectivity
- Verify Hugging Face is accessible
- Try deploying again (downloads are cached)

### "Text mode returns 0 objects"
- Try different text prompts
- Lower the `threshold` parameter (default: 0.5)
- Some concepts may not be in the training data

## Next Steps

1. ✅ Deploy endpoint
2. ✅ Test with text prompts
3. 🔲 Build UI for image upload
4. 🔲 Add text input for prompts
5. 🔲 Add visualization for masks
6. 🔲 Implement authentication
7. 🔲 Add error handling
8. 🔲 Monitor usage and costs

## Cost Estimation

With H200 GPU at ~$4.76/hour:
- Per request: ~$0.002-0.003 (1-2 seconds)
- 100 requests: ~$0.20-0.30
- 1000 requests: ~$2.00-3.00

Your free Modal credits should cover thousands of requests!

## Support

- Modal Docs: https://modal.com/docs
- SAM3 Hugging Face: https://huggingface.co/facebook/sam3
- Modal Discord: https://discord.gg/modal

Happy segmenting with SAM3! 🎉
