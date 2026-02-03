# SAM2 Image Segmentation on Modal

Deploy Segment Anything Model 2 (SAM2) as a serverless GPU endpoint on Modal.

## Prerequisites

1. Install Modal:
```bash
pip install modal
```

2. Set up Modal authentication:
```bash
modal setup
```

## Deployment Steps

### 1. Deploy to Modal

```bash
modal deploy modal.py
```

This will:
- Create a Modal app called "sam2-segmentation"
- Download SAM2 model weights from Hugging Face (cached automatically)
- Deploy the endpoint with H200 GPU support

### 2. Get Your Endpoint URL

After deployment, Modal will output your endpoint URL:
```
https://your-username--sam2-segmentation-segment-endpoint.modal.run
```

Copy this URL for use in your web application.

## Testing

### Test locally with Modal
```bash
modal run modal.py::test
```

### Test with Python client
```bash
python client_example.py
```

Update the `ENDPOINT_URL` in `client_example.py` with your actual endpoint URL.

## API Usage

### Endpoint: POST to your Modal URL

### Request Format (JSON)

```json
{
  "image": "base64_encoded_image_string",
  "mode": "point",  // "point", "box", or "auto"
  "point_coords": [[100, 150], [200, 250]],  // optional, for point mode
  "point_labels": [1, 0],  // optional, 1=foreground, 0=background
  "box_coords": [x_min, y_min, x_max, y_max],  // optional, for box mode
  "points_per_batch": 64  // optional, for auto mode (default: 64)
}
```

### Response Format

**Point-based segmentation:**
```json
{
  "mask": [[true, false, ...], ...],  // 2D boolean array
  "score": 0.95,
  "all_scores": [0.95, 0.87, 0.82],
  "image_shape": [height, width]
}
```

**Automatic segmentation:**
```json
{
  "masks": [
    {
      "segmentation": [[true, false, ...], ...],
      "area": 12345,
      "bbox": [x, y, width, height],
      "predicted_iou": 0.92,
      "stability_score": 0.88
    },
    ...
  ],
  "count": 5,
  "image_shape": [height, width]
}
```

## Integration Examples

### JavaScript/TypeScript (React, Next.js, etc.)

See `client_example.js` for a complete example with React component.

```javascript
const result = await segmentImage(imageFile, {
  endpointUrl: 'YOUR_MODAL_ENDPOINT_URL',
  pointCoords: [[100, 100]],
  pointLabels: [1],
  mode: 'point'
});
```

### Python

See `client_example.py` for a complete example.

```python
result = segment_image(
    image_path="image.jpg",
    endpoint_url="YOUR_MODAL_ENDPOINT_URL",
    point_coords=[[100, 100]],
    point_labels=[1],
    mode="point"
)
```

## Cost Optimization

- **GPU**: Using H200 (~$4.76/hour) - highest performance for demanding workloads
- **Cold starts**: First request may take 15-30 seconds to load model
- **Warm containers**: Subsequent requests are fast (~1-2 seconds)
- **HF Hub caching**: Model weights are cached in Modal volume, no re-download needed

### Tips to reduce costs:
1. Use `timeout` parameter to control max execution time
2. Consider using smaller GPU (A10G ~$1.10/hour or A100 ~$3.00/hour) if H200 is overkill
3. Batch multiple images in a single request if possible
4. Use `points_per_batch` parameter in auto mode to control memory usage

## Monitoring

View logs and metrics in Modal dashboard:
```bash
modal app logs sam2-segmentation
```

## Troubleshooting

### Model download fails
- Check internet connectivity in Modal container
- Verify Hugging Face is accessible
- Try alternative model sources

### Out of memory errors
- Reduce image size before sending
- Use smaller model variant
- Upgrade to larger GPU (A100)

### Slow cold starts
- Model loading takes time on first request
- Consider keeping container warm with periodic pings
- Use Modal's container keep-alive features

## Next Steps

1. Deploy the endpoint
2. Update endpoint URL in your web app
3. Implement image upload UI
4. Add visualization for segmentation masks
5. Consider adding authentication for production use
