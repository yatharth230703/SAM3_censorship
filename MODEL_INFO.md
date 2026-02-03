# SAM2 Model Information

## Model Files from Hugging Face

The script automatically downloads the SAM2 model from Hugging Face. Here's what you need to know:

### Model Repository
- **Hugging Face Repo**: `facebook/sam2-hiera-large`
- **URL**: https://huggingface.co/facebook/sam2-hiera-large

### Files Downloaded Automatically

When you deploy to Modal, the following files are downloaded automatically via `SAM2ImagePredictor.from_pretrained()`:

1. **sam2_hiera_large.pt** (~900 MB)
   - Main model checkpoint file
   - Contains the trained weights for the Hiera-Large backbone

2. **config.json**
   - Model configuration file
   - Defines architecture parameters

3. **preprocessor_config.json**
   - Image preprocessing configuration
   - Defines how images are normalized and resized

### Manual Download (Optional)

If you want to download the model manually for local testing, you can use:

```python
from huggingface_hub import hf_hub_download

# Download the checkpoint
checkpoint_path = hf_hub_download(
    repo_id="facebook/sam2-hiera-large",
    filename="sam2_hiera_large.pt",
    local_dir="./models"
)
```

Or using the command line:
```bash
huggingface-cli download facebook/sam2-hiera-large sam2_hiera_large.pt --local-dir ./models
```

## Model Variants

SAM2 comes in different sizes. You can change the model by updating `MODEL_TYPE` in `modal.py`:

| Model | Repo ID | Size | Performance |
|-------|---------|------|-------------|
| Tiny | `facebook/sam2-hiera-tiny` | ~38 MB | Fastest, lower accuracy |
| Small | `facebook/sam2-hiera-small` | ~184 MB | Fast, good accuracy |
| Base+ | `facebook/sam2-hiera-base-plus` | ~320 MB | Balanced |
| Large | `facebook/sam2-hiera-large` | ~900 MB | Best accuracy (default) |

## Caching in Modal

The Modal script uses a volume to cache the downloaded model:

```python
cache_vol = modal.Volume.from_name("sam2-hf-cache", create_if_missing=True)
```

This means:
- First deployment: Downloads ~900 MB (takes 2-3 minutes)
- Subsequent deployments: Uses cached model (starts in seconds)
- Cache persists across container restarts

## Model Architecture

SAM2 uses a Hierarchical Vision Transformer (Hiera) backbone:

- **Encoder**: Hiera-Large (144 embed dim, 2 heads, [2,6,36,4] stages)
- **Decoder**: Lightweight mask decoder
- **Prompt Encoder**: Handles points, boxes, and masks as prompts

## Model Capabilities

1. **Point Prompts**: Click on objects to segment them
2. **Box Prompts**: Draw bounding boxes around objects
3. **Mask Prompts**: Use previous masks to refine segmentation
4. **Automatic Segmentation**: Segment all objects without prompts
5. **Video Tracking**: Track objects across video frames (not implemented in this script)

## Memory Requirements

| GPU | VRAM | Batch Size | Notes |
|-----|------|------------|-------|
| T4 | 16 GB | 1 | May struggle with large images |
| A10G | 24 GB | 1-2 | Good for most use cases |
| A100 | 40/80 GB | 2-4 | Excellent performance |
| H200 | 141 GB | 4+ | Overkill for single images, great for batching |

## Performance Benchmarks

Approximate inference times on different GPUs (single 1024x1024 image):

- **H200**: ~0.5-1 second
- **A100**: ~1-1.5 seconds
- **A10G**: ~1.5-2 seconds
- **T4**: ~3-4 seconds

Cold start (first request) adds 15-30 seconds for model loading.

## References

- **Paper**: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
- **GitHub**: https://github.com/facebookresearch/sam2
- **Hugging Face**: https://huggingface.co/facebook/sam2-hiera-large
- **Demo**: https://sam2.metademolab.com/
