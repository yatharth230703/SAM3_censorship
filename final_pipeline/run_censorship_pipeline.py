"""
Complete Censorship Pipeline Runner

This script orchestrates the entire censorship workflow:
1. Agent 1: Select categories based on image + prompt
2. SAM3: Segment the selected categories
3. Agent 2: Decide which segments to censor
4. Apply: Generate the final censored image

Usage:
    python run_censorship_pipeline.py <image_path> "<censorship_prompt>" [--method blur|pixelate|blackbox]
    
Example:
    python run_censorship_pipeline.py photo.jpg "Blur all faces and license plates for privacy" --method blur
"""
import os
import sys
import json
import time
import base64
import subprocess
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and print output"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_censorship_pipeline.py <image_path> \"<censorship_prompt>\" [--method blur|pixelate|blackbox]")
        print("\nExample:")
        print('  python run_censorship_pipeline.py photo.jpg "Blur all faces and license plates" --method blur')
        sys.exit(1)
    
    image_path = sys.argv[1]
    censorship_prompt = sys.argv[2]
    
    # Parse method
    method = "blur"
    if "--method" in sys.argv:
        idx = sys.argv.index("--method")
        if idx + 1 < len(sys.argv):
            method = sys.argv[idx + 1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Check environment
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    stem = Path(image_path).stem
    parent = Path(image_path).parent
    
    print("=" * 60)
    print("🎬 CENSORSHIP PIPELINE")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Prompt: {censorship_prompt}")
    print(f"Method: {method}")
    print("=" * 60)
    
    total_start = time.time()
    
    # ========================================
    # Step 1: Agent 1 - Select Categories
    # ========================================
    step1_start = time.time()
    print("\n" + "="*60)
    print("📋 STEP 1: Category Selection (Agent 1)")
    print("="*60)
    
    # Import and run agent 1
    from agent1_category_selector import select_categories
    categories_result = select_categories(image_path, censorship_prompt)
    
    categories = categories_result["categories"]
    print(f"Selected categories: {categories}")
    
    step1_time = time.time() - step1_start
    print(f"\n⏱️  STEP 1 TIME: {step1_time:.2f}s")
    print("="*60)
    
    if not categories:
        print("❌ No categories selected. Exiting.")
        sys.exit(1)
    
    # ========================================
    # Step 2: SAM3 Segmentation
    # ========================================
    step2_start = time.time()
    print("\n" + "="*60)
    print("🔍 STEP 2: SAM3 Segmentation")
    print("="*60)
    
    categories_str = ",".join(categories)
    
    # Run Modal
    cmd = [
        "modal", "run", "sam3_censorship.py",
        "--image-path", image_path,
        "--categories", categories_str
    ]
    
    success = run_command(cmd, "Running SAM3 on Modal...")
    
    if not success:
        print("❌ SAM3 segmentation failed")
        sys.exit(1)
    
    step2_time = time.time() - step2_start
    print(f"\n⏱️  STEP 2 TIME: {step2_time:.2f}s")
    print("="*60)
    
    # Expected output files
    vis_image = parent / f"{stem}_censorship_preview.jpg"
    segments_json = parent / f"{stem}_segments.json"  # Lightweight, for Agent 2
    segments_full_json = parent / f"{stem}_segments_full.json"  # With masks, for censorship
    
    if not vis_image.exists() or not segments_json.exists() or not segments_full_json.exists():
        print(f"❌ Expected output files not found")
        print(f"   Looking for: {vis_image}, {segments_json}, {segments_full_json}")
        sys.exit(1)
    
    # ========================================
    # Step 3: Agent 2 - Select Segments
    # ========================================
    step3_start = time.time()
    print("\n" + "="*60)
    print("🎯 STEP 3: Segment Selection (Agent 2)")
    print("="*60)
    
    from agent2_segment_selector import select_segments_to_censor
    
    # Agent 2 reads the lightweight segments.json (without masks)
    with open(segments_json, "r") as f:
        segments_data = json.load(f)
    
    segments = segments_data.get("segments", [])
    
    censor_result = select_segments_to_censor(
        str(vis_image),
        segments,
        censorship_prompt
    )
    
    print(f"Segments to censor: {censor_result['segments_to_censor']}")
    
    # Save censor decision
    censor_decision_path = parent / f"{stem}_censor_decision.json"
    with open(censor_decision_path, "w") as f:
        json.dump(censor_result, f, indent=2)
    
    step3_time = time.time() - step3_start
    print(f"\n⏱️  STEP 3 TIME: {step3_time:.2f}s")
    print("="*60)
    
    if not censor_result['segments_to_censor']:
        print("✅ No segments need censoring!")
        sys.exit(0)
    
    # ========================================
    # Step 4: Apply Censorship
    # ========================================
    step4_start = time.time()
    print("\n" + "="*60)
    print("🖌 STEP 4: Applying Censorship")
    print("="*60)
    
    from apply_censorship import apply_censorship
    
    # Use segments_full.json which contains masks for censorship application
    output_path = apply_censorship(
        image_path,
        str(segments_full_json),  # Full segments with masks
        str(censor_decision_path),
        method=method
    )
    
    step4_time = time.time() - step4_start
    print(f"\n⏱️  STEP 4 TIME: {step4_time:.2f}s")
    print("="*60)
    
    # ========================================
    # Summary
    # ========================================
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE")
    print("="*60)
    print(f"\n⏱️  TIMING BREAKDOWN:")
    print("="*60)
    print(f"  Step 1 (Category Selection):  {step1_time:>6.2f}s  ({step1_time/total_time*100:>5.1f}%)")
    print(f"  Step 2 (SAM3 Segmentation):   {step2_time:>6.2f}s  ({step2_time/total_time*100:>5.1f}%)")
    print(f"  Step 3 (Segment Selection):   {step3_time:>6.2f}s  ({step3_time/total_time*100:>5.1f}%)")
    print(f"  Step 4 (Apply Censorship):    {step4_time:>6.2f}s  ({step4_time/total_time*100:>5.1f}%)")
    print("  " + "─"*56)
    print(f"  TOTAL PIPELINE TIME:          {total_time:>6.2f}s  (100.0%)")
    print("="*60)
    
    print(f"\n📁 Output Files:")
    print(f"   Categories:    {stem}_categories.json")
    print(f"   Segments:      {stem}_segments.json (metadata)")
    print(f"   Segments Full: {stem}_segments_full.json (with masks)")
    print(f"   Preview:       {stem}_censorship_preview.jpg")
    print(f"   Decision:      {stem}_censor_decision.json")
    print(f"   Final:         {stem}_censored.jpg")
    
    print(f"\n✅ Censored image: {output_path}")


if __name__ == "__main__":
    main()


    
