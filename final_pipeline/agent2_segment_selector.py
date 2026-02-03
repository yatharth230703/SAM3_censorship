"""
Agent 2: Censorship Segment Selector

This agent analyzes SAM3 segmentation results and determines which specific
segments should actually be censored.

Input: Original image + SAM3 segmentation results + Censorship prompt
Output: List of segment IDs to censor
"""
import base64
import os
import json
import sys
from pathlib import Path
from google import genai
from google.genai import types


def encode_image(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from file extension"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def select_segments_to_censor(
    visualization_image_path: str,
    segments_metadata: list,
    censorship_prompt: str
) -> dict:
    """
    Use Gemini to analyze segmentation results and select which to censor.
    
    Args:
        visualization_image_path: Path to the SAM3 visualization image (with labeled segments)
        segments_metadata: List of segment dictionaries from SAM3
        censorship_prompt: Original censorship requirement
        
    Returns:
        {
            "segments_to_censor": [0, 2, 5],  # Segment IDs
            "reasoning": "Explanation",
            "segment_details": [{"id": 0, "label": "face", "reason": "..."}, ...]
        }
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Encode visualization image
    image_data = encode_image(visualization_image_path)
    mime_type = get_image_mime_type(visualization_image_path)
    
    # Prepare segment info for the prompt
    segments_info = []
    for seg in segments_metadata:
        segments_info.append({
            "id": seg["id"],
            "label": seg["label"],
            "confidence": seg["confidence"],
            "bbox": seg["bbox"],
            "area": seg["area"],
            "center": seg["center"]
        })
    
    system_prompt = """You are a censorship decision agent for an image processing system.

You are provided with:
1. An image showing detected segments with ID labels (format: [ID] label)
2. Metadata about each segment
3. The censorship requirement

Your task is to decide which specific segments should be censored.

RULES:
1. Carefully look at the visualization image - segments are highlighted and labeled with [ID]
2. Only select segments that match the censorship requirement
3. Be precise - don't censor segments that shouldn't be censored
4. Consider context (e.g., if censoring "strangers' faces", don't censor the main subject if they're clearly the focus)
5. If a segment doesn't clearly match the requirement, don't include it

OUTPUT FORMAT:
You must respond with ONLY a valid JSON object:
{
    "segments_to_censor": [0, 2, 5],
    "reasoning": "Overall explanation",
    "segment_details": [
        {"id": 0, "label": "face", "reason": "This is a bystander's face"},
        {"id": 2, "label": "license plate", "reason": "Vehicle plate visible"}
    ]
}

Only include segments that SHOULD be censored. Do not include any text before or after the JSON."""

    user_prompt = f"""CENSORSHIP REQUIREMENT: {censorship_prompt}

DETECTED SEGMENTS:
{json.dumps(segments_info, indent=2)}

Look at the visualization image where each segment is labeled with [ID] and its category.
Decide which segments should be censored based on the requirement.

Output valid JSON only."""

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    data=base64.standard_b64decode(image_data),
                    mime_type=mime_type
                ),
                types.Part.from_text(text=user_prompt),
            ],
        ),
    ]
    
    generate_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,
    )
    
    print("Analyzing segments and selecting items to censor...")
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=generate_config,
    )
    
    response_text = response.text.strip()
    
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Validate segment IDs
        valid_ids = {seg["id"] for seg in segments_metadata}
        censored_ids = [sid for sid in result.get("segments_to_censor", []) if sid in valid_ids]
        
        return {
            "segments_to_censor": censored_ids,
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "segment_details": result.get("segment_details", [])
        }
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return {
            "segments_to_censor": [],
            "reasoning": "Failed to parse response",
            "raw_response": response_text
        }


def main():
    """
    CLI interface for the segment selector agent.
    
    Usage:
        python agent2_segment_selector.py <visualization_image> <segments_json> "<censorship_prompt>"
    """
    if len(sys.argv) < 4:
        print("Usage: python agent2_segment_selector.py <visualization_image> <segments_json> \"<censorship_prompt>\"")
        print("\nExample:")
        print('  python agent2_segment_selector.py photo_censorship_preview.jpg photo_segments.json "Blur faces of bystanders"')
        sys.exit(1)
    
    vis_image_path = sys.argv[1]
    segments_json_path = sys.argv[2]
    censorship_prompt = sys.argv[3]
    
    if not Path(vis_image_path).exists():
        print(f"Error: Visualization image not found: {vis_image_path}")
        sys.exit(1)
    
    if not Path(segments_json_path).exists():
        print(f"Error: Segments JSON not found: {segments_json_path}")
        sys.exit(1)
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Load segments
    with open(segments_json_path, "r") as f:
        segments_data = json.load(f)
    
    segments = segments_data.get("segments", segments_data)
    if isinstance(segments, dict):
        segments = segments.get("segments", [])
    
    print("=" * 60)
    print("Agent 2: Censorship Segment Selector")
    print("=" * 60)
    print(f"Visualization: {vis_image_path}")
    print(f"Segments: {len(segments)} detected")
    print(f"Censorship prompt: {censorship_prompt}")
    print("=" * 60)
    
    result = select_segments_to_censor(vis_image_path, segments, censorship_prompt)
    
    print(f"\n✅ Selected {len(result['segments_to_censor'])} segments to censor:")
    for detail in result.get('segment_details', []):
        print(f"  [{detail['id']}] {detail.get('label', 'unknown')}: {detail.get('reason', '')}")
    
    print(f"\n📝 Reasoning: {result['reasoning']}")
    
    # Save output
    output_path = Path(vis_image_path).parent / f"{Path(vis_image_path).stem.replace('_censorship_preview', '')}_censor_decision.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved to: {output_path}")
    
    # Print segment IDs for the final censorship script
    if result['segments_to_censor']:
        print(f"\n🚀 Segment IDs to censor: {result['segments_to_censor']}")
    
    return result


if __name__ == "__main__":
    main()