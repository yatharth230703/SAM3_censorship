"""
Agent 1: Censorship Category Selector

This agent analyzes an image and censorship requirements to determine
which SAM3 categories should be segmented.

Input: Image + Censorship prompt (what to censor)
Output: JSON list of categories to segment
"""
import base64
import os
import json
import sys
from pathlib import Path
from google import genai
from google.genai import types

# SAM3 Category Registry - all supported categories
SAM3_CATEGORIES = [
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
    "boat", "ship", "train", "license plate", "wheel", "tire", "windshield",
    # Animals
    "dog", "cat", "bird", "horse", "cow", "sheep", "pig", "fish", "elephant", "lion",
    "tiger", "bear", "deer", "rabbit", "snake", "turtle",
    # Electronics
    "tv", "monitor", "screen", "computer", "laptop", "keyboard", "phone", "smartphone",
    "tablet", "camera",
    # Documents & Text
    "document", "letter", "card", "id card", "credit card", "passport", "receipt",
    "newspaper", "sign", "label", "barcode", "qr code",
    # Weapons
    "gun", "pistol", "rifle", "knife", "sword", "weapon",
    # Other sensitive
    "tattoo", "logo", "brand", "text", "number", "address",
]


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


def select_categories(image_path: str, censorship_prompt: str) -> dict:
    """
    Use Gemini to analyze the image and select categories to censor.
    
    Args:
        image_path: Path to the image file
        censorship_prompt: Description of what should be censored
        
    Returns:
        {
            "categories": ["face", "license plate", ...],
            "reasoning": "Explanation of why these categories were selected"
        }
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    
    # Encode image
    image_data = encode_image(image_path)
    mime_type = get_image_mime_type(image_path)
    
    # Build the prompt
    system_prompt = f"""You are a censorship category selector for an image processing system.

Your task is to analyze an image and a censorship requirement, then select which object categories 
should be detected and potentially censored.

AVAILABLE CATEGORIES (you MUST only select from this list):
{json.dumps(SAM3_CATEGORIES, indent=2)}

CRITICAL RULES - MINIMIZE CATEGORIES:
1. ONLY select categories from the list above - do not invent new categories
2. Select ONLY the MOST SPECIFIC and DIRECTLY RELEVANT categories - be MINIMAL, not exhaustive
3. AVOID redundancy - if "face" covers the need, do NOT also add "head", "eye", "nose", "mouth"
4. PRIORITIZE precision over recall - it's better to select 2-3 highly relevant categories than 10+ loosely related ones
5. If the censorship requirement is specific (e.g., "blur license plates"), select ONLY that exact category
6. Only include related categories if they are EXPLICITLY mentioned or ABSOLUTELY necessary
7. Keep the list CONCISE and TO THE POINT - aim for the minimum number of categories that fulfill the requirement

EXAMPLES OF MINIMAL SELECTION:
- "Blur faces" → ["face"] (NOT face, head, eye, nose, mouth, ear, hair)
- "Hide license plates" → ["license plate"] (NOT license plate, car, vehicle, windshield)
- "Censor identity" → ["face", "id card", "license plate"] (only the core identity markers)
- "Remove brand logos" → ["logo"] (NOT logo, brand, text, label)

OUTPUT FORMAT:
You must respond with ONLY a valid JSON object in this exact format:
{{
    "categories": ["category1", "category2", ...],
    "reasoning": "Brief explanation of why you selected ONLY these minimal categories"
}}

Do not include any text before or after the JSON object."""

    user_prompt = f"""CENSORSHIP REQUIREMENT: {censorship_prompt}

Analyze the provided image and select the MINIMUM number of categories needed to fulfill the censorship requirement.

IMPORTANT: Be MINIMAL and PRECISE. Select only the most specific categories that directly match the requirement.
Avoid redundancy and over-selection. Quality over quantity.

Remember: Only select categories from the provided list. Output valid JSON only."""

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
        temperature=0.1,  # Low temperature for consistent outputs
    )
    
    print("Analyzing image and selecting categories...")
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=generate_config,
    )
    
    # Parse the response
    response_text = response.text.strip()
    
    # Try to extract JSON from the response
    try:
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        
        # Validate categories
        valid_categories = [c for c in result.get("categories", []) if c in SAM3_CATEGORIES]
        
        return {
            "categories": valid_categories,
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "original_response": result
        }
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        print(f"Raw response: {response_text}")
        
        # Fallback: try to extract categories mentioned in the text
        found_categories = [c for c in SAM3_CATEGORIES if c.lower() in response_text.lower()]
        return {
            "categories": found_categories,
            "reasoning": "Fallback extraction from non-JSON response",
            "raw_response": response_text
        }


def main():
    """
    CLI interface for the category selector agent.
    
    Usage:
        python agent1_category_selector.py <image_path> "<censorship_prompt>"
        
    Example:
        python agent1_category_selector.py photo.jpg "Blur all faces and license plates for privacy"
    """
    if len(sys.argv) < 3:
        print("Usage: python agent1_category_selector.py <image_path> \"<censorship_prompt>\"")
        print("\nExample:")
        print('  python agent1_category_selector.py photo.jpg "Blur all faces and license plates"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    censorship_prompt = sys.argv[2]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("=" * 60)
    print("Agent 1: Censorship Category Selector")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Censorship prompt: {censorship_prompt}")
    print("=" * 60)
    
    result = select_categories(image_path, censorship_prompt)
    
    print(f"\n✅ Selected {len(result['categories'])} categories:")
    for cat in result['categories']:
        print(f"  - {cat}")
    
    print(f"\n📝 Reasoning: {result['reasoning']}")
    
    # Save output
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_categories.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Saved to: {output_path}")
    
    # Print the categories as comma-separated for easy use with Modal
    print(f"\n🚀 For Modal CLI:")
    print(f"   modal run sam3_censorship.py --image-path {image_path} --categories \"{','.join(result['categories'])}\"")
    
    return result


if __name__ == "__main__":
    main()