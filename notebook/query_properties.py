import os
import base64
import json
from urllib import response
from openai import OpenAI

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

IMAGE_DIR    = "../data/gsv_out"   # update to your image folder on the server
RESULTS_FILE = "../result/result/new-prompt-result.json" 
MODEL        = "/u/capstone/hf_cache/Qwen3.5-35B-A3B"

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

SYSTEM_PROMPT = """
You are an expert in urban property assessment. You will be given a Google Street View image.

Focus on the house or building directly in the CENTER of the image at street level.
If the image shows a row of attached units, judge only the central unit, not its neighbors.

An image is UNSUITABLE if any of the following are true:
- The camera is looking down a street or alley with no single central property facing it
- There is no clearly visible house or building centered and facing the camera
- The central subject is fully obstructed by trees, vehicles, or other objects
- The image shows a parking lot, vacant lot, or open space in the center
- The image is too blurry or dark to assess the property
- The property is a large commercial or industrial building, not residential

If the image is unsuitable, respond ONLY with this JSON:
{
  "suitable": false,
  "reason": "Brief explanation of why the image cannot be assessed."
}

If the image is suitable, record what you observe and make a holistic judgment.
Respond ONLY with this JSON:
{
  "suitable": true,
  "attributes": {
    "broken_or_missing_windows": 0,
    "boarded_doors_or_windows": 0,
    "severe_structural_damage": 0,
    "overgrown_vegetation": 0,
    "graffiti_or_stains_on_walls": 0,
    "lights_on": 0
  },
  "final_decision": "abandoned",
  "confidence": "high",
  "reasoning": "Your reasoning here."
}

Output format:
- Each attribute must be 0 (not present) or 1 (present). Record what you actually see — these are observations, not a scoring rubric.
- final_decision must be either "abandoned" or "not_abandoned".
- confidence must be one of: "low", "medium", or "high".
- Do not include any text outside the JSON object.

How to decide:
- Use holistic visual judgment, the way an experienced inspector would. Do NOT count attributes or apply any fixed threshold (e.g. "one attribute means not abandoned"). Weigh the severity and clarity of what you see.
- The attributes above are a checklist of common signals, but the decision is yours — consider the overall condition of the property, not just whether a box is checked.

Context worth knowing (awareness, not rules):
- The following are common around BOTH occupied and abandoned homes and should NOT by themselves be treated as evidence of occupancy: trash or recycling bins on the sidewalk, parked cars, porch furniture, flags, potted plants, stickers or paper notices on doors or windows.
- Stronger evidence of occupancy: interior lights on, people present, curtains/blinds that look recently used, a clearly mowed or tended lawn, active utility equipment.
- Stronger evidence of abandonment: boarded openings, collapsed or sagging structure, visibly missing windows, heavy overgrowth reaching or covering the entrance, long-term decay that looks unmaintained.
- Surface wear alone (faded paint, weathered brick, minor stains) is not, on its own, a strong abandonment signal — many occupied older homes look like this.
"""

# ─────────────────────────────────────────────
# Helper: encode image to base64
# ─────────────────────────────────────────────

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ─────────────────────────────────────────────
# Helper: query model with a single image
# ─────────────────────────────────────────────

def query_image(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                  ".png": "image/png",  ".webp": "image/webp"}
    mime_type = mime_types.get(ext, "image/jpeg")

    image_data = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this property image and return the JSON assessment."
                        },
                    ],
                },
            ],
            max_tokens=30000,
            extra_body={"enable_thinking": True},
        )

        raw = response.choices[0].message.content

        if raw is None:
            return {"error": "Model returned empty content", "raw": str(response)}

        raw = raw.strip()

    except Exception as e:
        return {"error": str(e), "raw": ""}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(raw[start:end])
        return {"error": "Failed to parse response", "raw": raw}

# ─────────────────────────────────────────────
# Main: process one image per run
# ─────────────────────────────────────────────

def main():
    # Load existing results
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = []

    already_done = {r["filename"] for r in results}

    # Find all images not yet processed
    supported  = (".jpg", ".jpeg", ".png", ".webp")
    all_images = sorted([f for f in os.listdir(IMAGE_DIR)
                         if f.lower().endswith(supported)])
    remaining  = [f for f in all_images if f not in already_done]

    if not remaining:
        print(f"All {len(all_images)} images have been processed.")
        return

    for i, filename in enumerate(remaining):
        image_path = os.path.join(IMAGE_DIR, filename)
        current    = len(already_done) + i + 1

        print(f"Processing image {current} of {len(all_images)}: {filename}")
        print(f"{len(remaining) - i - 1} remaining after this one.\n")

        result = query_image(image_path)
        result["filename"] = filename

        if "error" not in result:
            if not result.get("suitable", True):
                print(f"  UNSUITABLE")
                print(f"  Reason: {result.get('reason')}")
            else:
                attrs = result.get("attributes", {})
                print(f"Decision:   {result.get('final_decision')}")
                print(f"Confidence: {result.get('confidence')}")
                print(f"Reasoning:  {result.get('reasoning')}")
                print(f"Attributes:")
                for attr, val in attrs.items():
                    print(f"  {attr}: {val}")
        else:
            print(f"ERROR: {result['error']}")
            if "raw" in result:
                print(f"Raw response: {result['raw']}")

        # Save after every image so progress is never lost
        results.append(result)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved to {RESULTS_FILE}\n")

    print(f"All {len(all_images)} images processed.")


if __name__ == "__main__":
    main()
