import os
import base64
import json
from urllib import response
from openai import OpenAI

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

IMAGE_DIR    = "../gsv_out"   # update to your image folder on the server
RESULTS_FILE = os.path.join(IMAGE_DIR, "results.json")
MODEL        = "/u/capstone/hf_cache/Qwen3.5-35B-A3B"

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

SYSTEM_PROMPT = """
You are an expert in urban property assessment. You will be given a Google Street View image.

Focus ONLY on the house or building directly in the CENTER of the image at street level. 
The target property should be the dominant subject facing the camera.

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

If the image is suitable, analyze the central residential property and respond ONLY with this JSON:
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
  "reasoning": "Your reasoning here. Explain which attributes you observed and how they influenced your decision."
}

Rules:
- Each attribute must be 0 (not present) or 1 (present). Be thorough and inspect the central property carefully.
- final_decision must be either "abandoned" or "not_abandoned".
- confidence must be one of: "low", "medium", or "high".
- Use "high" confidence when the visible evidence is clear and strongly supports the decision.
- Use "medium" confidence when the evidence is somewhat clear but limited, mixed, or partially ambiguous.
- Use "low" confidence when the evidence is weak, unclear, partially obstructed, or the decision is uncertain.
- The attributes are the PRIMARY factor in your decision. Do not override them based on general appearance alone.
- Do not include any text outside the JSON object.
- You MUST NOT output any reasoning or thinking before the JSON.

Interpret the attributes as follows:
- broken_or_missing_windows = 1 is evidence toward abandoned
- boarded_doors_or_windows = 1 is strong evidence toward abandoned, even if no other negative attributes are present
- severe_structural_damage = 1 is very strong evidence toward abandoned, even if no other negative attributes are present
- overgrown_vegetation = 1 is evidence toward abandoned
- graffiti_or_stains_on_walls = 1 is evidence toward abandoned, but weaker than boarded openings or structural damage
- lights_on = 1 is evidence toward not_abandoned, because visible active lighting suggests occupancy or recent use

Decision guidance:
- Do NOT use a simple rule such as "0 or 1 attributes means not abandoned."
- Some single attributes can be sufficient by themselves to support "abandoned," especially boarded_doors_or_windows or severe_structural_damage.
- Multiple negative attributes strongly support "abandoned."
- If lights_on = 1, treat that as counter-evidence against abandonment. It should reduce the likelihood of "abandoned" unless there is clear stronger evidence such as boarded openings, severe structural damage, or multiple other negative attributes.
- If both positive and negative signals are present, weigh the strength of the evidence rather than just counting attributes.
- Give more weight to boarded_doors_or_windows and severe_structural_damage than to graffiti/stains alone.
- Overgrown vegetation can support "abandoned," especially when combined with any other negative sign.
- Broken or missing windows are meaningful evidence toward "abandoned," especially when clearly visible.
- Do not assume a property is not abandoned just because it appears intact overall if one strong abandonment indicator is clearly present.

Reasoning guidance:
- In the reasoning field, briefly state which attributes were observed, which were not observed, and why the strongest signals led to the final decision.
- Keep the reasoning concise and focused on the listed attributes.
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
                            "text": "Analyze this property image and return the JSON assessment. Keep your thinking concise and output the JSON immediately after."
                        },
                    ],
                },
            ],
            max_tokens=30000,
            extra_body={"enable_thinking": False},
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
