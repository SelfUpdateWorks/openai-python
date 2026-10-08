import argparse
import base64
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Optional: load OPENAI_API_KEY from a local .env file (never commit .env!)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    APIStatusError,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # vision-capable, low-cost
IMAGES_DIR = Path("product_images")
OUTPUT_FILE = Path("generated_listings.json")
MAX_RETRIES = 3            # retry attempts for transient errors
BASE_DELAY = 2             # seconds; doubled after each failed attempt
DELAY_BETWEEN_CALLS = 1    # small pause between products (rate limits)
SUPPORTED_FORMATS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}

# Approximate prices (USD per 1M tokens) for cost tracking (bonus challenge 3).
# Check https://openai.com/api/pricing for current values.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


# ---------------------------------------------------------------------------
# Step 1: API client setup
# ---------------------------------------------------------------------------
def get_client() -> OpenAI:
    """Create the OpenAI client. The key comes ONLY from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY='sk-...'"
        )
    # timeout: vision calls are slower than text-only calls
    client = OpenAI(api_key=api_key, timeout=60)
    print(f"✓ OpenAI client initialised (model: {MODEL}, "
          f"key: {api_key[:5]}...{api_key[-4:]})")
    return client


# ---------------------------------------------------------------------------
# Step 2: Dataset preparation
# ---------------------------------------------------------------------------
# The HuggingFace dataset has no prices, so we assign a realistic price
# range per article type (seeded random → reproducible results).
PRICE_RANGES = {
    "Tshirts": (15, 35), "Shirts": (25, 60), "Jeans": (40, 90),
    "Watches": (60, 250), "Casual Shoes": (45, 120), "Sports Shoes": (50, 140),
    "Handbags": (35, 150), "Sunglasses": (20, 120), "Kurtas": (20, 60),
    "Tops": (15, 45), "Heels": (40, 110), "Belts": (15, 45),
    "Wallets": (15, 60), "Socks": (5, 15), "Backpacks": (30, 90),
}


def assign_price(article_type: str, rng: random.Random) -> float:
    low, high = PRICE_RANGES.get(article_type, (20, 80))
    return round(round(rng.uniform(low, high)) - 0.01, 2)  # e.g. 34.99


def load_products(n: int = 5) -> pd.DataFrame:
    """Load n products from HuggingFace; fall back to local images."""
    IMAGES_DIR.mkdir(exist_ok=True)
    rng = random.Random(42)
    print("Loading product dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("ashraq/fashion-product-images-small",
                               split=f"train[:{n}]")
        print(f"✓ Loaded {len(dataset)} products from HuggingFace")

        products = []
        for row in dataset:
            # Save the PIL image to disk so every product has an image_path
            image_path = IMAGES_DIR / f"product_{row['id']}.jpg"
            if not image_path.exists():
                row["image"].convert("RGB").save(image_path, "JPEG")
            products.append({
                "id": row["id"],
                "name": row["productDisplayName"],
                "price": assign_price(row["articleType"], rng),
                "category": f"{row['masterCategory']} > {row['subCategory']} > {row['articleType']}",
                "additional_info": (f"Colour: {row['baseColour']}; Gender: {row['gender']}; "
                                    f"Season: {row['season']}; Usage: {row['usage']}"),
                "image_path": str(image_path),
            })
        products_df = pd.DataFrame(products)

    except Exception as e:
        print(f"⚠ Could not load HuggingFace dataset: {e}")
        print("Using local images instead (put .jpg/.png files in product_images/)...")
        local_images = sorted(p for p in IMAGES_DIR.iterdir()
                              if p.suffix.lower() in SUPPORTED_FORMATS)[:n]
        if not local_images:
            raise FileNotFoundError("No images found in product_images/.")
        products_df = pd.DataFrame([{
            "id": i + 1,
            "name": p.stem.replace("_", " ").title(),
            "price": assign_price("", rng),
            "category": "General",
            "additional_info": None,
            "image_path": str(p),
        } for i, p in enumerate(local_images)])

    print(f"✓ Dataset prepared! Total products: {len(products_df)}\n")
    return products_df


# ---------------------------------------------------------------------------
# Step 3: Encode images to base64
# ---------------------------------------------------------------------------
def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """Return (base64_string, mime_subtype). Raises clear errors if invalid."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format '{ext}'. "
                         f"Use one of: {', '.join(SUPPORTED_FORMATS)}")
    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty (corrupted?): {image_path}")
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded, SUPPORTED_FORMATS[ext]


# ---------------------------------------------------------------------------
# Step 4: Prompt template
# ---------------------------------------------------------------------------
def create_product_listing_prompt(product_name, price, category, additional_info=None):
    """Build the instruction text that accompanies the image."""
    extra = f"- Additional Info: {additional_info}\n" if additional_info else ""
    return f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{extra}
Please create a professional product listing that includes:

1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
   - Highlight key features and benefits
   - Use persuasive language
   - Include relevant details visible in the image
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 relevant keywords)

Format your response as JSON with the following structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2", ...],
    "keywords": "keyword1, keyword2, ..."
}}

Be specific about what you see in the image. Mention colors, materials, design elements, and any distinctive features.
Only describe what is actually visible or given in the product information; do not invent technical specifications.
Do not claim a fabric, material or composition (e.g. cotton, leather, breathable) unless it is given in the product information."""


# ---------------------------------------------------------------------------
# Step 5: Call the vision API + parse the response
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> dict:
    """Parse model output as JSON; strip ```json fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        cleaned = cleaned.rsplit("```", 1)[0]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def call_vision_api(client: OpenAI, prompt: str, image_b64: str, mime: str,
                    temperature: float = 0.7) -> tuple[dict, dict]:
    """
    Send prompt + image, return (listing_dict, usage_dict).
    Transient errors (rate limit, timeout, connection, 5xx) are retried with
    exponential backoff; permanent errors (bad key, bad request) are raised.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/{mime};base64,{image_b64}",
                           "detail": "low"}},  # "low" = cheaper & faster
        ],
    }]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=800,
                response_format={"type": "json_object"},  # forces valid JSON
            )
            raw = response.choices[0].message.content
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            try:
                return parse_json_response(raw), usage
            except json.JSONDecodeError:
                # Fallback: keep the raw text so nothing is lost
                print("    ⚠ JSON parsing failed, returning raw text")
                return {"raw_response": raw, "parse_error": True}, usage

        except (AuthenticationError, BadRequestError):
            raise  # retrying will not fix an invalid key or a bad request
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            reason = type(e).__name__
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            reason = f"Server error {e.status_code}"

        if attempt == MAX_RETRIES:
            raise RuntimeError(f"Failed after {MAX_RETRIES} attempts ({reason})")
        wait = BASE_DELAY * 2 ** (attempt - 1)  # 2s, 4s, 8s ...
        print(f"    ⚠ {reason} – retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})")
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Bonus: cost tracking + quality check
# ---------------------------------------------------------------------------
def estimate_cost(usage: dict) -> float:
    prices = PRICING.get(MODEL)
    if not prices:
        return 0.0
    return (usage["prompt_tokens"] * prices["input"]
            + usage["completion_tokens"] * prices["output"]) / 1_000_000


def quality_check(listing: dict) -> list[str]:
    """Return a list of issues; empty list = listing meets the brief."""
    issues = []
    for key in ("title", "description", "features", "keywords"):
        if not listing.get(key):
            issues.append(f"missing '{key}'")
    if len(listing.get("title", "")) > 60:
        issues.append(f"title too long ({len(listing['title'])} chars)")
    words = len(listing.get("description", "").split())
    if listing.get("description") and not 130 <= words <= 220:
        issues.append(f"description length {words} words (target 150-200)")
    n_feat = len(listing.get("features", []))
    if listing.get("features") and not 5 <= n_feat <= 7:
        issues.append(f"{n_feat} features (target 5-7)")
    n_kw = len([k for k in listing.get("keywords", "").split(",") if k.strip()])
    if listing.get("keywords") and not 10 <= n_kw <= 15:
        issues.append(f"{n_kw} keywords (target 10-15)")
    return issues


# ---------------------------------------------------------------------------
# Step 6: Generate one listing / process the whole batch
# ---------------------------------------------------------------------------
def generate_listing(client: OpenAI, product: dict) -> dict:
    """Full pipeline for one product. Never raises: errors are recorded."""
    result = {
        "product_id": product["id"],
        "input": {k: product[k] for k in ("name", "price", "category", "image_path")},
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        image_b64, mime = encode_image_to_base64(product["image_path"])
        prompt = create_product_listing_prompt(
            product["name"], product["price"], product["category"],
            product.get("additional_info"))
        listing, usage = call_vision_api(client, prompt, image_b64, mime)
        result.update(
            status="success",
            listing=listing,
            usage=usage,
            estimated_cost_usd=round(estimate_cost(usage), 6),
            quality_issues=quality_check(listing),
        )
    except AuthenticationError:
        raise  # fatal for the whole batch – stop immediately
    except Exception as e:
        result.update(status="error", error_type=type(e).__name__, error=str(e))
    return result


def process_products(client: OpenAI, products_df: pd.DataFrame) -> list[dict]:
    results = []
    total = len(products_df)
    for i, product in enumerate(products_df.to_dict("records"), start=1):
        print(f"[{i}/{total}] {product['name']}")
        result = generate_listing(client, product)
        results.append(result)

        if result["status"] == "success":
            title = result["listing"].get("title", "(no title)")
            print(f"    ✓ {title}")
            print(f"    tokens: {result['usage']['total_tokens']}  "
                  f"cost: ${result['estimated_cost_usd']:.5f}")
            if result["quality_issues"]:
                print(f"    ⚑ review: {'; '.join(result['quality_issues'])}")
        else:
            print(f"    ✗ {result['error_type']}: {result['error']}")

        save_results(results)  # save after every product → nothing lost on crash
        if i < total:
            time.sleep(DELAY_BETWEEN_CALLS)
    return results


def save_results(results: list[dict], path: Path = OUTPUT_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def print_summary(results: list[dict]) -> None:
    ok = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    tokens = sum(r["usage"]["total_tokens"] for r in ok)
    cost = sum(r["estimated_cost_usd"] for r in ok)
    flagged = [r for r in ok if r["quality_issues"]]
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successful listings : {len(ok)}/{len(results)}")
    print(f"Failed              : {len(failed)}")
    print(f"Flagged for review  : {len(flagged)}")
    print(f"Total tokens        : {tokens}")
    print(f"Estimated cost      : ${cost:.5f}")
    if ok:
        print(f"Cost per 1,000 listings ≈ ${cost / len(ok) * 1000:.2f}")
    print(f"Results saved to    : {OUTPUT_FILE.resolve()}")


# ---------------------------------------------------------------------------
# Step 7: Error-handling demonstration (for the screenshots deliverable)
# ---------------------------------------------------------------------------
def demo_error_handling(client: OpenAI) -> None:
    print("=" * 50)
    print("ERROR HANDLING DEMO")
    print("=" * 50)

    print("\n1) Missing image file")
    r = generate_listing(client, {"id": "demo-1", "name": "Ghost Product", "price": 10,
                                  "category": "Test", "image_path": "does_not_exist.jpg"})
    print(f"   → handled: {r['error_type']}: {r['error']}")

    print("\n2) Unsupported image format")
    bad = Path("not_an_image.txt")
    bad.write_text("hello")
    r = generate_listing(client, {"id": "demo-2", "name": "Text File", "price": 10,
                                  "category": "Test", "image_path": str(bad)})
    print(f"   → handled: {r['error_type']}: {r['error']}")
    bad.unlink()

    print("\n3) Model returns markdown-wrapped JSON")
    sample = '```json\n{"title": "Test", "features": []}\n```'
    print(f"   input : {sample!r}")
    print(f"   parsed: {parse_json_response(sample)}")

    print("\n4) Invalid API key")
    try:
        OpenAI(api_key="sk-invalid-key-for-demo").chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=5)
    except AuthenticationError as e:
        print(f"   → handled: AuthenticationError (status {e.status_code})")

    print("\n5) Missing API key")
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        get_client()
    except EnvironmentError as e:
        print(f"   → handled: {e}")
    finally:
        if saved:
            os.environ["OPENAI_API_KEY"] = saved
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Automated product listing generator")
    parser.add_argument("--num", type=int, default=5, help="number of products")
    parser.add_argument("--demo-errors", action="store_true",
                        help="run the error-handling demo instead of the batch")
    args = parser.parse_args()

    try:
        client = get_client()
    except EnvironmentError as e:
        sys.exit(f"✗ {e}")

    if args.demo_errors:
        demo_error_handling(client)
        return

    products_df = load_products(args.num)
    print(products_df[["id", "name", "price"]].to_string(index=False), "\n")

    try:
        results = process_products(client, products_df)
    except AuthenticationError:
        sys.exit("✗ Invalid API key – check OPENAI_API_KEY and your account credits.")
    except KeyboardInterrupt:
        sys.exit("\nStopped by user. Results so far are saved.")

    print_summary(results)


if __name__ == "__main__":
    main()