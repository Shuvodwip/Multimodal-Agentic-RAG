"""Reading the contents of a figure with a vision model.

Deliberately separate from the rest of the pipeline and called only when a question
actually needs an image interpreted. Vision costs roughly 12x the text model per input
token ($0.60 vs $0.05 per 1M) and draws on a much smaller daily allowance, so routing
every query through it would be wasteful and would exhaust the quota quickly.

Set VISION_MODEL="" to disable; the figure tool then falls back to retrieval only.
"""

import base64
import mimetypes
import os
import sys

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

# Qwen accepts image input (verified) and is the only vision-capable model on this
# account. It is a reasoning model, so its replies carry a <think> block to strip.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b").strip()
VISION_ENABLED = bool(VISION_MODEL)

MAX_IMAGE_BYTES = 18 * 1024 * 1024  # Model accepts 20MB; leave room for base64 growth.

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


def strip_reasoning(text: str) -> str:
    """Drop the model's internal reasoning, keeping only the answer."""
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def describe_image(path: str, question: str, max_tokens: int = 2000) -> str:
    """Answer a question about one image. Raises on unreadable or oversized files."""
    if not VISION_ENABLED:
        raise RuntimeError("Vision is disabled (VISION_MODEL is unset).")

    size = os.path.getsize(path)
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image is {size / 1e6:.1f}MB; the model accepts at most 20MB.")

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode()

    response = _get_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{question}\n\n"
                            "Answer from what is visibly present in the image. If the "
                            "image does not show what was asked about, say so plainly "
                            "rather than guessing. Finish with a short, direct answer to "
                            "the question, stating any values you read from the image."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }
        ],
        max_tokens=max_tokens,
    )

    return strip_reasoning(response.choices[0].message.content or "")


if __name__ == "__main__":
    import glob

    candidates = sorted(glob.glob(".chroma/images/*")) or sorted(glob.glob("data/images/*"))
    if not candidates:
        print("No extracted figures found — run ingestion first.")
    else:
        target = candidates[0]
        print(f"Model: {VISION_MODEL}\nImage: {target}\n")
        print(describe_image(target, "What does this figure show?"))
