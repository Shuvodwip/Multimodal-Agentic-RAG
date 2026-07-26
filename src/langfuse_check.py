"""Verify Langfuse credentials and send one test trace.

Run after adding LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST to .env:

    python src/langfuse_check.py
"""

import os
import sys

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

REQUIRED = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]


def main() -> int:
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        print(f"Missing in .env: {', '.join(missing)}")
        print("Get keys at https://cloud.langfuse.com -> Settings -> API Keys")
        return 1

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    print(f"Host: {host}")
    print(f"Public key: {os.getenv('LANGFUSE_PUBLIC_KEY')[:12]}...")

    client = get_client()

    if not client.auth_check():
        print("FAILED: credentials rejected. Check the keys and that LANGFUSE_HOST")
        print("matches your project's region (EU vs US).")
        return 1

    print("Auth OK.")

    with client.start_as_current_span(name="connection-test") as span:
        span.update(
            input={"note": "verifying Langfuse setup"},
            output={"status": "ok"},
        )
        span.update_trace(tags=["setup-check"])

    client.flush()
    print("Test trace sent. Check the Traces view in your Langfuse project.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
