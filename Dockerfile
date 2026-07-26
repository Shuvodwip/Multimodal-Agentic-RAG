# Multi-stage build: compile wheels in the builder, ship only the installed packages.
# The final image carries no build toolchain and no pip cache.

FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install CPU-only torch first. The default wheel pulls ~2GB of CUDA libraries that a
# CPU container can never use, so pinning the CPU index is the single biggest size win.
RUN pip install --prefix=/install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --prefix=/install -r requirements-api.txt


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface

# curl is needed for the container healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

COPY --from=builder /install /usr/local

WORKDIR /app
COPY --chown=app:app src/ ./src/

USER app

# Bake the embedding model into the image so the first request doesn't pay a download.
# Only the text encoder is fetched — CLIP is used for image ingestion, not for serving.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# The vector store is mounted at runtime (see docker-compose), not baked in — it is
# derived from the source document, which stays out of the image.
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
