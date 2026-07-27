# Multi-stage build: compile wheels in the builder, ship only the installed packages.
# The final image carries no build toolchain and no pip cache.

FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install into a venv rather than --prefix. With --prefix, pip cannot tell that a
# package is already present, so the later requirements install silently pulled the
# CUDA build of torch on top of the CPU one — 2.7GB of unusable nvidia libraries.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch: the default wheel bundles CUDA libraries a CPU container can never
# use. Installing it first means sentence-transformers finds its dependency satisfied.
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt \
    && python -c "import torch; assert not torch.version.cuda, 'CUDA torch leaked into the image'"


FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PATH="/opt/venv/bin:$PATH"

# curl is needed for the container healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app src/ ./src/

# Create the vector-store directory owned by the runtime user *before* anything mounts
# over it. Docker initialises an empty named volume from the image path it covers,
# ownership included — without this the volume arrives root-owned and ChromaDB, running
# as `app`, cannot create its SQLite file.
RUN mkdir -p /app/.chroma && chown app:app /app/.chroma

USER app

# Bake the models into the image so the first request doesn't pay a download, and so
# the container works without network access to Hugging Face. CLIP is skipped — it is
# used for image ingestion, not for serving.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# The vector store is mounted at runtime (see docker-compose), not baked in — it is
# derived from the source document, which stays out of the image.
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
