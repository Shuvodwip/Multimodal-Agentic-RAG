# Deploying to Hugging Face Spaces

The Space runs the Streamlit UI with the agent in-process — one container, no separate
API service. Free tier gives 16GB RAM and 2 vCPU, which is ample; the constraint is
model download time on first boot, not memory.

## 1. Build the bundle

```bash
bash scripts/prepare_space.sh
```

This assembles `build/space/` with the source, a Space `README.md` carrying the required
frontmatter, and Space-specific requirements. It deliberately leaves out `data/`,
`.chroma/`, `.env`, tests, and the development requirements freeze, and aborts if any of
those slip in.

Why separate requirements: the repository's `requirements.txt` is a full development
freeze of ~130 packages including the evaluation stack, and it would resolve the **CUDA**
build of torch — gigabytes of GPU libraries a CPU Space cannot use, likely exceeding the
build timeout. The Space file pins CPU-only torch instead.

## 2. Create the Space

At <https://huggingface.co/new-space>:

- **SDK**: Streamlit
- **Hardware**: CPU basic (free)
- **Visibility**: your choice

## 3. Add secrets

In the Space's *Settings → Variables and secrets*, add:

| Secret | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | The app fails to start without it — the chat model is constructed at import |
| `LANGFUSE_PUBLIC_KEY` | No | Enables tracing |
| `LANGFUSE_SECRET_KEY` | No | Enables tracing |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com`, or the US host |
| `VISION_MODEL` | No | Defaults to `qwen/qwen3.6-27b`; set empty to disable figure reading |

Tracing degrades gracefully: without the Langfuse keys the app runs untraced rather
than failing.

## 4. Push

```bash
cd build/space
git init
git remote add origin https://huggingface.co/spaces/<username>/<space-name>
git add .
git commit -m "Deploy agentic multimodal RAG"
git push -u origin main
```

Hugging Face asks for a token as the password — create one at
<https://huggingface.co/settings/tokens> with **write** access.

## What to expect

- **First build takes several minutes** — torch and the transformer libraries are large.
- **The first question is slow.** The embedding, reranking, and vision models download
  and load on demand, not at build time.
- **Storage is ephemeral.** Uploaded documents and the vector store are wiped when the
  Space restarts or sleeps. The app starts with nothing indexed, which is the intended
  behaviour here — visitors upload their own files.
- **Free-tier rate limits apply.** The chat model allows 6,000 tokens per minute, so a
  burst of large requests can return a rate-limit error rather than degrading.

## Updating

Re-run `prepare_space.sh` and push again from `build/space/`. Because the bundle is
regenerated from source each time, the Space never drifts from the repository.
