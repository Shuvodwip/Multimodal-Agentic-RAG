#!/usr/bin/env bash
# Assemble a Hugging Face Space bundle in build/space/.
#
#   bash scripts/prepare_space.sh
#
# Copies only what the Space needs to run: the source, the Space README (which carries
# the required frontmatter), and Space-specific requirements. Deliberately excludes
# data/, .chroma/, .env, tests, and the development requirements freeze — the Space
# starts with nothing indexed and secrets are configured in its own settings.

set -euo pipefail

OUT="build/space"
rm -rf "$OUT"
mkdir -p "$OUT"

cp -r src "$OUT/src"
cp deploy/huggingface/README.md "$OUT/README.md"
cp deploy/huggingface/requirements.txt "$OUT/requirements.txt"

# Drop the evaluation and ad-hoc test scripts. The serving path never imports them,
# and they depend on ragas, which the Space deliberately does not install — shipping
# modules that reference absent packages is just confusing.
rm -f "$OUT"/src/eval_*.py "$OUT"/src/run_eval.py "$OUT"/src/ragas_compat.py \
      "$OUT"/src/build_eval_set.py "$OUT"/src/test_*.py "$OUT"/src/check_overlap.py \
      "$OUT"/src/compare_retrieval.py "$OUT"/src/metrics.py "$OUT"/src/graph_basics.py

# Strip caches that add weight and nothing else.
find "$OUT" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true

# Guard against shipping anything private: these must never reach a public Space.
for forbidden in .env data .chroma; do
  if [ -e "$OUT/$forbidden" ]; then
    echo "ERROR: $forbidden ended up in the bundle — aborting." >&2
    exit 1
  fi
done

echo "Bundle ready in $OUT/"
echo
echo "Contents:"
find "$OUT" -maxdepth 2 -not -path '*/\.*' | sed "s|^$OUT|  .|"
echo
echo "Size: $(du -sh "$OUT" | cut -f1)"
