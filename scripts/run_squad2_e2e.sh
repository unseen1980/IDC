#!/usr/bin/env bash
set -euo pipefail

# End-to-end SQuAD2.0 (dev) runner: download → convert → run per doc → merge → evaluate

# Config
SQUAD_URL_DEV="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
DATA_DIR="data/squad"
INPUT_DIR="data/input"
OUT_ROOT="out/squad"
DIM="${DIM:-1536}"
EMBEDDER="${EMBEDDER:-gemini-embedding-001}"
LIMIT="${LIMIT:-50}"
TARGET_AVG="${TARGET_AVG:-8}"
TOL="${TOL:-1}"
VARIANTS="${VARIANTS:-paragraphs idc fixed sliding coh}"
AUTO_TUNE="${AUTO_TUNE:-1}"
AUTO_TUNE_BASELINES="${AUTO_TUNE_BASELINES:-1}"

red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
blue()   { printf "\033[34m%s\033[0m\n" "$*"; }

need() {
  command -v "$1" >/dev/null 2>&1 || { red "Missing command: $1"; exit 1; }
}

need python
mkdir -p "$DATA_DIR" "$INPUT_DIR" "$OUT_ROOT"

# 1) Download SQuAD v2.0 dev
DEV_JSON="${DATA_DIR}/dev-v2.0.json"
if [[ ! -f "$DEV_JSON" ]]; then
  blue "Downloading SQuAD2.0 dev → $DEV_JSON"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$SQUAD_URL_DEV" -o "$DEV_JSON"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEV_JSON" "$SQUAD_URL_DEV"
  else
    red "Neither curl nor wget found. Please download $SQUAD_URL_DEV to $DEV_JSON manually."
    exit 1
  fi
else
  yellow "SQuAD dev exists → $DEV_JSON"
fi

# 2) Convert N articles
DOC_IDS_FILE="${DATA_DIR}/doc_ids.txt"
blue "Converting first ${LIMIT} articles to IDC inputs"
python src/convert_squad.py --squad "$DEV_JSON" --limit "$LIMIT" --doc-ids-out "$DOC_IDS_FILE"

if [[ ! -s "$DOC_IDS_FILE" ]]; then
  red "No doc ids found at $DOC_IDS_FILE"
  exit 1
fi

# 3) Run pipeline per doc (keep gold spans)
blue "Running pipeline per doc (AUTO_TUNE=${AUTO_TUNE}, AUTO_TUNE_BASELINES=${AUTO_TUNE_BASELINES}, keep gold spans)"
while IFS= read -r doc; do
  [[ -z "$doc" ]] && continue
  echo "--- ${doc} ---"
  # Use CLI; default --force-spans is off → preserves gold spans
  FLAGS=( --doc "$doc" --target-avg "$TARGET_AVG" --tol "$TOL" )
  if [[ "$AUTO_TUNE" != "1" ]]; then
    [[ -n "${LAMBDA:-}" ]] && FLAGS+=( --lam "$LAMBDA" )
    [[ -n "${BOUNDARY_PENALTY:-}" ]] && FLAGS+=( --boundary-penalty "$BOUNDARY_PENALTY" )
    [[ -n "${MAX_LEN:-}" ]] && FLAGS+=( --max-len "$MAX_LEN" )
    [[ -n "${MIN_LEN:-}" ]] && FLAGS+=( --min-len "$MIN_LEN" )
    [[ -n "${COHERENCE_WEIGHT:-}" ]] && FLAGS+=( --coherence-weight "$COHERENCE_WEIGHT" )
  fi
  if [[ "$AUTO_TUNE" == "1" ]]; then FLAGS+=( --auto-tune ); fi
  if [[ "$AUTO_TUNE_BASELINES" == "1" ]]; then FLAGS+=( --auto-tune-baselines ); fi
  RESPECT_EXISTING_INTENTS=1 python src/cli.py run "${FLAGS[@]}" || { red "Pipeline failed for $doc"; exit 1; }
done < "$DOC_IDS_FILE"

# 4) Merge + evaluate for each variant
for variant in $VARIANTS; do
  blue "Merging $variant across docs → $OUT_ROOT"
  python src/merge_corpus.py \
    --docs-file "$DOC_IDS_FILE" \
    --variant "$variant" \
    --dim "$DIM" \
    --out-root "$OUT_ROOT"

  CH="${OUT_ROOT}/chunks.${variant}.jsonl"
  CE="${OUT_ROOT}/chunk_embs.${variant}.d${DIM}.npy"
  Q="${OUT_ROOT}/intents.flat.jsonl"
  S="${OUT_ROOT}/gt_spans.jsonl"

  echo "=== ${variant} (doc mode) ==="
  python src/eval_retrieval.py \
    --chunk-embs "$CE" \
    --chunks "$CH" \
    --queries "$Q" \
    --embedder "$EMBEDDER" --dim "$DIM" --mode doc --topk 5 || true

  echo "=== ${variant} (span mode) ==="
  python src/eval_retrieval.py \
    --chunk-embs "$CE" \
    --chunks "$CH" \
    --queries "$Q" \
    --spans "$S" \
    --embedder "$EMBEDDER" --dim "$DIM" --mode span --topk 5 || true
done

# 5) Summary with CIs (optional)
blue "Computing stats with bootstrap CIs across available variants"
python src/stats_summary.py \
  --out-dir "$OUT_ROOT" \
  --embedder "$EMBEDDER" \
  --dim "$DIM" \
  --json-out "${OUT_ROOT}/stats.json" || true

green "Done. Results in: $OUT_ROOT"
