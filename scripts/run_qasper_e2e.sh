#!/usr/bin/env bash
set -euo pipefail

# End-to-end Qasper dev runner (official JSON format): convert → run per doc → merge → evaluate

DATA_DIR="data/qasper"
INPUT_DIR="data/input"
OUT_ROOT="out/qasper"
QASPER_JSON="${QASPER_PATH:-${DATA_DIR}/qasper-dev-v0.3.json}"
DIM="${DIM:-1536}"
EMBEDDER="${EMBEDDER:-gemini-embedding-001}"
DEFAULT_DOC_LIMIT="${DEFAULT_DOC_LIMIT:-100}"
LIMIT="${LIMIT:-${DEFAULT_DOC_LIMIT}}"
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

if [[ ! -f "$QASPER_JSON" ]]; then
  red "Qasper JSON not found: $QASPER_JSON"
  echo "Run scripts/download_qasper.sh to fetch qasper-dev-v0.3.json"
  exit 1
fi

DOC_IDS_FILE="${DATA_DIR}/doc_ids.txt"
blue "Converting Qasper → IDC inputs (limit=${LIMIT})"
if [[ "${LIMIT}" -gt 0 ]]; then
  python src/convert_qasper.py --qasper "$QASPER_JSON" --doc-ids-out "$DOC_IDS_FILE" --limit "$LIMIT"
else
  python src/convert_qasper.py --qasper "$QASPER_JSON" --doc-ids-out "$DOC_IDS_FILE"
fi

if [[ ! -s "$DOC_IDS_FILE" ]]; then
  red "No doc ids found at $DOC_IDS_FILE"
  exit 1
fi

blue "Running pipeline per doc (AUTO_TUNE=${AUTO_TUNE}, AUTO_TUNE_BASELINES=${AUTO_TUNE_BASELINES}, keep gold spans)"
while IFS= read -r doc; do
  [[ -z "$doc" ]] && continue
  echo "--- ${doc} ---"
  FLAGS=( --doc "$doc" --target-avg "$TARGET_AVG" --tol "$TOL" )
  if [[ "$AUTO_TUNE" == "1" ]]; then FLAGS+=( --auto-tune ); fi
  if [[ "$AUTO_TUNE_BASELINES" == "1" ]]; then FLAGS+=( --auto-tune-baselines ); fi
  RESPECT_EXISTING_INTENTS=1 python src/cli.py run "${FLAGS[@]}"
done < "$DOC_IDS_FILE"

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

blue "Computing stats with bootstrap CIs across available variants"
python src/stats_summary.py \
  --out-dir "$OUT_ROOT" \
  --embedder "$EMBEDDER" \
  --dim "$DIM" \
  --json-out "${OUT_ROOT}/stats.json" || true

green "Done. Results in: $OUT_ROOT"
