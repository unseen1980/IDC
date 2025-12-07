#!/usr/bin/env bash
set -euo pipefail

# --------------------------
# Config (adjust as needed)
# --------------------------
# Allow direct INPUT_FILE override. If provided, derive DOC_NAME from it unless
# explicitly set. Otherwise, default to data/input/<DOC_NAME>.txt
if [[ -n "${INPUT_FILE:-}" ]]; then
  # Normalize to relative path if possible for nicer outputs
  if [[ -f "${INPUT_FILE}" ]]; then
    : # ok
  else
    echo "[WARN] INPUT_FILE set but not found: ${INPUT_FILE}" >&2
  fi
  # Derive DOC_NAME if not given, from basename without extension
  DOC_NAME="${DOC_NAME:-$(basename "${INPUT_FILE}")}"; DOC_NAME="${DOC_NAME%.*}"
else
  DOC_NAME="${DOC_NAME:-idc}"
  INPUT_FILE="data/input/${DOC_NAME}.txt"
fi

# Compute input directory and glob for single-document operations
INPUT_DIR="$(dirname "${INPUT_FILE}")"
INPUT_GLOB="$(basename "${INPUT_FILE}")"

# Models & dimensions
GEN_MODEL="gemini-2.5-flash"
EMBEDDER="gemini-embedding-001"
DIM="${DIM:-1536}"

# IDC hyperparams
LAMBDA="${LAMBDA:-0.1}"         # e.g., 0.001..0.1 (lower = longer chunks, higher = more boundaries)
MAX_LEN="${MAX_LEN:-20}"        # sentences per chunk cap (optimal from auto-tuning)
MIN_LEN="${MIN_LEN:-3}"         # minimum sentences per chunk (raise to reduce micro-chunks)
BOUNDARY_PENALTY="${BOUNDARY_PENALTY:-1.2}"   # penalty per chunk/boundary (optimal from auto-tuning)
COHERENCE_WEIGHT="${COHERENCE_WEIGHT:-0.3}"   # weight for intra-chunk coherence bonus (optimal from auto-tuning)
MERGE_ADJACENT="${MERGE_ADJACENT:-1}"         # merge adjacent chunks with same intent
AUTO_TUNE="${AUTO_TUNE:-0}"                   # enable auto-tuning of lambda and boundary penalty
AUTO_TUNE_BASELINES="${AUTO_TUNE_BASELINES:-0}" # enable auto-tuning for baseline segmenters
TUNE_TARGET_AVG="${TUNE_TARGET_AVG:-7.5}"     # target ~7-8 sentences per chunk
TUNE_TOL="${TUNE_TOL:-2.0}"                    # tolerance for target avg (±2.0 sentences)
EVAL_EMBEDDER="${EVAL_EMBEDDER:-}"             # alternate embedder just for pseudo-span query embeddings

# NEW: Contextual embeddings (default enabled for improved retrieval)
CONTEXTUAL_EMBEDDINGS="${CONTEXTUAL_EMBEDDINGS:-1}"  # enable contextual embeddings with adjacent context
CONTEXT_WEIGHT="${CONTEXT_WEIGHT:-0.15}"             # weight for adjacent chunk context (0.15 per side)

# NEW: Information density-aware segmentation (default enabled for better content-aware chunking)
DENSITY_AWARE="${DENSITY_AWARE:-1}"                  # enable density-aware segmentation
DENSITY_DISCOUNT_FACTOR="${DENSITY_DISCOUNT_FACTOR:-0.3}"  # discount factor for dense regions (0.0-1.0)

# Optional IDC tuning grids (space-separated)
IDC_LAM_GRID="${IDC_LAM_GRID:-0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1}"
IDC_BP_GRID="${IDC_BP_GRID:-0.20 0.25 0.30 0.40 0.60 0.80 1.00 1.20}"  # Expand range so penalties matter
IDC_MAXLEN_GRID="${IDC_MAXLEN_GRID:-12 16 20}"  # Larger chunks for better coverage
IDC_COHW_GRID="${IDC_COHW_GRID:-0.0 0.1 0.2 0.3}"  # Higher coherence weights

# Coherence baseline hyperparams
COH_WIN="${COH_WIN:-1}"
COH_MIN="${COH_MIN:-1}"
COH_MAX="${COH_MAX:-10}"
COH_APPROX="${COH_APPROX:-6}"
COH_TAG="w${COH_WIN}.min${COH_MIN}.max${COH_MAX}.a${COH_APPROX}"

# Intent generation for large docs
# Auto-switch to chunked intents if file > THRESHOLD_CHARS
THRESHOLD_CHARS=12000
CHUNKED_INTENTS="${CHUNKED_INTENTS:-auto}"     # auto|1|0
# Respect pre-existing intents (e.g., from dataset converters) regardless of timestamps
RESPECT_EXISTING_INTENTS="${RESPECT_EXISTING_INTENTS:-0}"
CHARS_PER_CHUNK=12000
QUESTIONS_PER_CHUNK=3
MAX_OUTPUT_TOKENS=8192      # for chunked
NUM_QUESTIONS_ONEPASS="${NUM_QUESTIONS_ONEPASS:-15}"    # for one-pass (intents.py) - can be auto-adapted
GENERATION_MULTIPLIER="${GENERATION_MULTIPLIER:-1.5}"   # intent generation multiplier
DIVERSITY_THRESHOLD="${DIVERSITY_THRESHOLD:-0.4}"       # diversity threshold for intent filtering
AUTO_ADAPT_INTENTS="${AUTO_ADAPT_INTENTS:-0}"          # auto-adapt intent params based on doc length (1=enable)

# Paths
OUT_DIR="out/${DOC_NAME}"
SENTS_JSON="${OUT_DIR}/sentences.jsonl"
SENTS_META="${OUT_DIR}/sentences.meta.jsonl"
INTENTS_JSON="${OUT_DIR}/predicted_intents.jsonl"
INTENTS_FLAT="${OUT_DIR}/intents.flat.jsonl"
SENT_EMBS="${OUT_DIR}/sentence_embs.d${DIM}.npy"
INTENT_EMBS="${OUT_DIR}/intent_embs.d${DIM}.npy"

IDC_SEG="${OUT_DIR}/segments.idc.l$(printf "%.0f" "$(echo "${LAMBDA} * 1000" | bc 2>/dev/null || echo 2)").L${MAX_LEN}.m${MIN_LEN}.b$(printf "%.0f" "$(echo "${BOUNDARY_PENALTY} * 100" | bc 2>/dev/null || echo 25)").c$(printf "%.0f" "$(echo "${COHERENCE_WEIGHT} * 100" | bc 2>/dev/null || echo 5)").jsonl"
IDC_CHUNKS="${OUT_DIR}/chunks.idc.jsonl"
IDC_CHUNK_EMBS="${OUT_DIR}/chunk_embs.idc.d${DIM}.npy"

FIXED_SEG="${OUT_DIR}/segments.fixed.s6.jsonl"
SLID_SEG="${OUT_DIR}/segments.sliding.s6r3.jsonl"
COH_SEG="${OUT_DIR}/segments.coh.${COH_TAG}.jsonl"
PARA_SEG="${OUT_DIR}/segments.paragraphs.jsonl"

FIXED_CHUNKS="${OUT_DIR}/chunks.fixed.s6.jsonl"
SLID_CHUNKS="${OUT_DIR}/chunks.sliding.s6r3.jsonl"
COH_CHUNKS="${OUT_DIR}/chunks.coh.${COH_TAG}.jsonl"
PARA_CHUNKS="${OUT_DIR}/chunks.paragraphs.jsonl"

FIXED_EMBS="${OUT_DIR}/chunk_embs.fixed.s6.d${DIM}.npy"
SLID_EMBS="${OUT_DIR}/chunk_embs.sliding.s6r3.d${DIM}.npy"
COH_EMBS="${OUT_DIR}/chunk_embs.coh.${COH_TAG}.d${DIM}.npy"
PARA_EMBS="${OUT_DIR}/chunk_embs.paragraphs.d${DIM}.npy"

# --------------- helpers ---------------
red()    { printf "\033[31m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
blue()   { printf "\033[34m%s\033[0m\n" "$*"; }

need() {
  command -v "$1" >/dev/null 2>&1 || { red "Missing command: $1"; exit 1; }
}

# Check if file1 is newer than file2 (or file2 doesn't exist)
is_newer() {
  local file1="$1"
  local file2="$2"
  [[ ! -f "$file2" ]] || [[ "$file1" -nt "$file2" ]]
}

# Check if output file exists and is newer than input file
should_skip() {
  local input_file="$1"
  local output_file="$2"
  if [[ -f "$output_file" ]] && [[ "$output_file" -nt "$input_file" ]]; then
    return 0  # true - should skip
  else
    return 1  # false - should not skip
  fi
}

# --------------- checks ----------------
need python
mkdir -p "${OUT_DIR}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  red "Input file not found: ${INPUT_FILE}"
  exit 1
fi

# Compute byte size (portable)
FILE_BYTES=$(wc -c < "${INPUT_FILE}" | tr -d '[:space:]')
yellow "Doc: ${INPUT_FILE} — ${FILE_BYTES} bytes"

# Decide chunked vs one-pass intents
USE_CHUNKED="0"
if [[ "${CHUNKED_INTENTS}" == "1" ]]; then
  USE_CHUNKED="1"
elif [[ "${CHUNKED_INTENTS}" == "0" ]]; then
  USE_CHUNKED="0"
else
  # auto
  if (( FILE_BYTES > THRESHOLD_CHARS )); then
    USE_CHUNKED="1"
  fi
fi

# --------------- Step 1: Sentences ---------------
blue "Step 1/9: Preprocess → sentences"
if should_skip "${INPUT_FILE}" "${SENTS_JSON}"; then
  yellow "Skipping sentence preprocessing (${SENTS_JSON} is up-to-date)"
else
  python src/preprocess.py \
    --input "${INPUT_DIR}" \
    --glob "${INPUT_GLOB}" \
    --out "${SENTS_JSON}"
fi

# --------------- Auto-Adapt Intent Parameters ----
if [[ "${AUTO_ADAPT_INTENTS}" == "1" ]] && [[ -f "${SENTS_JSON}" ]]; then
  blue "🎯 Auto-adapting intent generation parameters based on document characteristics..."
  if [[ -f "src/adaptive_params.py" ]]; then
    # Capture adaptive params (overwrites user-provided values)
    eval $(python src/adaptive_params.py --sentences "${SENTS_JSON}" --output-env)
    echo "  Adapted: NUM_QUESTIONS_ONEPASS=${NUM_QUESTIONS_ONEPASS}, GENERATION_MULTIPLIER=${GENERATION_MULTIPLIER}, DIVERSITY_THRESHOLD=${DIVERSITY_THRESHOLD}"
  else
    yellow "Warning: src/adaptive_params.py not found, using default parameters"
  fi
fi

# --------------- Step 2: Intents -----------------
blue "Step 2/10: Generate intents ($([[ "${USE_CHUNKED}" == "1" ]] && echo 'chunked' || echo 'one-pass'))"

# Respect existing intents; allow UI to force regeneration via FORCE_INTENTS=1
if [[ "${RESPECT_EXISTING_INTENTS}" == "1" ]] && [[ -f "${INTENTS_JSON}" ]]; then
  yellow "Respecting existing intents (${INTENTS_JSON})"
elif [[ "${FORCE_INTENTS:-0}" != "1" ]] && should_skip "${INPUT_FILE}" "${INTENTS_JSON}"; then
  yellow "Skipping intent generation (${INTENTS_JSON} is up-to-date)"
else
  # Check for universal intent generation
  if [[ "${UNIVERSAL_INTENTS:-false}" == "true" ]] && [[ -f "src/intents_universal.py" ]]; then
    blue "Using Universal Intent Generation (structure-aware, no dataset-specific cheating)"
    python src/intents_universal.py \
      --input "${INPUT_DIR}" \
      --out "${INTENTS_JSON}" \
      --model "${GEN_MODEL}" \
      --num-questions "${NUM_QUESTIONS_ONEPASS}" \
      --max-output-tokens "${ONEPASS_MAX_TOKENS:-2048}"
  elif [[ "${USE_CHUNKED}" == "1" ]]; then
    if [[ -f "src/intents_chunked.py" ]]; then
      python src/intents_chunked.py \
        --input "${INPUT_FILE}" \
        --out "${INTENTS_JSON}" \
        --model "${GEN_MODEL}" \
        --chars-per-chunk "${CHARS_PER_CHUNK}" \
        --questions-per-chunk "${QUESTIONS_PER_CHUNK}" \
        --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
        ${INTENTS_MAX:+--max-intents "${INTENTS_MAX}"} \
        ${GENERATION_MULTIPLIER:+--generation-multiplier "${GENERATION_MULTIPLIER}"} \
        $([ "${DISABLE_MMR:-0}" = "1" ] && echo "--disable-mmr") \
        ${DIVERSITY_EMBEDDER:+--embedder "${DIVERSITY_EMBEDDER}"} \
        ${DIVERSITY_DIM:+--embed-dim "${DIVERSITY_DIM}"}
    else
      yellow "src/intents_chunked.py not found; falling back to one-pass intents.py"
      python src/intents.py \
        --input "${INPUT_DIR}" \
        --out "${INTENTS_JSON}" \
        --model "${GEN_MODEL}" \
        --num-questions "${NUM_QUESTIONS_ONEPASS}" \
        --max-output-tokens "${ONEPASS_MAX_TOKENS:-2048}"
    fi
  else
    python src/intents.py \
      --input "${INPUT_DIR}" \
      --out "${INTENTS_JSON}" \
      --model "${GEN_MODEL}" \
      --num-questions "${NUM_QUESTIONS_ONEPASS}" \
      --max-output-tokens "${ONEPASS_MAX_TOKENS:-2048}"
  fi
fi

# --------------- Step 3: Embeddings --------------
blue "Step 3/10: Embeddings (sentences + intents)"

# Sentence embeddings
if should_skip "${SENTS_JSON}" "${SENT_EMBS}"; then
  yellow "Skipping sentence embeddings (${SENT_EMBS} is up-to-date)"
else
  # Check for query-aware embeddings
  if [[ "${QUERY_AWARE:-false}" == "true" ]] && [[ -f "${INTENTS_JSON}" ]]; then
    blue "Using Query-Aware Embeddings (optimized for retrieval tasks)"
    # Extract sample queries for query-aware embedding
    SAMPLE_QUERIES=($(python -c "
import json
queries = []
try:
    with open('${INTENTS_JSON}', 'r') as f:
        for line in f:
            data = json.loads(line)
            queries.extend(data.get('predicted_queries', [])[:2])
            if len(queries) >= 5:
                break
    for q in queries[:5]:
        print(repr(q))
except:
    pass
" 2>/dev/null || echo))
    
    if [[ ${#SAMPLE_QUERIES[@]} -gt 0 ]]; then
      blue "Found ${#SAMPLE_QUERIES[@]} sample queries for query-aware embeddings"
      python src/embed.py --embedder "${EMBEDDER}" --dim "${DIM}" sentences \
        --sentences "${SENTS_JSON}" \
        --out-npy "${SENT_EMBS}" \
        --out-meta "${SENTS_META}" \
        --query-aware --sample-queries "${SAMPLE_QUERIES[@]}"
    else
      yellow "No sample queries found, falling back to standard sentence embeddings"
      python src/embed.py --embedder "${EMBEDDER}" --dim "${DIM}" sentences \
        --sentences "${SENTS_JSON}" \
        --out-npy "${SENT_EMBS}" \
        --out-meta "${SENTS_META}"
    fi
  else
    python src/embed.py --embedder "${EMBEDDER}" --dim "${DIM}" sentences \
      --sentences "${SENTS_JSON}" \
      --out-npy "${SENT_EMBS}" \
      --out-meta "${SENTS_META}"
  fi
fi

# Intent embeddings
if should_skip "${INTENTS_JSON}" "${INTENT_EMBS}"; then
  yellow "Skipping intent embeddings (${INTENT_EMBS} is up-to-date)"
else
  python src/embed.py --embedder "${EMBEDDER}" --dim "${DIM}" intents \
    --intents "${INTENTS_JSON}" \
    --out-npy "${INTENT_EMBS}" \
    --out-flat "${INTENTS_FLAT}"
fi

# --------------- Step 4: IDC segmentation --------
blue "Step 4/10: IDC segmentation (λ=${LAMBDA}, max_len=${MAX_LEN})"

# Check if we need to regenerate IDC segmentation (depends on multiple inputs)
SKIP_IDC_SEG=false
if [[ -f "${IDC_SEG}" ]]; then
  if [[ "${IDC_SEG}" -nt "${SENTS_JSON}" ]] && \
     [[ "${IDC_SEG}" -nt "${SENT_EMBS}" ]] && \
     [[ "${IDC_SEG}" -nt "${INTENTS_FLAT}" ]] && \
     [[ "${IDC_SEG}" -nt "${INTENT_EMBS}" ]]; then
    SKIP_IDC_SEG=true
  fi
fi

# Force segmentation when auto-tuning
if [[ "${AUTO_TUNE}" == "1" ]]; then
  SKIP_IDC_SEG=false
fi

if [[ "${SKIP_IDC_SEG}" == "true" ]]; then
  yellow "Skipping IDC segmentation (${IDC_SEG} is up-to-date)"
else
  if [[ "${AUTO_TUNE}" == "1" ]]; then
    yellow "Auto-tuning IDC parameters for maximum coverage..."
    # Prefer gold spans for tuning if available; otherwise create pseudo spans
    TUNE_SPANS="${OUT_DIR}/gt_spans.jsonl"
    if [[ ! -f "${TUNE_SPANS}" ]]; then
      TUNE_SPANS="${OUT_DIR}/tune_spans.jsonl"
      if [[ ! -f "${TUNE_SPANS}" ]]; then
        python src/make_pseudo_spans.py \
          --intents "${INTENTS_FLAT}" \
          --sentences "${SENTS_JSON}" \
          --sentence-embs "${SENT_EMBS}" \
          --out "${TUNE_SPANS}" \
          --embedder "${EMBEDDER}" ${EVAL_EMBEDDER:+--eval-embedder "${EVAL_EMBEDDER}"} --dim "${DIM}" \
          --threshold 0.45 --neighbor-frac 0.90 --max-span-len 6
      fi
    fi
    
    # Run auto-tuning
    TUNE_REPORT="${OUT_DIR}/auto_tune_report.json"
    set +e
    python src/auto_tune.py \
      --sentences "${SENTS_JSON}" \
      --sentence-embs "${SENT_EMBS}" \
      --sentences-meta "${SENTS_META}" \
      --intents-flat "${INTENTS_FLAT}" \
      --intent-embs "${INTENT_EMBS}" \
      --spans "${TUNE_SPANS}" \
      --max-len "${MAX_LEN}" \
      --min-len "${MIN_LEN}" \
      --coherence-weight "${COHERENCE_WEIGHT}" \
      --lambda-values ${IDC_LAM_GRID} \
      --boundary-penalty-values ${IDC_BP_GRID} \
      --max-len-grid ${IDC_MAXLEN_GRID} \
      --coherence-weight-grid ${IDC_COHW_GRID} \
      ${TUNE_TARGET_AVG:+--target-avg "${TUNE_TARGET_AVG}"} \
      ${TUNE_TOL:+--tolerance "${TUNE_TOL}"} \
      --optimize "${IDC_OPTIMIZE:-combined}" \
      --eval-embedder "${EVAL_EMBEDDER:-${EMBEDDER}}" \
      --dim "${DIM}" \
      $([ "${MERGE_ADJACENT}" = "1" ] && echo "--merge-adjacent") \
      --out "${IDC_SEG}" \
      --report "${TUNE_REPORT}"
    TUNE_RC=$?
    set -e
    if [[ "${TUNE_RC}" != "0" ]]; then
      yellow "Auto-tune failed or coverage=0. Falling back to regular segmentation."
      # Check for paragraph-aware algorithm fallback
      if [[ "${PARAGRAPH_AWARE:-false}" == "true" ]] && [[ -f "src/idc_paragraph_aware.py" ]]; then
        blue "Falling back to Paragraph-Aware Algorithm (respects natural boundaries + intent optimization)"
        python src/idc_paragraph_aware.py \
          --sentences "${SENTS_JSON}" \
          --sentence-embs "${SENT_EMBS}" \
          --intents-flat "${INTENTS_FLAT}" \
          --intent-embs "${INTENT_EMBS}" \
          --out "${IDC_SEG}"
      else
        DENSITY_FLAG=""
        if [[ "${DENSITY_AWARE}" == "1" ]]; then
          DENSITY_FLAG="--density-aware --density-discount-factor ${DENSITY_DISCOUNT_FACTOR}"
        fi
        python src/idc_core.py \
          --sentences "${SENTS_JSON}" \
          --sentence-embs "${SENT_EMBS}" \
          --sentences-meta "${SENTS_META}" \
          --intents-flat "${INTENTS_FLAT}" \
          --intent-embs "${INTENT_EMBS}" \
          --lambda "${LAMBDA}" \
          --max-len "${MAX_LEN}" \
          --min-len "${MIN_LEN}" \
          --boundary-penalty "${BOUNDARY_PENALTY}" \
          --coherence-weight "${COHERENCE_WEIGHT}" \
          --doc-id "${DOC_NAME}" \
          --respect-paragraphs \
          $([ "${MERGE_ADJACENT}" = "1" ] && echo "--merge-adjacent") \
          ${DENSITY_FLAG} \
          --out "${IDC_SEG}"
      fi
    fi
    
    if [[ -f "${TUNE_REPORT}" ]]; then
      green "Auto-tuning completed. Report: ${TUNE_REPORT}"
      # Extract best parameters for display
      BEST_LAMBDA=$(python -c "import json; r=json.load(open('${TUNE_REPORT}')); print(r.get('best_lambda', 'N/A'))")
      BEST_BP=$(python -c "import json; r=json.load(open('${TUNE_REPORT}')); print(r.get('best_boundary_penalty', 'N/A'))")
      BEST_COV=$(python -c "import json; r=json.load(open('${TUNE_REPORT}')); print(r.get('best_coverage', 'N/A'))")
      BEST_ML=$(python -c "import json; r=json.load(open('${TUNE_REPORT}')); print(r.get('best_max_len', 'N/A'))")
      BEST_CW=$(python -c "import json; r=json.load(open('${TUNE_REPORT}')); print(r.get('best_coherence_weight', 'N/A'))")
      green "Best params: λ=${BEST_LAMBDA}, boundary_penalty=${BEST_BP}, max_len=${BEST_ML}, C=${BEST_CW}, coverage=${BEST_COV}"
      # Optionally rename IDC_SEG to reflect tuned params
      if [[ "${BEST_LAMBDA}" != "N/A" ]] && [[ "${BEST_BP}" != "N/A" ]] && [[ "${BEST_ML}" != "N/A" ]] && [[ -f "${IDC_SEG}" ]]; then
        NEW_IDC_SEG="${OUT_DIR}/segments.idc.l$(printf '%.0f' "$(echo "${BEST_LAMBDA} * 1000" | bc 2>/dev/null)").L${BEST_ML}.m${MIN_LEN}.b$(printf '%.0f' "$(echo "${BEST_BP} * 100" | bc 2>/dev/null)").c$(printf '%.0f' "$(echo "${BEST_CW} * 100" | bc 2>/dev/null)").jsonl"
      if [[ "${NEW_IDC_SEG}" != "${IDC_SEG}" ]]; then
        mv -f "${IDC_SEG}" "${NEW_IDC_SEG}"
        IDC_SEG="${NEW_IDC_SEG}"
      fi
    fi

      # If paragraph-aware mode is enabled, re-apply advanced paragraph-aware segmentation
      if [[ "${PARAGRAPH_AWARE:-false}" == "true" ]] && [[ -f "src/idc_paragraph_aware.py" ]]; then
        blue "Re-applying Paragraph-Aware Algorithm (overriding tuned segments)"
        python src/idc_paragraph_aware.py \
          --sentences "${SENTS_JSON}" \
          --sentence-embs "${SENT_EMBS}" \
          --intents-flat "${INTENTS_FLAT}" \
          --intent-embs "${INTENT_EMBS}" \
          --out "${IDC_SEG}"
      fi
    fi
  else
    # Check for paragraph-aware algorithm
    if [[ "${PARAGRAPH_AWARE:-false}" == "true" ]] && [[ -f "src/idc_paragraph_aware.py" ]]; then
      blue "Using Paragraph-Aware Algorithm (respects natural boundaries + intent optimization)"
      python src/idc_paragraph_aware.py \
        --sentences "${SENTS_JSON}" \
        --sentence-embs "${SENT_EMBS}" \
        --intents-flat "${INTENTS_FLAT}" \
        --intent-embs "${INTENT_EMBS}" \
        --out "${IDC_SEG}"
    else
      # Regular segmentation with enhanced features
      DENSITY_FLAG=""
      if [[ "${DENSITY_AWARE}" == "1" ]]; then
        DENSITY_FLAG="--density-aware --density-discount-factor ${DENSITY_DISCOUNT_FACTOR}"
      fi
      python src/idc_core.py \
        --sentences "${SENTS_JSON}" \
        --sentence-embs "${SENT_EMBS}" \
        --sentences-meta "${SENTS_META}" \
        --intents-flat "${INTENTS_FLAT}" \
        --intent-embs "${INTENT_EMBS}" \
        --lambda "${LAMBDA}" \
        --max-len "${MAX_LEN}" \
        --min-len "${MIN_LEN}" \
        --boundary-penalty "${BOUNDARY_PENALTY}" \
        --coherence-weight "${COHERENCE_WEIGHT}" \
        --doc-id "${DOC_NAME}" \
        --respect-paragraphs \
        $([ "${MERGE_ADJACENT}" = "1" ] && echo "--merge-adjacent") \
        ${DENSITY_FLAG} \
        --out "${IDC_SEG}"
    fi
  fi
fi

# --------------- Step 5: Baselines ----------------
blue "Step 5/10: Baselines (fixed/sliding/coherence/paragraphs)"

if [[ "${AUTO_TUNE_BASELINES}" == "1" ]]; then
  yellow "Auto-tuning baselines against dev spans..."
  # Ensure dev spans exist for tuning
  TUNE_SPANS="${OUT_DIR}/tune_spans.jsonl"
  if [[ ! -f "${TUNE_SPANS}" ]]; then
    python src/make_pseudo_spans.py \
      --intents "${INTENTS_FLAT}" \
      --sentences "${SENTS_JSON}" \
      --sentence-embs "${SENT_EMBS}" \
      --out "${TUNE_SPANS}" \
      --embedder "${EMBEDDER}" ${EVAL_EMBEDDER:+--eval-embedder "${EVAL_EMBEDDER}"} --dim "${DIM}" \
      --threshold 0.45 --neighbor-frac 0.90 --max-span-len 6
  fi
  python src/auto_tune_baselines.py \
    --sentences "${SENTS_JSON}" \
    --sentence-embs "${SENT_EMBS}" \
    --input-dir "data/input" \
    --glob "${DOC_NAME}.txt" \
    --spans "${TUNE_SPANS}" \
    --out-fixed "${FIXED_SEG}" \
    --out-sliding "${SLID_SEG}" \
    --out-coherence "${COH_SEG}" \
    --out-paragraphs "${PARA_SEG}" \
    ${TUNE_TARGET_AVG:+--target-avg "${TUNE_TARGET_AVG}"} \
    ${TUNE_TOL:+--tolerance "${TUNE_TOL}"} \
    --report "${OUT_DIR}/auto_tune_baselines_report.json"
  # Rename tuned outputs to parameterized filenames and update variables
  REP="${OUT_DIR}/auto_tune_baselines_report.json"
  if [[ -f "${REP}" ]]; then
    BEST_TL=$(python -c "import json; r=json.load(open('${REP}')); print(int(r['fixed']['params']['target_len']))")
    NEW_FIXED_SEG="${OUT_DIR}/segments.fixed.s${BEST_TL}.jsonl"
    if [[ -f "${FIXED_SEG}" ]] && [[ "${FIXED_SEG}" != "${NEW_FIXED_SEG}" ]]; then mv -f "${FIXED_SEG}" "${NEW_FIXED_SEG}"; fi
    FIXED_SEG="${NEW_FIXED_SEG}"
    FIXED_CHUNKS="${OUT_DIR}/chunks.fixed.s${BEST_TL}.jsonl"
    FIXED_EMBS="${OUT_DIR}/chunk_embs.fixed.s${BEST_TL}.d${DIM}.npy"

    BEST_SIZE=$(python -c "import json; r=json.load(open('${REP}')); print(int(r['sliding']['params']['size']))")
    BEST_STRIDE=$(python -c "import json; r=json.load(open('${REP}')); print(int(r['sliding']['params']['stride']))")
    NEW_SLID_SEG="${OUT_DIR}/segments.sliding.s${BEST_SIZE}r${BEST_STRIDE}.jsonl"
    if [[ -f "${SLID_SEG}" ]] && [[ "${SLID_SEG}" != "${NEW_SLID_SEG}" ]]; then mv -f "${SLID_SEG}" "${NEW_SLID_SEG}"; fi
    SLID_SEG="${NEW_SLID_SEG}"
    SLID_CHUNKS="${OUT_DIR}/chunks.sliding.s${BEST_SIZE}r${BEST_STRIDE}.jsonl"
    SLID_EMBS="${OUT_DIR}/chunk_embs.sliding.s${BEST_SIZE}r${BEST_STRIDE}.d${DIM}.npy"

    BEST_WIN=$(python -c "import json; r=json.load(open('${REP}')); v=r['coherence']['params'].get('win', None); print(v if v is not None else -1)")
    BEST_APPROX=$(python -c "import json; r=json.load(open('${REP}')); v=r['coherence']['params'].get('approx_chunk_len', None); print(v if v is not None else -1)")
    if [[ "${BEST_WIN}" != "-1" ]] && [[ "${BEST_APPROX}" != "-1" ]]; then
      COH_TAG="w${BEST_WIN}.min${COH_MIN}.max${COH_MAX}.a${BEST_APPROX}"
      NEW_COH_SEG="${OUT_DIR}/segments.coh.${COH_TAG}.jsonl"
      if [[ -f "${COH_SEG}" ]] && [[ "${COH_SEG}" != "${NEW_COH_SEG}" ]]; then mv -f "${COH_SEG}" "${NEW_COH_SEG}"; fi
      COH_SEG="${NEW_COH_SEG}"
      COH_CHUNKS="${OUT_DIR}/chunks.coh.${COH_TAG}.jsonl"
      COH_EMBS="${OUT_DIR}/chunk_embs.coh.${COH_TAG}.d${DIM}.npy"
    fi
  fi
else
  # Fixed segmentation
  if should_skip "${SENTS_JSON}" "${FIXED_SEG}"; then
    yellow "Skipping fixed segmentation (${FIXED_SEG} is up-to-date)"
  else
    python src/baselines.py --sentences "${SENTS_JSON}" --out "${FIXED_SEG}" \
      fixed --target-len 6
  fi

  # Sliding segmentation
  if should_skip "${SENTS_JSON}" "${SLID_SEG}"; then
    yellow "Skipping sliding segmentation (${SLID_SEG} is up-to-date)"
  else
    python src/baselines.py --sentences "${SENTS_JSON}" --out "${SLID_SEG}" \
      sliding --size 6 --stride 3
  fi

  # Coherence segmentation (depends on both sentences and embeddings)
  SKIP_COH=false
  if [[ -f "${COH_SEG}" ]]; then
    if [[ "${COH_SEG}" -nt "${SENTS_JSON}" ]] && [[ "${COH_SEG}" -nt "${SENT_EMBS}" ]]; then
      SKIP_COH=true
    fi
  fi
  if [[ "${SKIP_COH}" == "true" ]]; then
    yellow "Skipping coherence segmentation (${COH_SEG} is up-to-date)"
  else
    python src/baselines.py --sentences "${SENTS_JSON}" --sentence-embs "${SENT_EMBS}" --out "${COH_SEG}" \
      coherence --win "${COH_WIN}" --min-len "${COH_MIN}" --max-len "${COH_MAX}" --approx-chunk-len "${COH_APPROX}"
  fi

  # Paragraph segmentation (depends on both sentences and input file)
  SKIP_PARA=false
  if [[ -f "${PARA_SEG}" ]]; then
    if [[ "${PARA_SEG}" -nt "${SENTS_JSON}" ]] && [[ "${PARA_SEG}" -nt "${INPUT_FILE}" ]]; then
      SKIP_PARA=true
    fi
  fi
  if [[ "${SKIP_PARA}" == "true" ]]; then
    yellow "Skipping paragraph segmentation (${PARA_SEG} is up-to-date)"
  else
    python src/baselines.py --sentences "${SENTS_JSON}" --input-dir "${INPUT_DIR}" --glob "${INPUT_GLOB}" --out "${PARA_SEG}" \
      paragraphs
  fi
fi

# --------------- Step 6: Build chunk embeddings ---
blue "Step 6/10: Build chunk embeddings (IDC + baselines)"

# IDC chunk embeddings
SKIP_IDC_CHUNKS=false
if [[ -f "${IDC_CHUNK_EMBS}" ]] && [[ -f "${IDC_CHUNKS}" ]]; then
  if [[ "${IDC_CHUNK_EMBS}" -nt "${IDC_SEG}" ]] && [[ "${IDC_CHUNK_EMBS}" -nt "${SENT_EMBS}" ]]; then
    SKIP_IDC_CHUNKS=true
  fi
fi
if [[ "${SKIP_IDC_CHUNKS}" == "true" ]]; then
  yellow "Skipping IDC chunk embeddings (${IDC_CHUNK_EMBS} is up-to-date)"
else
  CONTEXTUAL_FLAG=""
  if [[ "${CONTEXTUAL_EMBEDDINGS}" == "1" ]]; then
    CONTEXTUAL_FLAG="--contextual-embeddings --context-weight ${CONTEXT_WEIGHT}"
  fi
  python src/build_chunks.py \
    --sentences "${SENTS_JSON}" \
    --sentence-embs "${SENT_EMBS}" \
    --segments "${IDC_SEG}" \
    --out-embs "${IDC_CHUNK_EMBS}" \
    --out-chunks "${IDC_CHUNKS}" \
    --out-index "${OUT_DIR}/chunks.index.idc.jsonl" \
    --normalize \
    --intent-weighted --intent-embs "${INTENT_EMBS}" --intents-flat "${INTENTS_FLAT}" \
    ${CONTEXTUAL_FLAG}
fi

# Fixed baseline chunk embeddings
SKIP_FIXED_CHUNKS=false
if [[ -f "${FIXED_EMBS}" ]] && [[ -f "${FIXED_CHUNKS}" ]]; then
  if [[ "${FIXED_EMBS}" -nt "${FIXED_SEG}" ]] && [[ "${FIXED_EMBS}" -nt "${SENT_EMBS}" ]]; then
    SKIP_FIXED_CHUNKS=true
  fi
fi
if [[ "${SKIP_FIXED_CHUNKS}" == "true" ]]; then
  yellow "Skipping fixed chunk embeddings (${FIXED_EMBS} is up-to-date)"
else
  python src/build_chunks.py \
    --sentences "${SENTS_JSON}" \
    --sentence-embs "${SENT_EMBS}" \
    --segments "${FIXED_SEG}" \
    --out-embs "${FIXED_EMBS}" \
    --out-chunks "${FIXED_CHUNKS}" \
    --out-index "${OUT_DIR}/chunks.index.fixed.jsonl" \
    --normalize
fi

# Sliding baseline chunk embeddings
SKIP_SLID_CHUNKS=false
if [[ -f "${SLID_EMBS}" ]] && [[ -f "${SLID_CHUNKS}" ]]; then
  if [[ "${SLID_EMBS}" -nt "${SLID_SEG}" ]] && [[ "${SLID_EMBS}" -nt "${SENT_EMBS}" ]]; then
    SKIP_SLID_CHUNKS=true
  fi
fi
if [[ "${SKIP_SLID_CHUNKS}" == "true" ]]; then
  yellow "Skipping sliding chunk embeddings (${SLID_EMBS} is up-to-date)"
else
  python src/build_chunks.py \
    --sentences "${SENTS_JSON}" \
    --sentence-embs "${SENT_EMBS}" \
    --segments "${SLID_SEG}" \
    --out-embs "${SLID_EMBS}" \
    --out-chunks "${SLID_CHUNKS}" \
    --out-index "${OUT_DIR}/chunks.index.sliding.jsonl" \
    --normalize
fi

# Coherence baseline chunk embeddings  
SKIP_COH_CHUNKS=false
if [[ -f "${COH_EMBS}" ]] && [[ -f "${COH_CHUNKS}" ]]; then
  if [[ "${COH_EMBS}" -nt "${COH_SEG}" ]] && [[ "${COH_EMBS}" -nt "${SENT_EMBS}" ]]; then
    SKIP_COH_CHUNKS=true
  fi
fi
if [[ "${SKIP_COH_CHUNKS}" == "true" ]]; then
  yellow "Skipping coherence chunk embeddings (${COH_EMBS} is up-to-date)"
else
  python src/build_chunks.py \
    --sentences "${SENTS_JSON}" \
    --sentence-embs "${SENT_EMBS}" \
    --segments "${COH_SEG}" \
    --out-embs "${COH_EMBS}" \
    --out-chunks "${COH_CHUNKS}" \
    --out-index "${OUT_DIR}/chunks.index.coherence.jsonl" \
    --normalize
fi

# Paragraph baseline chunk embeddings
SKIP_PARA_CHUNKS=false
if [[ -f "${PARA_EMBS}" ]] && [[ -f "${PARA_CHUNKS}" ]]; then
  if [[ "${PARA_EMBS}" -nt "${PARA_SEG}" ]] && [[ "${PARA_EMBS}" -nt "${SENT_EMBS}" ]]; then
    SKIP_PARA_CHUNKS=true
  fi
fi
if [[ "${SKIP_PARA_CHUNKS}" == "true" ]]; then
  yellow "Skipping paragraph chunk embeddings (${PARA_EMBS} is up-to-date)"
else
  python src/build_chunks.py \
    --sentences "${SENTS_JSON}" \
    --sentence-embs "${SENT_EMBS}" \
    --segments "${PARA_SEG}" \
    --out-embs "${PARA_EMBS}" \
    --out-chunks "${PARA_CHUNKS}" \
    --out-index "${OUT_DIR}/chunks.index.paragraphs.jsonl" \
    --normalize
fi

# --------------- Step 7: Retrieval Evaluation -----
blue "Step 7/10: Retrieval evaluation"

# Extract variant suffixes from actual filenames to avoid hardcoding
FIXED_VARIANT=$(basename "${FIXED_CHUNKS}" .jsonl | sed 's/^chunks\.//')
SLID_VARIANT=$(basename "${SLID_CHUNKS}" .jsonl | sed 's/^chunks\.//')
COH_VARIANT=$(basename "${COH_CHUNKS}" .jsonl | sed 's/^chunks\.//')

# 7a) Doc-mode retrieval (existing)
echo "=== Doc-mode retrieval ==="
for variant in idc "${FIXED_VARIANT}" "${SLID_VARIANT}" "${COH_VARIANT}" paragraphs; do
  case "$variant" in
    idc)                emb="${IDC_CHUNK_EMBS}"; chunks="${IDC_CHUNKS}";;
    "${FIXED_VARIANT}") emb="${FIXED_EMBS}";     chunks="${FIXED_CHUNKS}";;
    "${SLID_VARIANT}")  emb="${SLID_EMBS}";      chunks="${SLID_CHUNKS}";;
    "${COH_VARIANT}")   emb="${COH_EMBS}";       chunks="${COH_CHUNKS}";;
    paragraphs)         emb="${PARA_EMBS}";      chunks="${PARA_CHUNKS}";;
  esac
  echo "=== ${variant} (doc mode) ==="
  python src/eval_retrieval.py \
    --chunk-embs "${emb}" \
    --chunks "${chunks}" \
    --queries "${INTENTS_FLAT}" \
    --embedder "${EMBEDDER}" --dim "${DIM}" --topk 5 --mode doc
done

# --------------- Step 8: Pseudo-gold spans ----------
SPANS="${OUT_DIR}/gt_spans.jsonl"
if [[ "${FORCE_SPANS:-1}" == "1" ]] || [[ ! -f "${SPANS}" ]]; then
  blue "Step 8/10: Create pseudo-gold spans"
  python src/make_pseudo_spans.py \
    --intents "${INTENTS_FLAT}" \
    --sentences "${SENTS_JSON}" \
    --sentence-embs "${SENT_EMBS}" \
    --out "${SPANS}" \
    --embedder "${EMBEDDER}" ${EVAL_EMBEDDER:+--eval-embedder "${EVAL_EMBEDDER}"} --dim "${DIM}" \
    --threshold 0.45 --neighbor-frac 0.90 --max-span-len 6
else
  blue "Step 8/10: Create pseudo-gold spans (skipped; up-to-date)"
fi

# --------------- Step 9: Retrieval (span mode) + coverage  -----
blue "Step 9/10: Span-mode retrieval + answer coverage"
for variant in idc "${FIXED_VARIANT}" "${SLID_VARIANT}" "${COH_VARIANT}" paragraphs; do
  case "$variant" in
    idc)                emb="${IDC_CHUNK_EMBS}"; chunks="${IDC_CHUNKS}"; seg="${IDC_SEG}";;
    "${FIXED_VARIANT}") emb="${FIXED_EMBS}";     chunks="${FIXED_CHUNKS}"; seg="${FIXED_SEG}";;
    "${SLID_VARIANT}")  emb="${SLID_EMBS}";      chunks="${SLID_CHUNKS}"; seg="${SLID_SEG}";;
    "${COH_VARIANT}")   emb="${COH_EMBS}";       chunks="${COH_CHUNKS}"; seg="${COH_SEG}";;
    paragraphs)         emb="${PARA_EMBS}";      chunks="${PARA_CHUNKS}"; seg="${PARA_SEG}";;
  esac
  echo "=== ${variant} (span mode) ==="
  python src/eval_retrieval.py \
    --chunk-embs "${emb}" \
    --chunks "${chunks}" \
    --queries "${INTENTS_FLAT}" \
    --spans "${SPANS}" \
    --embedder "${EMBEDDER}" --dim "${DIM}" --topk 5 --mode span
  echo "=== ${variant} coverage ==="
  python src/eval_coverage.py \
    --segments "${seg}" \
    --spans "${SPANS}" \
    --diagnostics "${OUT_DIR}/uncovered.${variant}.jsonl"
done

# --------------- Step 10: Generate stats.json for QT app -----
blue "Step 10/10: Generate stats.json for visualization"
python scripts/generate_stats_json.py \
  --output-dir "${OUT_DIR}" \
  --methods "idc fixed sliding coh paragraphs"

# --------------- Summary -----
echo ""
green "✅ Pipeline completed successfully!"
green "📁 All outputs saved to: ${OUT_DIR}"
green "📊 Stats file for visualization: ${OUT_DIR}/stats.json"
echo ""
blue "💡 Tip: On subsequent runs, existing files will be automatically skipped if they're up-to-date"
blue "    This makes the pipeline much faster when iterating on analysis or evaluation steps."
