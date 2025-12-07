#!/usr/bin/env bash
set -euo pipefail

# Download official QASPER splits (train/dev/test) and extract JSON files

DATA_DIR="data/qasper"
TRAIN_DEV_URL="https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
TEST_URL="https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

DEV_JSON="${DATA_DIR}/qasper-dev-v0.3.json"
TRAIN_JSON="${DATA_DIR}/qasper-train-v0.3.json"
TEST_JSON="${DATA_DIR}/qasper-test-v0.3.json"

download_and_extract() {
    local url="$1"; shift
    local target_dir="$1"; shift
    local archive_name
    archive_name="${target_dir}/$(basename "${url}")"

    echo "Downloading ${url} → ${archive_name}";
    if command -v curl >/dev/null 2>&1; then
        curl -L "${url}" -o "${archive_name}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${archive_name}" "${url}"
    else
        echo "❌ Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    echo "Extracting $(basename "${archive_name}")"
    tar -xzf "${archive_name}" -C "${target_dir}"
    rm -f "${archive_name}"
}

echo "📥 Preparing QASPER dataset (official v0.3 splits)..."
mkdir -p "${DATA_DIR}"

if [[ ! -f "${DEV_JSON}" ]] || [[ ! -f "${TRAIN_JSON}" ]]; then
    download_and_extract "${TRAIN_DEV_URL}" "${DATA_DIR}"
else
    echo "✅ Train/dev JSON files already present."
fi

if [[ ! -f "${TEST_JSON}" ]]; then
    download_and_extract "${TEST_URL}" "${DATA_DIR}"
else
    echo "✅ Test JSON file already present."
fi

echo ""
echo "📊 Dataset info:"
if command -v jq >/dev/null 2>&1; then
    if [[ -f "${DEV_JSON}" ]]; then
        num_papers=$(jq 'length' "${DEV_JSON}")
        echo "  Dev papers: ${num_papers}"
    fi
    if [[ -f "${TRAIN_JSON}" ]]; then
        num_papers=$(jq 'length' "${TRAIN_JSON}")
        echo "  Train papers: ${num_papers}"
    fi
    if [[ -f "${TEST_JSON}" ]]; then
        num_papers=$(jq 'length' "${TEST_JSON}")
        echo "  Test papers: ${num_papers}"
    fi
else
    echo "  Install 'jq' to print dataset statistics."
fi

echo ""
echo "✨ QASPER dataset (train/dev/test) is ready in ${DATA_DIR}"
echo "You can now run the QASPER pipeline from the QT app."
