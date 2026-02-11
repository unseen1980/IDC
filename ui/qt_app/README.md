# IDC Benchmark UI

A PySide6 (Qt for Python) desktop application for running and visualising IDC benchmarks interactively.

## Features

- **Dataset selection** -- Choose from all paper datasets: SQuAD, NewsQA, arXiv, Fiori, QASPER
- **Paper defaults** -- Lock IDC parameters to the exact values from `configs/paper_experiments.json`
- **Hyperparameter controls** -- Adjust lambda, max/min length, boundary penalty, coherence weight, and advanced options (contextual embeddings, density-aware segmentation, auto-adaptive intents)
- **Baseline variants** -- Toggle IDC, Fixed, Sliding, Coherence, and Paragraphs baselines
- **Live logging** -- Real-time stdout/stderr from pipeline scripts
- **Results visualisation** -- Bar charts for R@1, R@5, MRR, Coverage, Completeness, Diversity, Redundancy, Efficiency, Avg Length, and Avg Tokens
- **Data table** -- Colour-coded detailed metrics table
- **CSV export** -- Export results for further analysis

## Requirements

- Python 3.10+
- PySide6 (`pip install PySide6`)
- A valid `GEMINI_API_KEY` in your environment or `.env` at the repo root

## Usage

From the repository root:

```bash
python ui/qt_app/main.py
```

## How It Works

1. Select a dataset from the dropdown
2. Check "Paper Defaults" to use the exact parameters from the paper (recommended for reproducibility)
3. Choose which variants to evaluate (IDC, Fixed, Sliding, Coherence, Paragraphs)
4. Click **Run** to execute the pipeline
5. Results load automatically on completion; you can also load any `stats.json` via **File > Load stats.json**

## Architecture

The app wraps the repository's shell scripts (`scripts/run_idc_pipeline.sh`, `scripts/run_squad2_e2e.sh`, etc.) using `QProcess`, passing IDC parameters as environment variables. Results are read from `out/<dataset>/stats.json`.

## Validated Datasets

| Dataset | Gold Spans | Script |
|---------|------------|--------|
| SQuAD 2.0 | 293 | `run_squad2_e2e.sh` |
| SQuAD 1-doc (Normans) | 12 | `run_idc_pipeline.sh` |
| NewsQA corpus | 15 | `run_idc_pipeline.sh` |
| arXiv (BERT/Finance) | 15 | `run_idc_pipeline.sh` |
| Fiori Tools | 15 | `run_idc_pipeline.sh` |
| QASPER | 10 | `run_qasper_e2e.sh` |
