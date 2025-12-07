Qt UI for IDC Benchmarking

This Qt for Python (PySide6) desktop app lets you:
- Choose a dataset (SQuAD, NewsQA, arXiv, Fiori)
- Configure core IDC hyperparameters (λ, max/min sentences, boundary penalty, coherence) and choose variants
- Start/stop evaluations using the repository's end-to-end scripts
- Load and visualize results (R@1/R@5/MRR/Coverage/Avg len) as charts
- Auto-configure optimal settings per dataset

Requirements
- Python 3.10+
- PySide6: pip install PySide6
- A valid GEMINI_API_KEY in your environment or .env at the repo root

Run
- From the repo root:
  - python ui/qt_app/main.py

Notes
- The app wraps scripts/run_idc_pipeline.sh using QProcess, piping stdout/stderr to the UI.
- Results are auto-loaded on completion. You can also load any stats.json via the UI.
- Four validated datasets supported: SQuAD (n=293), NewsQA corpus (n=15), arXiv (n=15), Fiori (n=15)
