# IDC Documentation Index

Documentation supporting the Intent-Driven Dynamic Chunking (IDC) research. These documents contain methodology, results, and analysis details for the thesis and paper.

---

## Algorithm and Design

- **[IDC_ALGORITHM.md](./IDC_ALGORITHM.md)** -- Core algorithm description with DP objective function, Mermaid diagrams, and baseline comparisons
- **[IMPLEMENTATION_DETAILS.md](./IMPLEMENTATION_DETAILS.md)** -- Implementation architecture, Python modules, embedding models, retrieval pipeline, and computational performance
- **[AUTO_ADAPTIVE_INTENTS.md](./AUTO_ADAPTIVE_INTENTS.md)** -- Auto-scaling intent generation for long documents

---

## Results and Evaluation

- **[CHAPTER_5_RESULTS.md](./CHAPTER_5_RESULTS.md)** -- Complete results chapter with 15 publication-quality tables
- **[THESIS_READY_RESULTS.md](./THESIS_READY_RESULTS.md)** -- Publication-quality result tables and summary

---

## Methodology Decisions

- **[DATASET_SELECTION_RATIONALE.md](./DATASET_SELECTION_RATIONALE.md)** -- Dataset diversity justification and selection criteria
- **[EXPERIMENTAL_CONFIGURATION.md](./EXPERIMENTAL_CONFIGURATION.md)** -- Complete parameter settings and auto-tuning grid search space
- **[EVALUATION_CONCEPTS.md](./EVALUATION_CONCEPTS.md)** -- Metrics explanation (R@1, R@5, MRR, Coverage)

---

## Quick Reference: Key Results

| Dataset | Domain | IDC R@1 | Best Baseline | Improvement |
|---------|--------|---------|---------------|-------------|
| SQuAD | Wikipedia | **0.689** | Fixed (0.655) | +5.2% |
| NewsQA | News | **0.933** | Coherence (0.867) | +7.6% |
| arXiv | Academic | **0.667** | Fixed (0.400) | +67% |
| Fiori | Technical | **0.533** | Fixed (0.333) | +60% |
