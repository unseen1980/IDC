#!/usr/bin/env python3
"""
IDC Orchestrator CLI

One place to run the pipeline with sensible defaults, tune baselines/IDC,
and generate stats with confidence intervals.

Examples:
  # Interactive menu
  python src/cli.py menu

  # Non-interactive end-to-end run on a given doc
  python src/cli.py run --doc idc --auto-tune --auto-tune-baselines \
    --eval-embedder models/text-embedding-004 --target-avg 7 --tol 1

  # Stats only
  python src/cli.py stats --doc idc --json-out out/idc/stats.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DATA_DIR = Path("data/input")
OUT_ROOT = Path("out")
SCRIPT = Path("scripts/run_idc_pipeline.sh")


def list_docs() -> List[str]:
    return [p.stem for p in sorted(DATA_DIR.glob("*.txt"))]


def ensure_exists(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")


@dataclass
class RunOptions:
    doc: str
    dim: int = 1536
    auto_tune: bool = False
    auto_tune_baselines: bool = False
    eval_embedder: Optional[str] = None
    target_avg: Optional[float] = None
    tol: Optional[float] = None
    force_spans: bool = False
    lambda_: Optional[float] = None
    boundary_penalty: Optional[float] = None
    max_len: Optional[int] = None
    min_len: Optional[int] = None
    coherence_weight: Optional[float] = None
    input_file: Optional[str] = None


def run_pipeline(opts: RunOptions) -> int:
    ensure_exists(SCRIPT)
    # Allow explicit input file override; else fall back to data/input/<doc>.txt
    if opts.input_file:
        ensure_exists(Path(opts.input_file))
    else:
        ensure_exists(DATA_DIR / f"{opts.doc}.txt")
    env = os.environ.copy()
    env.update({
        "DOC_NAME": opts.doc,
        "DIM": str(opts.dim),
        "AUTO_TUNE": "1" if opts.auto_tune else "0",
        "AUTO_TUNE_BASELINES": "1" if opts.auto_tune_baselines else "0",
        "FORCE_SPANS": "1" if opts.force_spans else "0",
    })
    if opts.input_file:
        env["INPUT_FILE"] = str(Path(opts.input_file))
    if opts.eval_embedder:
        env["EVAL_EMBEDDER"] = opts.eval_embedder
    if opts.target_avg is not None:
        env["TUNE_TARGET_AVG"] = str(opts.target_avg)
    if opts.tol is not None:
        env["TUNE_TOL"] = str(opts.tol)
    # Optional direct overrides
    if opts.lambda_ is not None:
        env["LAMBDA"] = str(opts.lambda_)
    if opts.boundary_penalty is not None:
        env["BOUNDARY_PENALTY"] = str(opts.boundary_penalty)
    if opts.max_len is not None:
        env["MAX_LEN"] = str(opts.max_len)
    if opts.min_len is not None:
        env["MIN_LEN"] = str(opts.min_len)
    if opts.coherence_weight is not None:
        env["COHERENCE_WEIGHT"] = str(opts.coherence_weight)

    print(f"→ Running pipeline for doc='{opts.doc}' (dim={opts.dim})")
    print(f"   AUTO_TUNE={env['AUTO_TUNE']} AUTO_TUNE_BASELINES={env['AUTO_TUNE_BASELINES']}")
    if opts.eval_embedder:
        print(f"   EVAL_EMBEDDER={opts.eval_embedder}")
    if opts.target_avg is not None:
        print(f"   Target avg sentences/chunk: {opts.target_avg} (tol={opts.tol})")
    rc = subprocess.call(["bash", str(SCRIPT)], env=env)
    return rc


def run_stats(doc: str, embedder: str, dim: int, json_out: Optional[Path]) -> int:
    out_dir = OUT_ROOT / doc
    cmd = [
        "python", "src/stats_summary.py",
        "--out-dir", str(out_dir),
        "--embedder", embedder,
        "--dim", str(dim),
    ]
    if json_out:
        cmd += ["--json-out", str(json_out)]
    print("→ Computing stats:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_run(args: argparse.Namespace) -> int:
    opts = RunOptions(
        doc=args.doc,
        dim=args.dim,
        auto_tune=args.auto_tune,
        auto_tune_baselines=args.auto_tune_baselines,
        eval_embedder=args.eval_embedder,
        target_avg=args.target_avg,
        tol=args.tol,
        force_spans=args.force_spans,
        lambda_=args.lam,
        boundary_penalty=args.boundary_penalty,
        max_len=args.max_len,
        min_len=args.min_len,
        coherence_weight=args.coherence_weight,
        input_file=args.input_file,
    )
    return run_pipeline(opts)


def cmd_stats(args: argparse.Namespace) -> int:
    json_out = Path(args.json_out) if args.json_out else None
    return run_stats(args.doc, args.embedder, args.dim, json_out)


def prompt_bool(q: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{q} [{d}]: ").strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}: return True
        if ans in {"n", "no"}: return False


def prompt_float(q: str, default: Optional[float]) -> Optional[float]:
    d = " (blank to skip)" if default is None else f" (default {default})"
    ans = input(f"{q}{d}: ").strip()
    if not ans:
        return default
    try:
        return float(ans)
    except ValueError:
        print("Invalid number; skipping")
        return default


def cmd_menu(_: argparse.Namespace) -> int:
    docs = list_docs()
    if not docs:
        print(f"No .txt files in {DATA_DIR}. Add documents first.")
        return 1
    print("Available documents:")
    for i, d in enumerate(docs, start=1):
        print(f"  {i}. {d}")
    while True:
        s = input("Select a document by number: ").strip()
        try:
            k = int(s)
            if 1 <= k <= len(docs):
                doc = docs[k - 1]
                break
        except ValueError:
            pass
        print("Please enter a valid number.")

    dim = 1536
    sdim = input("Vector dim [1536]: ").strip()
    if sdim:
        try:
            dim = int(sdim)
        except ValueError:
            print("Invalid dim; using 1536")
            dim = 1536

    auto_idc = prompt_bool("Auto-tune IDC?", True)
    auto_base = prompt_bool("Auto-tune baselines?", True)
    use_alt = prompt_bool("Use alternate embedder for pseudo spans?", True)
    eval_emb = None
    if use_alt:
        eval_emb = input("Eval embedder (e.g., models/text-embedding-004): ").strip() or None
    target_avg = prompt_float("Target avg sentences/chunk", 7.0)
    tol = prompt_float("Tolerance (sentences)", 1.0)

    opts = RunOptions(
        doc=doc,
        dim=dim,
        auto_tune=auto_idc,
        auto_tune_baselines=auto_base,
        eval_embedder=eval_emb,
        target_avg=target_avg,
        tol=tol,
        force_spans=True,
    )
    rc = run_pipeline(opts)
    if rc != 0:
        return rc
    if prompt_bool("Compute stats with CIs now?", True):
        return run_stats(doc, embedder="gemini-embedding-001", dim=dim, json_out=OUT_ROOT / doc / "stats.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="IDC Orchestrator CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run pipeline end-to-end with options")
    p_run.add_argument("--doc", required=True, help="Document name (stem of data/input/<doc>.txt)")
    p_run.add_argument("--input-file", help="Optional explicit path to a .txt file; overrides data/input/<doc>.txt")
    p_run.add_argument("--dim", type=int, default=1536)
    p_run.add_argument("--auto-tune", action="store_true")
    p_run.add_argument("--auto-tune-baselines", action="store_true")
    p_run.add_argument("--eval-embedder")
    p_run.add_argument("--target-avg", type=float)
    p_run.add_argument("--tol", type=float)
    p_run.add_argument("--force-spans", action="store_true")
    # Optional direct IDC overrides
    p_run.add_argument("--lam", type=float)
    p_run.add_argument("--boundary-penalty", type=float)
    p_run.add_argument("--max-len", type=int)
    p_run.add_argument("--min-len", type=int)
    p_run.add_argument("--coherence-weight", type=float)
    p_run.set_defaults(func=cmd_run)

    p_stats = sub.add_parser("stats", help="Compute summary stats with CIs")
    p_stats.add_argument("--doc", required=True)
    p_stats.add_argument("--embedder", default="gemini-embedding-001")
    p_stats.add_argument("--dim", type=int, default=1536)
    p_stats.add_argument("--json-out")
    p_stats.set_defaults(func=cmd_stats)

    p_menu = sub.add_parser("menu", help="Interactive menu to run pipeline and stats")
    p_menu.set_defaults(func=cmd_menu)

    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
