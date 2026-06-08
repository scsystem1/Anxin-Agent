"""
Main entry point for the Anxin sandbox.

Usage:
    python run.py                          # run both advisors on default case
    python run.py --advisor anxin          # run only Anxin
    python run.py --advisor doubao         # run only Doubao
    python run.py --case cases/<other>.json
    python run.py --max-turns 20
    python run.py --quiet                  # less verbose

Once the LLM API is configured in .env, this should run end-to-end.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

# Ensure stdout can handle Unicode (e.g. ⚠) on Windows GBK consoles.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import get_run_config
from environment.env import Environment
from worker.simulated_worker import SimulatedWorker
from advisor.anxin_advisor import AnxinAdvisor
from advisor.doubao_advisor import DoubaoAdvisor
from runner.episode import EpisodeRunner
from runner.comparison import (
    run_comparison,
    render_comparison_markdown,
    save_comparison_artifacts,
)
from judge.judgment_engine import judgment_to_markdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anxin sandbox runner")
    p.add_argument("--advisor", choices=["anxin", "doubao", "both"], default="both")
    p.add_argument("--case", default=None, help="Path to case JSON")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_run_config()

    case_path = args.case or cfg.case_path
    max_turns = args.max_turns or cfg.max_turns
    verbose = not args.quiet
    out_dir = args.out_dir or cfg.out_dir

    print(f"Case:      {case_path}")
    print(f"Max turns: {max_turns}")
    print(f"Advisor:   {args.advisor}")
    print(f"Out dir:   {out_dir}")
    print()

    if args.advisor == "both":
        report = run_comparison(case_path, max_turns=max_turns, verbose=verbose)
        save_comparison_artifacts(report, out_dir=out_dir)
        # also print summary to stdout
        print("\n" + "=" * 60)
        print("█ FINAL COMPARISON SUMMARY █")
        print("=" * 60)
        print(render_comparison_markdown(report))
        return 0

    # single-advisor mode (for debugging / quick iteration)
    env = Environment.from_case_file(case_path)
    worker = SimulatedWorker()
    advisor = AnxinAdvisor() if args.advisor == "anxin" else DoubaoAdvisor()
    result = EpisodeRunner(env, worker, advisor, max_turns=max_turns, verbose=verbose).run()

    # Save result to JSON for later comparison
    out_path = Path(out_dir) / f"{args.advisor}_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use the same serialization format as save_comparison_artifacts
    result_dict = {
        "advisor_name": result.advisor_name,
        "total_turns": result.total_turns,
        "total_days": result.total_days,
        "terminal_reason": result.terminal_reason,
        "transcript": [asdict(t) for t in result.transcript],
        "final_judgment": asdict(result.final_judgment) if result.final_judgment else None,
    }
    out_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 60)
    print(f"█ {args.advisor.upper()} RESULT █")
    print("=" * 60)
    if result.final_judgment:
        print(judgment_to_markdown(result.final_judgment))
    return 0


if __name__ == "__main__":
    sys.exit(main())
