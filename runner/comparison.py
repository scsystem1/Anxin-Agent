"""
Comparison runner.

Runs the same case twice — once with Anxin, once with Doubao — and
produces a side-by-side comparison report. This is what gets shown in
the demo.

Important: each run gets its OWN fresh Environment instance, so the two
runs are independent. The case_data is the same (deterministic), but
LLM calls have inherent stochasticity, so set temperature carefully if
you want repeatability.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json

from environment.env import Environment
from worker.simulated_worker import SimulatedWorker
from advisor.anxin_advisor import AnxinAdvisor
from advisor.doubao_advisor import DoubaoAdvisor
from runner.episode import EpisodeRunner, EpisodeResult
from judge.judgment_engine import judgment_to_markdown


@dataclass
class ComparisonReport:
    case_name: str
    anxin_result: EpisodeResult
    doubao_result: EpisodeResult


def run_comparison(
    case_path: str,
    max_turns: int = 30,
    verbose: bool = True,
) -> ComparisonReport:
    """Run Anxin and Doubao on the same case, return both results."""

    # ---- run 1: Anxin ----
    if verbose:
        print("\n" + "=" * 60)
        print("█  RUNNING ANXIN  █")
        print("=" * 60)
    env_a = Environment.from_case_file(case_path)
    worker_a = SimulatedWorker()
    anxin = AnxinAdvisor()
    anxin_result = EpisodeRunner(env_a, worker_a, anxin, max_turns, verbose).run()

    # ---- run 2: Doubao ----
    if verbose:
        print("\n" + "=" * 60)
        print("█  RUNNING DOUBAO  █")
        print("=" * 60)
    env_d = Environment.from_case_file(case_path)
    worker_d = SimulatedWorker()
    doubao = DoubaoAdvisor()
    doubao_result = EpisodeRunner(env_d, worker_d, doubao, max_turns, verbose).run()

    case_name = env_a.case_data.get("case_name", "unnamed_case")
    return ComparisonReport(
        case_name=case_name,
        anxin_result=anxin_result,
        doubao_result=doubao_result,
    )


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Produce a side-by-side markdown report suitable for the demo."""
    lines = []
    lines.append(f"# 安薪 vs 豆包：对比报告 — {report.case_name}\n")

    # ---- summary table ----
    a, d = report.anxin_result, report.doubao_result
    aj = a.final_judgment
    dj = d.final_judgment
    lines.append("## 关键指标对比\n")
    lines.append("| 指标 | Anxin | Doubao |")
    lines.append("|------|-------|--------|")
    lines.append(f"| 终止原因 | {a.terminal_reason} | {d.terminal_reason} |")
    lines.append(f"| 用时（天）| {a.total_days} | {d.total_days} |")
    lines.append(f"| 对话轮数 | {a.total_turns} | {d.total_turns} |")
    if aj and dj:
        lines.append(f"| 主要被告 | {aj.primary_respondent or '—'} | {dj.primary_respondent or '—'} |")
        lines.append(f"| 判决金额（元）| {aj.monetary_award.total} | {dj.monetary_award.total} |")
        lines.append(f"| 关键失误数 | {len(aj.critical_misses)} | {len(dj.critical_misses)} |")
        lines.append(f"| 已采证据数 | {len(aj.evidence_used)} | {len(dj.evidence_used)} |")
    lines.append("")

    # ---- procedural paths ----
    lines.append("## 走过的路径")
    lines.append("\n### Anxin")
    for t in a.transcript:
        params = (
            "(" + ", ".join(f"{k}={v}" for k, v in t.chosen_action_params.items()) + ")"
            if t.chosen_action_params else ""
        )
        lines.append(f"- 第{t.day}天 [{t.chosen_action_id}{params}] {t.action_narration[:80]}")
    lines.append("\n### Doubao")
    for t in d.transcript:
        params = (
            "(" + ", ".join(f"{k}={v}" for k, v in t.chosen_action_params.items()) + ")"
            if t.chosen_action_params else ""
        )
        lines.append(f"- 第{t.day}天 [{t.chosen_action_id}{params}] {t.action_narration[:80]}")
    lines.append("")

    # ---- judgments ----
    if aj:
        lines.append("\n---\n")
        lines.append(judgment_to_markdown(aj))
    if dj:
        lines.append("\n---\n")
        lines.append(judgment_to_markdown(dj))

    return "\n".join(lines)


def save_comparison_artifacts(report: ComparisonReport, out_dir: str = "./out") -> None:
    """Dump all relevant artifacts to disk for the demo."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    # markdown report
    md = render_comparison_markdown(report)
    with open(f"{out_dir}/comparison_report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # raw transcripts as JSON
    def _serialize_episode(r: EpisodeResult) -> dict:
        from dataclasses import asdict
        return {
            "advisor_name": r.advisor_name,
            "total_turns": r.total_turns,
            "total_days": r.total_days,
            "terminal_reason": r.terminal_reason,
            "transcript": [asdict(t) for t in r.transcript],
            "final_judgment": asdict(r.final_judgment) if r.final_judgment else None,
        }

    with open(f"{out_dir}/anxin_run.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_episode(report.anxin_result), f, ensure_ascii=False, indent=2)
    with open(f"{out_dir}/doubao_run.json", "w", encoding="utf-8") as f:
        json.dump(_serialize_episode(report.doubao_result), f, ensure_ascii=False, indent=2)
    print(f"\n✓ Artifacts saved to {out_dir}/")
