"""
Judgment engine.

When an episode ends (terminal state reached), the judgment engine produces
a structured "judgment book" by feeding the full case state into an LLM
prompted as a labor law judge.

The judgment is the FINAL DELIVERABLE of the simulation — it's what gets
shown side-by-side against the other advisor's run.

DESIGN PRINCIPLES
-----------------
1. The judge has full visibility of the ground truth, the evidence pool,
   the actions taken, and the NPC interactions. It is NOT a blind judge —
   it knows what really happened.
2. But the judge writes the verdict based ONLY on what was procedurally
   admitted (i.e. evidence in state.evidence_pool, defenses raised by
   respondents). This mimics how real adjudication works: facts not
   introduced by the parties don't get into the verdict.
3. The output is a structured object so we can compare two runs
   field-by-field, not just diff two paragraphs of prose.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Any

from llm.client import LLMClient
from environment.state import CaseState, TerminalReason


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class FactFinding:
    """A single fact the court found established."""
    fact: str
    supporting_evidence_ids: list[str]
    found_established: bool
    reasoning: str


@dataclass
class LiabilityFinding:
    """The court's determination of one party's liability."""
    party: str                                # e.g. "宏基建设集团"
    role: str                                 # e.g. "施工总承包"
    liable: bool
    liability_type: str                       # e.g. "先行清偿" / "连带" / "直接" / "无"
    legal_basis: list[str]                    # e.g. ["《保障农民工工资支付条例》第30条"]
    reasoning: str


@dataclass
class JudgmentMonetaryAward:
    principal: int = 0                        # 欠薪本金
    additional_compensation: int = 0          # 加付赔偿金
    interest: int = 0
    legal_costs: int = 0
    total: int = 0


@dataclass
class Judgment:
    """The final structured verdict for an episode."""
    case_id: str
    advisor_name: str                         # "anxin" or "doubao"
    terminal_reason: str                      # from TerminalReason
    days_elapsed: int

    # the case as it was actually adjudicated
    procedural_stage_at_end: str
    primary_respondent: str | None            # e.g. "宏基建设集团"; None if abandoned
    fact_findings: list[FactFinding] = field(default_factory=list)
    liability_findings: list[LiabilityFinding] = field(default_factory=list)
    monetary_award: JudgmentMonetaryAward = field(default_factory=JudgmentMonetaryAward)

    # narrative
    summary_in_plain_chinese: str = ""        # 1-2 paragraph summary for the user
    formal_judgment_text: str = ""            # 仿判决书 formal text

    # for diagnostic comparison
    procedural_path_taken: list[str] = field(default_factory=list)
    # ↑ ["A001", "A011 → wang_pei推诿", "A006(target=李大海) 错对象", ...]
    evidence_used: list[str] = field(default_factory=list)
    critical_misses: list[str] = field(default_factory=list)
    # ↑ things the agent should have done but didn't, e.g.
    #   "未向劳动监察申请调取实名台账（错过 E006）"
    #   "未将宏基列为先行清偿被申请人"

    # raw judge output for audit
    judge_raw_output: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
你是一位资深劳动法领域的法官，长期审理建筑业农民工欠薪案件。
你将对一个【模拟案件】的最终结果作出裁判，并出具一份结构化的判决意见。

裁判原则（请严格遵守）：
1. 你拥有上帝视角（知道完整的Ground Truth），但你的判决必须基于本案
   procedural record（证据池、当事人提出的抗辩、走过的程序）。
2. 如果原告未将某主体列为被告/被申请人，即使事实上该主体应当承担责任，
   你也不能在判决中令其承担——但你可以在 critical_misses 中明确指出。
3. 严格区分先行清偿责任（《保障农民工工资支付条例》第30条）和连带责任
   （第31条）。前者只追总包，后者追违法分包方。
4. 加付赔偿金（《劳动合同法》第85条）只在劳动监察下达限期整改令、相对方
   逾期不支付的情况下触发。
5. 对放弃维权（A099）、超时（>365天）、走错程序的情况，分别给出对应的
   裁判结论。

输出格式：严格 JSON，schema 见下文。不要有 markdown 代码块。
"""


JUDGE_OUTPUT_SCHEMA = """\
{
  "primary_respondent": "...",
  "fact_findings": [
    {"fact": "...", "supporting_evidence_ids": ["E001","E006"],
     "found_established": true, "reasoning": "..."}
  ],
  "liability_findings": [
    {"party": "宏基建设集团", "role": "施工总承包",
     "liable": true, "liability_type": "先行清偿",
     "legal_basis": ["《保障农民工工资支付条例》第30条"],
     "reasoning": "..."}
  ],
  "monetary_award": {
    "principal": 76600, "additional_compensation": 38300,
    "interest": 4592, "legal_costs": 0, "total": 119492
  },
  "summary_in_plain_chinese": "1-2段，给赵建国能看懂的版本",
  "formal_judgment_text": "完整仿判决书正文，包含'本院查明''本院认为''判决如下'三段",
  "critical_misses": [
    "未向劳动监察申请调取实名台账（错过E006）",
    "未将宏基列为先行清偿被申请人"
  ]
}
"""


def _build_judge_user_prompt(state: CaseState, case_data: dict, advisor_name: str) -> str:
    """Assemble the user prompt with all the facts the judge needs."""
    gt = case_data["ground_truth"]
    fin = state.financial

    actions_log = "\n".join(
        f"  第{a.day}天 [{a.action_id}] {a.action_name}"
        f"{('参数=' + json.dumps(a.parameters, ensure_ascii=False)) if a.parameters else ''}"
        f" → {a.narration}"
        for a in state.actions_taken
    )
    npc_log = "\n".join(
        f"  第{n.day}天 [{n.npc_id}] 工人说：{n.worker_message[:60]}…\n"
        f"      → 对方：{n.npc_response[:120]}…"
        for n in state.npc_interactions
    )
    evidence_summary = state.evidence_summary()
    defenses_summary = "\n".join(
        f"  - {party}：{', '.join(defs)}"
        for party, defs in state.respondent_defenses.items()
    ) or "  （无对方明确抗辩记录）"

    return f"""\
# 案件基本信息
案号：{state.case_id}
原告：{state.worker_name}（{state.worker_id_card}）
本次模拟由 advisor = "{advisor_name}" 担任军师指导。
模拟运行天数：{state.current_day} 天
程序终止阶段：{state.procedural_stage.value}
终止原因：{state.terminal_reason.value if state.terminal_reason else '未终止'}

# Ground Truth（你作为法官知道的客观真相）
- 总包：{gt['general_contractor']['name']}（依法负有先行清偿责任）
- 分包：{gt['subcontractor']['name']}（违法分包，应承担连带责任。已于2023-12-05向李大海支付全部劳务款 1,427,000 元）
- 包工头：{gt['labor_contractor']['name']}（已收款后逃匿）
- 欠薪本金：{fin.total_owed} 元
- 工人持有手写结算单（E001），是核心证据

# 程序记录（仅以下进入了诉讼/裁决记录）
## 已固定的证据池
{evidence_summary}

## 已采取的行动序列
{actions_log if actions_log else '  （无）'}

## 与各方的交涉记录
{npc_log if npc_log else '  （无）'}

## 对方提出的抗辩
{defenses_summary}

# 程序性事实（影响裁判要素）
- 是否申请劳动监察并下达限期整改令：{state.flags.get('limit_order_issued', False)}
- 整改期满是否仍未支付：{state.flags.get('limit_order_expired_unpaid', False)}
- 是否申请财产保全：{state.flags.get('asset_freeze_active', False)}
- 是否有法援律师：{state.flags.get('has_legal_aid_lawyer', False)}
- 是否提起刑事报案（拒不支付劳动报酬罪）：{state.flags.get('criminal_case_filed', False)}
- 已实际追回金额：{fin.principal_recovered} 元

# 你的任务
按下面的 JSON schema 输出判决意见。要求：
1. fact_findings 必须严格基于上面"已固定的证据池"和"行动序列"——不在记录里
   的事实不能直接采信（即便Ground Truth有）。
2. liability_findings 必须严格限制在原告实际列为被告/被申请人的范围内。
   如果工人只走 A011（找王主任谈判）这种非正式接触，不构成对恒达起诉。
3. 如果终止原因是 abandoned/timeout，仍要出具一份判决意见，但 monetary_award
   全部为 0，并在 critical_misses 中说明本应能拿到多少。
4. critical_misses 是给项目方看的复盘，要具体到行动 + 后果。

# 输出 schema
{JUDGE_OUTPUT_SCHEMA}
"""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class JudgmentEngine:
    """LLM-backed adjudicator."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient.from_env(role="judge")

    def adjudicate(
        self,
        state: CaseState,
        case_data: dict,
        advisor_name: str = "unknown",
    ) -> Judgment:
        """Produce a Judgment from the terminal state."""
        user_prompt = _build_judge_user_prompt(state, case_data, advisor_name)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Codex: in case the judge LLM doesn't reliably emit JSON, wrap in
        # try/except and fall back to a minimal Judgment with the error.
        try:
            parsed = self.llm.chat_json(
                messages=messages,
                temperature=0.2,
                purpose="judge",
            )
        except Exception as e:
            return Judgment(
                case_id=state.case_id,
                advisor_name=advisor_name,
                terminal_reason=(state.terminal_reason.value if state.terminal_reason else "unknown"),
                days_elapsed=state.current_day,
                procedural_stage_at_end=state.procedural_stage.value,
                primary_respondent=None,
                summary_in_plain_chinese=f"（判决引擎调用失败：{e}）",
                formal_judgment_text="",
                judge_raw_output=str(e),
            )

        # Hydrate the structured object
        award_data = parsed.get("monetary_award", {}) or {}
        award = JudgmentMonetaryAward(
            principal=int(award_data.get("principal", 0)),
            additional_compensation=int(award_data.get("additional_compensation", 0)),
            interest=int(award_data.get("interest", 0)),
            legal_costs=int(award_data.get("legal_costs", 0)),
            total=int(award_data.get("total", 0)),
        )

        return Judgment(
            case_id=state.case_id,
            advisor_name=advisor_name,
            terminal_reason=(state.terminal_reason.value if state.terminal_reason else "unknown"),
            days_elapsed=state.current_day,
            procedural_stage_at_end=state.procedural_stage.value,
            primary_respondent=parsed.get("primary_respondent"),
            fact_findings=[FactFinding(**ff) for ff in parsed.get("fact_findings", [])],
            liability_findings=[LiabilityFinding(**lf) for lf in parsed.get("liability_findings", [])],
            monetary_award=award,
            summary_in_plain_chinese=parsed.get("summary_in_plain_chinese", ""),
            formal_judgment_text=parsed.get("formal_judgment_text", ""),
            procedural_path_taken=[
                f"{a.action_id}({a.parameters})" if a.parameters else a.action_id
                for a in state.actions_taken
            ],
            evidence_used=list(state.evidence_pool.keys()),
            critical_misses=parsed.get("critical_misses", []),
            judge_raw_output=json.dumps(parsed, ensure_ascii=False, indent=2),
        )


def judgment_to_markdown(j: Judgment) -> str:
    """Render a Judgment as a human-readable markdown report."""
    lines = []
    lines.append(f"# 判决报告 — {j.advisor_name.upper()} 路径")
    lines.append("")
    lines.append(f"- 案号：{j.case_id}")
    lines.append(f"- 终止原因：{j.terminal_reason}")
    lines.append(f"- 用时：{j.days_elapsed} 天")
    lines.append(f"- 程序终止阶段：{j.procedural_stage_at_end}")
    lines.append(f"- 主要被告：{j.primary_respondent or '（未起诉任何主体）'}")
    lines.append("")
    lines.append("## 金额裁判")
    lines.append(f"- 欠薪本金：{j.monetary_award.principal} 元")
    lines.append(f"- 加付赔偿金：{j.monetary_award.additional_compensation} 元")
    lines.append(f"- 利息：{j.monetary_award.interest} 元")
    lines.append(f"- 合计：{j.monetary_award.total} 元")
    lines.append("")
    lines.append("## 责任认定")
    for lf in j.liability_findings:
        lines.append(f"- **{lf.party}** ({lf.role}) — {'承担' if lf.liable else '不承担'} {lf.liability_type}责任")
        lines.append(f"  - 法律依据：{', '.join(lf.legal_basis)}")
        lines.append(f"  - 理由：{lf.reasoning}")
    lines.append("")
    lines.append("## 关键失误（critical misses）")
    if j.critical_misses:
        for m in j.critical_misses:
            lines.append(f"- {m}")
    else:
        lines.append("（无）")
    lines.append("")
    lines.append("## 通俗版总结")
    lines.append(j.summary_in_plain_chinese)
    lines.append("")
    lines.append("## 仿判决书正文")
    lines.append(j.formal_judgment_text)
    return "\n".join(lines)
