"""
Pure data model for Anxin's internal case state.

Updated exclusively by StateManageAgent via tool calls — no keyword matching.
The state is a structured memory that the advisor builds from dialogue.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PartyInfo:
    name: str            # short name: "宏基", "恒达"
    role: str            # "总包单位", "分包单位", "包工头", "监察员", "项目经理", "工友"
    full_name: str = ""  # "宏基建设集团股份有限公司"


@dataclass
class EvidenceRecord:
    evidence_id: str  # E001
    name: str = ""    # 手写工资结算单
    proves: str = ""  # 欠薪金额


@dataclass
class AnxinInternalState:
    """
    Anxin's structured memory of the case, built entirely from dialogue
    via StateManageAgent tool calls.
    """
    # Parties
    parties: dict[str, PartyInfo] = field(default_factory=dict)

    # Evidence
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)

    # Procedural milestones
    milestones: dict[str, bool] = field(default_factory=lambda: {
        "evidence_organized": False,
        "queried_company_info": False,
        "has_legal_aid": False,
        "complained_to_inspection": False,
        "limit_order_issued": False,
        "arbitration_filed": False,
        "asset_preservation_applied": False,
        "criminal_report_filed": False,
        "contacted_witness": False,
    })

    # Action history
    actions_done: list[str] = field(default_factory=list)
    turn_count: int = 0

    # --- Tool-callable update methods ---

    def add_party(self, name: str, role: str, full_name: str = "") -> dict:
        key = name
        self.parties[key] = PartyInfo(name=name, role=role, full_name=full_name)
        return {"status": "ok", "party": name, "role": role}

    def add_evidence(self, evidence_id: str, name: str = "", proves: str = "") -> dict:
        self.evidence[evidence_id] = EvidenceRecord(
            evidence_id=evidence_id, name=name, proves=proves,
        )
        return {"status": "ok", "evidence": evidence_id}

    def record_action_done(self, action_id: str, description: str = "") -> dict:
        self.turn_count += 1
        if action_id not in self.actions_done:
            self.actions_done.append(action_id)
        return {"status": "ok", "action": action_id, "turn": self.turn_count}

    def set_milestone(self, milestone: str, details: str = "") -> dict:
        self.milestones[milestone] = True
        return {"status": "ok", "milestone": milestone}

    # --- Derived properties ---

    def get_total_contractor(self) -> str:
        for p in self.parties.values():
            if p.role == "总包单位" and p.full_name:
                return p.full_name
        for p in self.parties.values():
            if p.role == "总包单位":
                return p.name
        return ""

    def get_subcontractor(self) -> str:
        for p in self.parties.values():
            if p.role == "分包单位" and p.full_name:
                return p.full_name
        for p in self.parties.values():
            if p.role == "分包单位":
                return p.name
        return ""

    def get_respondent_list(self) -> list[str]:
        respondents = []
        tc = self.get_total_contractor()
        sc = self.get_subcontractor()
        if tc:
            respondents.append(tc)
        if sc:
            respondents.append(sc)
        return respondents

    def infer_stage(self) -> str:
        m = self.milestones
        if m.get("arbitration_filed") and m.get("asset_preservation_applied"):
            return "arbitration_and_preservation_done"
        if m.get("arbitration_filed"):
            return "arbitration_filed"
        if m.get("limit_order_issued"):
            return "limit_order_issued"
        if m.get("complained_to_inspection"):
            return "inspection_started"
        if m.get("evidence_organized"):
            return "evidence_ready"
        return "initial"

    def get_state_summary_for_prompt(self) -> str:
        stage = self.infer_stage()
        parties_str = ", ".join(
            f"{p.name}({p.role})" for p in self.parties.values()
        ) or "尚未了解"
        evidence_str = ", ".join(
            f"{e.evidence_id}:{e.name}" for e in self.evidence.values()
        ) or "尚未整理"
        respondents = self.get_respondent_list()

        lines = [
            f"## 当前案件记忆（第{self.turn_count}轮，阶段：{stage}）",
            f"- 当事人：{parties_str}",
            f"- 总包：{self.get_total_contractor() or '未确认'}",
            f"- 分包：{self.get_subcontractor() or '未确认'}",
            f"- 证据：{evidence_str}",
            f"- 整改令：{'已下达' if self.milestones.get('limit_order_issued') else '未下达'}",
            f"- 已投诉监察：{'是' if self.milestones.get('complained_to_inspection') else '否'}",
            f"- 已申请仲裁：{'是' if self.milestones.get('arbitration_filed') else '否'}",
            f"- 已申请保全：{'是' if self.milestones.get('asset_preservation_applied') else '否'}",
            f"- 已刑事报案：{'是' if self.milestones.get('criminal_report_filed') else '否'}",
            f"- 有法援：{'是' if self.milestones.get('has_legal_aid') else '否'}",
            f"- 仲裁被申请人：{', '.join(respondents) if respondents else '未确定'}",
            f"- 已完成行动：{', '.join(self.actions_done) or '无'}",
        ]
        return "\n".join(lines)
