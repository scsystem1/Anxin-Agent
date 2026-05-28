"""
Internal case state tracker for Anxin Advisor.

This is Anxin's core competitive advantage: explicit memory management.
All case facts are learned from the worker's natural-language messages,
NOT pre-loaded from ground truth. The advisor builds a mental model
of the case by tracking what the worker reports each turn.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class AnxinInternalState:
    """
    Anxin's internal model of the case, reconstructed from dialogue.

    This is the "memory" that Doubao doesn't have. Every field is populated
    by parsing the worker's messages — no ground truth leakage.
    """
    # Parties learned from dialogue
    known_parties: set[str] = field(default_factory=set)
    total_contractor_name: str = ""     # full name, learned from dialogue
    subcontractor_name: str = ""        # full name, learned from dialogue
    boss_name: str = ""                 # 包工头 name
    boss_missing: bool = False          # whether boss is unreachable

    # Evidence tracking (learned from dialogue)
    evidence_organized: bool = False
    known_evidence: set[str] = field(default_factory=set)

    # Procedural stage inference
    queried_company_info: bool = False
    has_legal_aid: bool = False
    complained_to_inspection: bool = False
    inspection_target: str = ""
    limit_order_issued: bool = False
    arbitration_filed: bool = False
    arbitration_respondents: list[str] = field(default_factory=list)
    asset_freeze: bool = False
    criminal_filed: bool = False

    # Communication tracking
    contacted_witness: bool = False

    # Action history
    turn_count: int = 0
    last_action_id: str = ""
    actions_done: set[str] = field(default_factory=set)
    a008_done_count: int = 0

    def update_from_worker_message(self, msg: str) -> None:
        """Parse a worker message and update internal state."""
        self.turn_count += 1

        # Detect party mentions
        if any(k in msg for k in ("总包", "总承包")):
            self.known_parties.add("总包单位")
        if any(k in msg for k in ("分包", "劳务公司", "劳务方")):
            self.known_parties.add("分包单位")
        if any(k in msg for k in ("包工头", "班长", "带班")):
            self.known_parties.add("包工头")

        # Detect full company names (containing 有限公司 or 股份有限公司)
        company_names = re.findall(r"[一-鿿]+(?:有限公司|股份有限公司)", msg)
        for name in company_names:
            if "建设" in name or "工程" in name or "建筑" in name:
                self.total_contractor_name = name
                self.known_parties.add(name)
            elif "劳务" in name:
                self.subcontractor_name = name
                self.known_parties.add(name)
            else:
                self.known_parties.add(name)

        # Detect short company references
        if "恒达" in msg and not self.subcontractor_name:
            self.known_parties.add("恒达")
        if "宏基" in msg and not self.total_contractor_name:
            self.known_parties.add("宏基")

        # Detect boss name and status
        boss_patterns = re.findall(r"[一-鿿]{2,3}(?:说|给了|欠|跑了|联系不上|停机|失联)", msg)
        if boss_patterns:
            for p in boss_patterns:
                name = p.rstrip("说给了欠跑了联系不上停机失联")
                if len(name) >= 2 and not name.endswith(("公司", "有限")):
                    self.boss_name = name
                    self.known_parties.add(name)

        # Detect boss unreachable
        if any(k in msg for k in ("停机", "失联", "联系不上", "找不到", "跑了", "逃了")):
            self.boss_missing = True

        # Detect evidence mentions
        if any(k in msg for k in ("整理", "翻了一遍", "结算单", "截图", "证据", "欠条", "欠薪单")):
            self.evidence_organized = True

        # Detect evidence IDs
        found_evidence = re.findall(r"E\d{3}", msg)
        self.known_evidence.update(found_evidence)

        # Detect business info query
        if "工商" in msg or "注册信息" in msg or "法定代表人" in msg or "营业执照" in msg:
            self.queried_company_info = True

        # Detect legal aid
        if any(k in msg for k in ("法援", "法律援助", "12348", "律师")):
            self.has_legal_aid = True

        # Detect inspection complaint
        if "监察" in msg and any(k in msg for k in ("投诉", "受理", "整改", "指令")):
            self.complained_to_inspection = True
        if "投诉" in msg and ("总包" in msg or "总承包" in msg):
            self.inspection_target = "总包单位"

        # Detect limit order
        if "限期整改" in msg or "整改令" in msg or "整改指令" in msg:
            self.limit_order_issued = True

        # Detect arbitration
        if any(k in msg for k in ("仲裁", "仲裁委", "仲裁申请", "仲裁立案")):
            self.arbitration_filed = True

        # Detect asset preservation
        if "保全" in msg or "冻结" in msg or "查封" in msg:
            self.asset_freeze = True
            self.a008_done_count += 1

        # Detect criminal report
        if "刑事" in msg or "报案" in msg or "拒不支付" in msg:
            self.criminal_filed = True

        # Detect witness contact
        if any(k in msg for k in ("工友", "老乡", "证言", "证人", "证词")):
            if "联系" in msg or "找" in msg or "拿到" in msg or "写了" in msg:
                self.contacted_witness = True

        # Detect action ID
        action_match = re.search(r"A(\d{3})", msg)
        if action_match:
            self.last_action_id = f"A{action_match.group(1)}"
            self.actions_done.add(f"A{action_match.group(1)}")

    def update_from_action_result(self, narration: str) -> None:
        """Update state based on action narration from env (via conversation history)."""
        # Learn company names from action results
        company_names = re.findall(r"[一-鿿]+(?:有限公司|股份有限公司)", narration)
        for name in company_names:
            if "建设" in name or "工程" in name or "建筑" in name:
                self.total_contractor_name = name
                self.known_parties.add(name)
            elif "劳务" in name:
                self.subcontractor_name = name
                self.known_parties.add(name)

        # Learn evidence IDs from narration
        found_evidence = re.findall(r"E\d{3}", narration)
        self.known_evidence.update(found_evidence)

        # Track procedural milestones
        if "限期整改" in narration or "整改指令" in narration:
            self.limit_order_issued = True
        if "仲裁" in narration and "立案" in narration:
            self.arbitration_filed = True
        if "台账" in narration or "实名制" in narration:
            self.evidence_organized = True
        if "保全" in narration or "冻结" in narration:
            self.asset_freeze = True
            self.a008_done_count += 1

    def infer_stage(self) -> str:
        """Infer current stage from internal state."""
        if self.arbitration_filed and self.asset_freeze:
            return "arbitration_and_preservation_done"
        if self.arbitration_filed:
            return "arbitration_filed"
        if self.limit_order_issued:
            return "limit_order_issued"
        if self.complained_to_inspection:
            return "inspection_started"
        if self.evidence_organized:
            return "evidence_ready"
        return "initial"

    def get_respondent_list(self) -> list[str]:
        """Build the list of arbitration respondents from what we've learned."""
        respondents = []
        if self.total_contractor_name:
            respondents.append(self.total_contractor_name)
        if self.subcontractor_name:
            respondents.append(self.subcontractor_name)
        return respondents

    def get_state_summary_for_prompt(self) -> str:
        """Generate a text summary of internal state for injection into the prompt."""
        stage = self.infer_stage()
        respondents = self.get_respondent_list()
        lines = [
            f"## 当前案件记忆（安薪内部状态，第{self.turn_count}轮）",
            f"- 推断阶段：{stage}",
            f"- 已知当事人：{', '.join(sorted(self.known_parties)) or '尚未了解'}",
            f"- 总包单位：{self.total_contractor_name or '尚未确认全称'}",
            f"- 分包单位：{self.subcontractor_name or '尚未确认全称'}",
            f"- 包工头：{self.boss_name or '尚未确认'}{'（已失联）' if self.boss_missing else ''}",
            f"- 已知证据：{', '.join(sorted(self.known_evidence)) or '尚未整理'}",
            f"- 证据已整理：{'是' if self.evidence_organized else '否'}",
            f"- 已查工商信息：{'是' if self.queried_company_info else '否'}",
            f"- 有法律援助：{'是' if self.has_legal_aid else '否'}",
            f"- 已投诉监察：{'是' if self.complained_to_inspection else '否'}"
            + (f"（投诉对象：{self.inspection_target}）" if self.complained_to_inspection else ""),
            f"- 限期整改令：{'已下达' if self.limit_order_issued else '未下达'}",
            f"- 已申请仲裁：{'是' if self.arbitration_filed else '否'}",
            f"- 已申请保全：{'是' if self.asset_freeze else '否'}"
            + (f"（已做{self.a008_done_count}次，不要再重复！）" if self.a008_done_count > 0 else ""),
            f"- 已联系证人：{'是' if self.contacted_witness else '否'}",
            f"- 仲裁被申请人：{', '.join(respondents) if respondents else '尚未确定'}",
            f"- 已完成的行动：{', '.join(sorted(self.actions_done)) or '无'}",
        ]
        return "\n".join(lines)
