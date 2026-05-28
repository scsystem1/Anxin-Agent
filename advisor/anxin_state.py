"""
Internal case state tracker for Anxin Advisor.

Parses worker messages to maintain a best-effort model of the case state.
The advisor never receives CaseState directly — it infers everything from
the worker's natural-language messages.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class AnxinInternalState:
    """
    Anxin's internal model of the case, reconstructed from dialogue.

    Updated every turn based on what the worker says.
    """
    # What we know about the case
    known_parties: set[str] = field(default_factory=lambda: {"李大海"})
    known_evidence: set[str] = field(default_factory=set)
    evidence_organized: bool = False
    queried_hongji_info: bool = False
    has_legal_aid: bool = False

    # Procedural stage inference
    complained_to_inspection: bool = False
    inspection_target: str = ""  # who was complained about
    limit_order_issued: bool = False
    arbitration_filed: bool = False
    arbitration_respondents: list[str] = field(default_factory=list)
    asset_freeze: bool = False
    criminal_filed: bool = False

    # Communication tracking
    contacted_li_dahai: bool = False
    contacted_wang_pei: bool = False
    contacted_zhang_guohua: bool = False
    contacted_chen_wei: bool = False

    # Turn counter
    turn_count: int = 0
    last_action_id: str = ""

    # What the worker seems to know
    worker_knows_hongji_responsibility: bool = False
    worker_knows_legal_aid: bool = False

    def update_from_worker_message(self, msg: str) -> None:
        """Parse a worker message and update internal state."""
        self.turn_count += 1

        # Detect party mentions
        if any(k in msg for k in ("宏基", "总包", "张国华")):
            self.known_parties.add("宏基建设")
        if any(k in msg for k in ("恒达", "王主任", "王培")):
            self.known_parties.add("恒达劳务")
        if "陈" in msg and any(k in msg for k in ("监察", "科长", "投诉")):
            self.contacted_chen_wei = True

        # Detect evidence mentions
        if any(k in msg for k in ("整理", "翻了一遍", "结算单", "截图", "证据")):
            self.evidence_organized = True
            if not self.known_evidence:
                self.known_evidence = {"E001", "E002", "E003", "E004", "E005"}

        # Detect action outcomes from narration
        if "工商" in msg or "注册信息" in msg or "法定代表人" in msg:
            self.queried_hongji_info = True

        if any(k in msg for k in ("法援", "法律援助", "12348", "律师")):
            self.has_legal_aid = True
            self.worker_knows_legal_aid = True

        if "监察" in msg and any(k in msg for k in ("投诉", "受理", "陈科", "整改")):
            self.complained_to_inspection = True

        if "宏基" in msg and ("投诉" in msg or "整改" in msg or "指令" in msg):
            self.inspection_target = "宏基建设"

        if "限期整改" in msg or "整改令" in msg or "整改指令" in msg:
            self.limit_order_issued = True

        if any(k in msg for k in ("仲裁", "仲裁委", "仲裁申请")):
            self.arbitration_filed = True

        if "保全" in msg or "冻结" in msg or "查封" in msg:
            self.asset_freeze = True

        if "刑事" in msg or "报案" in msg or "拒不支付" in msg:
            self.criminal_filed = True

        # NPC contacts
        if any(k in msg for k in ("李大海", "包工头", "李哥")) and ("电话" in msg or "打" in msg or "停机" in msg):
            self.contacted_li_dahai = True
        if any(k in msg for k in ("王主任", "王培", "恒达")) and ("谈" in msg or "找" in msg or "说" in msg):
            self.contacted_wang_pei = True
        if any(k in msg for k in ("张国华", "张经理", "宏基项目")) and ("联系" in msg or "找" in msg or "说" in msg):
            self.contacted_zhang_guohua = True

        # Worker knowledge detection
        if "先行清偿" in msg or "总包有责任" in msg or "宏基应该付" in msg:
            self.worker_knows_hongji_responsibility = True

        # Detect action ID from worker's message (they sometimes mention it)
        action_match = re.search(r"A(\d{3})", msg)
        if action_match:
            self.last_action_id = f"A{action_match.group(1)}"

    def update_from_action_result(self, narration: str) -> None:
        """Update state based on action narration from env (via conversation history)."""
        # This is called with the advisor's own response context
        if "台账" in narration or "实名制" in narration:
            self.known_evidence.add("E006")
        if "汇款" in narration and "恒达" in narration:
            self.known_evidence.add("E008")
        if "承包协议" in narration:
            self.known_evidence.add("E009")
        if "证言" in narration or "王兴林" in narration:
            self.known_evidence.add("E010")
        if "工资专用账户" in narration:
            self.known_evidence.add("E007")
        if "限期整改" in narration or "整改指令" in narration:
            self.limit_order_issued = True
        if "仲裁" in narration and "立案" in narration:
            self.arbitration_filed = True

    def infer_stage(self) -> str:
        """Infer current stage from internal state."""
        if self.arbitration_filed:
            return "arbitration_filed"
        if self.limit_order_issued:
            return "limit_order_issued"
        if self.complained_to_inspection:
            return "inspection_started"
        if self.evidence_organized:
            return "evidence_ready"
        return "initial"

    def get_state_summary_for_prompt(self) -> str:
        """Generate a text summary of internal state for injection into the prompt."""
        stage = self.infer_stage()
        lines = [
            f"## 当前案件状态（安薪内部推断，第{self.turn_count}轮）",
            f"- 推断阶段：{stage}",
            f"- 已知当事人：{', '.join(sorted(self.known_parties))}",
            f"- 已知证据：{', '.join(sorted(self.known_evidence)) or '未整理'}",
            f"- 证据是否已整理：{'是' if self.evidence_organized else '否'}",
            f"- 是否已查宏基工商信息：{'是' if self.queried_hongji_info else '否'}",
            f"- 是否有法援律师：{'是' if self.has_legal_aid else '否'}",
            f"- 是否已投诉监察：{'是' if self.complained_to_inspection else '否'}"
            + (f"（投诉对象：{self.inspection_target}）" if self.complained_to_inspection else ""),
            f"- 是否有限期整改令：{'是' if self.limit_order_issued else '否'}",
            f"- 是否已申请仲裁：{'是' if self.arbitration_filed else '否'}",
            f"- 是否已联系NPC：李大海={'是' if self.contacted_li_dahai else '否'}，"
            f"王培={'是' if self.contacted_wang_pei else '否'}，"
            f"张国华={'是' if self.contacted_zhang_guohua else '否'}，"
            f"陈维={'是' if self.contacted_chen_wei else '否'}",
        ]
        return "\n".join(lines)
