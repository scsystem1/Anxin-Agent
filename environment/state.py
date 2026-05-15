"""
Core state objects for the Anxin sandbox environment.

This module defines the data structures that represent the case state at any
point in time. EVERYTHING that the environment tracks lives here: evidence,
timeline, procedural stage, NPC relationships, money, etc.

DESIGN NOTE: The advisor (Anxin / Doubao) NEVER receives a CaseState object
directly. The advisor only sees what the simulated worker tells them in
natural language. Keeping CaseState as a Python object inside the environment
is what makes the comparison fair.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any


class ProceduralStage(str, Enum):
    """The legal/procedural stage the case is currently in."""
    INITIAL_INTAKE = "initial_intake"
    EVIDENCE_GATHERING = "evidence_gathering"
    NEGOTIATION = "negotiation"
    LABOR_INSPECTION = "labor_inspection"
    LABOR_INSPECTION_ORDER_ISSUED = "labor_inspection_order_issued"
    LABOR_INSPECTION_ORDER_EXPIRED = "labor_inspection_order_expired"
    ARBITRATION_FILED = "arbitration_filed"
    ARBITRATION_AWARDED = "arbitration_awarded"
    CIVIL_LITIGATION_DIRECT = "civil_litigation_direct"
    CIVIL_JUDGMENT = "civil_judgment"
    EXECUTION = "execution"
    SETTLEMENT = "settlement"
    ABANDONED = "abandoned"


class TerminalReason(str, Enum):
    SUCCESS_FULL_RECOVERY = "success_full_recovery"
    SUCCESS_PARTIAL = "success_partial"
    JUDGMENT_RENDERED = "judgment_rendered"  # case went all the way to judgment
    ABANDONED = "abandoned"
    TIMEOUT = "timeout"


@dataclass
class Evidence:
    """A single piece of evidence in the case."""
    id: str                       # e.g. "E001"
    name: str                     # e.g. "手写工资结算单"
    details: str                  # human-readable description
    evidentiary_strength: int     # 1-5
    proves: str                   # what it establishes
    obtained_at_day: int = 0      # virtual day when added


@dataclass
class NpcInteraction:
    """A single back-and-forth with an NPC."""
    day: int
    npc_id: str
    worker_message: str
    npc_response: str


@dataclass
class ActionRecord:
    """A record of an action that was executed."""
    day: int
    action_id: str
    action_name: str
    parameters: dict[str, Any]
    success: bool
    narration: str
    state_changes: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinancialState:
    total_owed: int
    daily_wage: int
    working_days: int
    total_earned: int
    already_paid: int
    principal_recovered: int = 0
    additional_compensation_recovered: int = 0
    interest_recovered: int = 0
    legal_costs_paid: int = 0

    @property
    def total_recovered(self) -> int:
        return self.principal_recovered + self.additional_compensation_recovered + self.interest_recovered


@dataclass
class FinalSubmission:
    """
    The advisor-guided final submission at the end of an episode.

    This is intentionally structured: the environment and judge should not
    infer the chosen legal route from free-form prose.
    """
    channel_id: str
    channel_name: str
    advisor_reasoning: str
    drafted_documents: list[dict[str, Any]]
    evidence_ids_submitted: list[str]
    respondents: list[str]


@dataclass
class CaseState:
    """
    The complete state of the case at the current moment.

    This object lives inside the Environment and is mutated by action handlers
    and NPC interactions. It is NEVER serialized to the advisor.
    """
    # --- identity ---
    case_id: str
    worker_name: str
    worker_id_card: str

    # --- time ---
    start_date: date              # virtual start of the simulation (e.g. 2024-04-20)
    current_day: int = 0          # days since start_date

    # --- financial ---
    financial: FinancialState = field(default_factory=lambda: FinancialState(0, 0, 0, 0, 0))

    # --- procedural ---
    procedural_stage: ProceduralStage = ProceduralStage.INITIAL_INTAKE

    # --- evidence ---
    evidence_pool: dict[str, Evidence] = field(default_factory=dict)

    # --- parties identified by the worker so far ---
    liable_parties_identified: list[str] = field(default_factory=list)
    # ↑ e.g. ["李大海"] initially, may grow to ["李大海", "宏基建设", "恒达劳务"]

    # --- knowledge the worker has acquired (initially mostly empty) ---
    worker_known_facts: set[str] = field(default_factory=set)
    # ↑ e.g. "宏基有先行清偿责任", "实名台账存在", "可申请财产保全免担保", ...

    # --- NPC relationship state ---
    npc_pressure_level: dict[str, int] = field(default_factory=dict)
    # ↑ e.g. {"zhang_guohua": 0, "wang_pei": 1}; raised by formal procedures

    # --- procedural flags ---
    flags: dict[str, Any] = field(default_factory=dict)
    # ↑ examples: {"limit_order_issued": True, "asset_freeze_active": False,
    #             "criminal_case_filed": False, "has_legal_aid_lawyer": True}

    # --- statute of limitations tracking ---
    statute_of_limitations_interrupted: bool = True
    # ↑ initially True because catching-up WeChat messages count as 中断

    # --- history ---
    actions_taken: list[ActionRecord] = field(default_factory=list)
    npc_interactions: list[NpcInteraction] = field(default_factory=list)

    # --- adversary positions (tracked separately, used by judgment engine) ---
    respondent_defenses: dict[str, list[str]] = field(default_factory=dict)
    # ↑ {"宏基建设": ["已付清恒达工程款", "无直接合同关系"], ...}

    # --- final submission ---
    final_submission: FinalSubmission | None = None

    # --- terminal ---
    is_terminal: bool = False
    terminal_reason: TerminalReason | None = None
    giving_up_intent_count: int = 0

    # ------------------------------------------------------------------
    # convenience methods
    # ------------------------------------------------------------------

    @property
    def current_date(self) -> date:
        return self.start_date + timedelta(days=self.current_day)

    def add_evidence(self, ev: Evidence) -> None:
        ev.obtained_at_day = self.current_day
        self.evidence_pool[ev.id] = ev

    def has_evidence(self, evidence_id: str) -> bool:
        return evidence_id in self.evidence_pool

    def evidence_summary(self) -> str:
        """Compact textual summary of evidence pool, used by NPCs / judge."""
        if not self.evidence_pool:
            return "（无证据）"
        lines = []
        for e in self.evidence_pool.values():
            lines.append(f"  - {e.id} {e.name}（强度{e.evidentiary_strength}/5）：{e.proves}")
        return "\n".join(lines)

    def advance_days(self, n: int) -> None:
        self.current_day += n

    def mark_terminal(self, reason: TerminalReason) -> None:
        self.is_terminal = True
        self.terminal_reason = reason
