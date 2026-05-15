"""
Action definitions and result types.

The Action Space is the FORMAL INTERFACE between the simulated worker and
the environment. The worker does not "freely act" — it must select an action
from the Action Space (loaded from cases/*.json) and provide structured
parameters. This makes the simulation deterministic and replayable.

Flow:
    1. Environment.available_actions(state) → list[ActionSpec]
    2. SimulatedWorker.choose_action(observation, advice) → Action
    3. Environment.execute(action) → ActionResult
"""

from __future__ import annotations
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from environment.state import ProceduralStage


@dataclass
class ActionSpec:
    """The static definition of an action loaded from JSON."""
    id: str                           # e.g. "A006"
    name: str                         # e.g. "向劳动监察大队投诉总包"
    category: str                     # "evidence" | "procedure" | "communication" | "information" | "terminal"
    preconditions: list[str]          # informal preconditions (env evaluates them)
    duration_days: int                # how many virtual days this consumes
    parameters_required: list[str] = field(default_factory=list)
    # rich metadata from the case JSON, opaque to handlers — interpret as needed
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.category == "terminal"


@dataclass
class Action:
    """A concrete action chosen by the worker, with parameters filled in."""
    action_id: str                    # e.g. "A006"
    parameters: dict[str, Any] = field(default_factory=dict)
    # e.g. {"target_company": "宏基建设"}


@dataclass
class ActionResult:
    """What the environment returns after executing an action."""
    action: Action
    success: bool
    narration: str
    # ↑ A natural-language description of what happened, written from the
    #   worker's perspective. THIS becomes part of the next observation.
    new_evidence_ids: list[str] = field(default_factory=list)
    npc_interactions: list[tuple[str, str]] = field(default_factory=list)
    # ↑ list of (npc_id, npc_response_text)
    state_changes: dict[str, Any] = field(default_factory=dict)
    days_elapsed: int = 0
    error: str | None = None


FINAL_ACTION_ID = "A_FINAL"


def make_final_action(
    channel_id: str,
    channel_name: str,
    advisor_reasoning: str,
    drafted_documents: list[dict],
    evidence_ids: list[str],
    respondents: list[str],
) -> Action:
    """Build the structured final-channel submission action."""
    return Action(
        action_id=FINAL_ACTION_ID,
        parameters={
            "channel_id": channel_id,
            "channel_name": channel_name,
            "advisor_reasoning": advisor_reasoning,
            "drafted_documents": drafted_documents,
            "evidence_ids_submitted": evidence_ids,
            "respondents": respondents,
        },
    )


# ---------------------------------------------------------------------------
# Helpers for action spec loading and precondition evaluation
# ---------------------------------------------------------------------------

def evaluate_preconditions(
    spec: ActionSpec,
    state,                             # CaseState (avoiding circular import)
) -> tuple[bool, str]:
    """
    Evaluate the informal precondition strings against the current state.

    Codex: implement a small DSL here. Examples of precondition strings:
        "evidence_pool_size>=2"
        "procedural_stage>=labor_inspection"
        "E001 in evidence_pool"
        "state.限期整改令已下达 == true"
        "worker_knows_wang_xinglin"
        "recent_npc_interaction.wang_pei"

    Return (passes, reason). If passes is False, reason explains why.
    Keep this dumb-simple — string parsing is fine for MVP.
    """
    for pre in spec.preconditions:
        ok, why = _eval_one(pre, state)
        if not ok:
            return False, why
    return True, ""


def _eval_one(pre: str, state) -> tuple[bool, str]:
    pre = pre.strip()
    if not pre:
        return True, ""

    m = re.match(r"^(E\d+)\s+in\s+evidence_pool$", pre)
    if m:
        ev_id = m.group(1)
        return state.has_evidence(ev_id), f"缺少证据 {ev_id}"

    m = re.match(r"^evidence_pool_size\s*(>=|>|==|<|<=)\s*(\d+)$", pre)
    if m:
        op, n = m.group(1), int(m.group(2))
        return (
            _cmp(len(state.evidence_pool), op, n),
            f"证据数量不满足 {pre}（当前 {len(state.evidence_pool)}）",
        )

    m = re.match(r"^procedural_stage\s*(>=|>|==|<|<=)\s*([\w_]+)$", pre)
    if m:
        op, stage_name = m.group(1), _normalize_stage_name(m.group(2))
        order = [s.value for s in ProceduralStage]
        cur = order.index(state.procedural_stage.value)
        try:
            target = order.index(stage_name)
        except ValueError:
            return False, f"未知阶段 {stage_name}"
        return _cmp(cur, op, target), f"程序阶段不满足 {pre}"

    m = re.match(r"^state\.([^\s]+)\s*==\s*(true|false)$", pre)
    if m:
        key, val = m.group(1), m.group(2) == "true"
        actual = _flag_value(state, key)
        return actual == val, f"标志 {key} 不为 {val}"

    if pre == "worker_knows_wang_xinglin":
        return True, ""

    m = re.match(r"^recent_npc_interaction\.([\w_]+)$", pre)
    if m:
        npc_id = m.group(1)
        recent = state.npc_interactions[-5:]
        return any(i.npc_id == npc_id for i in recent), f"最近未与 {npc_id} 交互"

    print(f"[Warn] unknown precondition: {pre!r}", file=sys.stderr)
    return True, ""


def _normalize_stage_name(stage_name: str) -> str:
    aliases = {
        "arbitration": ProceduralStage.ARBITRATION_FILED.value,
        "arbitration_award": ProceduralStage.ARBITRATION_AWARDED.value,
        "settlement_with_full_payment": ProceduralStage.SETTLEMENT.value,
    }
    return aliases.get(stage_name, stage_name)


def _flag_value(state, key: str) -> Any:
    aliases = {
        "限期整改令已下达": "limit_order_issued",
        "整改期满未支付": "limit_order_expired_unpaid",
    }
    keys = [key]
    if key in aliases:
        keys.append(aliases[key])
    reverse_aliases = {v: k for k, v in aliases.items()}
    if key in reverse_aliases:
        keys.append(reverse_aliases[key])

    for k in keys:
        if k in state.flags:
            return state.flags[k]

    if key in ("整改期满未支付", "limit_order_expired_unpaid"):
        due_day = state.flags.get("limit_order_due_day")
        if state.flags.get("limit_order_issued") and due_day is not None:
            return (
                state.current_day >= due_day
                and state.financial.principal_recovered < state.financial.total_owed
            )
    return None


def _cmp(a, op: str, b) -> bool:
    return {
        ">=": a >= b,
        ">": a > b,
        "==": a == b,
        "<": a < b,
        "<=": a <= b,
    }[op]
