"""
The Environment.

Gym-style interface:
  env = Environment.from_case_file("cases/tianjiao_mingyuan.json")
  obs = env.reset()
  while not env.is_terminal:
      action = ...
      obs, result = env.step(action)
  judgment = env.adjudicate()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from environment.state import (
    CaseState,
    FinancialState,
    ProceduralStage,
    TerminalReason,
    ActionRecord,
    NpcInteraction,
)
from environment.actions import Action, ActionSpec, ActionResult, evaluate_preconditions
from environment.action_handlers import get_handler, HandlerDeps
from environment.npc_manager import NpcManager

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Observation: what the env shows the worker each turn
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    What the worker sees each turn. This is the worker's-eye view, NOT
    the environment's full state. It's also what (filtered through the
    worker's voice) the advisor will eventually hear.
    """
    day: int
    date: date
    procedural_stage: str
    recent_events: list[str] = field(default_factory=list)
    # ↑ narrations from the most recent N turns (env updates + NPC responses)

    last_action_result: ActionResult | None = None
    last_npc_message: str | None = None

    actions_taken_summary: list[str] = field(default_factory=list)
    # ↑ short summary like ["第1天 A001 整理证据", "第2天 A009 申请法援"]

    def format_recent_events(self) -> str:
        if not self.recent_events:
            return "（暂无新进展）"
        return "\n".join(f"- {e}" for e in self.recent_events)

    def format_action_history(self, last_n: int = 5) -> str:
        if not self.actions_taken_summary:
            return "（这是赵建国第一次开口求助）"
        return "\n".join(f"- {a}" for a in self.actions_taken_summary[-last_n:])


# ---------------------------------------------------------------------------
# The Environment
# ---------------------------------------------------------------------------

class Environment:
    def __init__(self, case_data: dict):
        self.case_data = case_data
        self.state: CaseState = self._build_initial_state()
        self.npc_manager = NpcManager(case_data["npcs"])
        self.action_specs: dict[str, ActionSpec] = self._load_action_specs()
        self._recent_events: list[str] = []

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_case_file(cls, path: str) -> "Environment":
        import json
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def _build_initial_state(self) -> CaseState:
        cd = self.case_data
        worker = cd["worker"]
        fin = cd["financial"]
        return CaseState(
            case_id=cd["case_id"],
            worker_name=worker["name"],
            worker_id_card=worker["id_card"],
            start_date=date(2024, 4, 20),
            financial=FinancialState(
                total_owed=fin["total_owed"],
                daily_wage=fin["daily_wage"],
                working_days=fin["working_days"],
                total_earned=fin["total_earned"],
                already_paid=fin["already_paid"],
            ),
            liable_parties_identified=["李大海"],
        )

    def _load_action_specs(self) -> dict[str, ActionSpec]:
        specs = {}
        for raw in self.case_data["action_space"]:
            specs[raw["id"]] = ActionSpec(
                id=raw["id"],
                name=raw["name"],
                category=raw["category"],
                preconditions=raw.get("preconditions", []),
                duration_days=raw.get("duration_days", 1),
                parameters_required=raw.get("parameters_required", []),
                raw=raw,
            )
        return specs

    # ------------------------------------------------------------------
    # gym-like API
    # ------------------------------------------------------------------
    def reset(self) -> Observation:
        self.state = self._build_initial_state()
        self._recent_events = [
            "赵建国今天打开了'安薪'App，准备求助。",
            f"他手头持有：手写工资结算单一张（欠{self.state.financial.total_owed}元），"
            f"微信转账记录8笔，催讨记录3张，工地照片若干。",
            "李大海手机停机，恒达和宏基都把他推回来。他不知道下一步该怎么办。",
        ]
        return self._build_observation()

    @property
    def is_terminal(self) -> bool:
        if self.state.is_terminal:
            return True
        if self.state.current_day > 365:
            self.state.mark_terminal(TerminalReason.TIMEOUT)
            return True
        if self.state.giving_up_intent_count >= 3:
            self.state.mark_terminal(TerminalReason.ABANDONED)
            return True
        # success conditions
        recovered_ratio = (
            self.state.financial.principal_recovered / self.state.financial.total_owed
            if self.state.financial.total_owed > 0
            else 0
        )
        if recovered_ratio >= 0.8:
            self.state.mark_terminal(TerminalReason.SUCCESS_FULL_RECOVERY)
            return True
        return False

    def available_actions(self) -> list[ActionSpec]:
        """Return action specs whose preconditions currently pass."""
        out = []
        for spec in self.action_specs.values():
            ok, _ = evaluate_preconditions(spec, self.state)
            if ok:
                out.append(spec)
        return out

    def step(self, action: Action) -> tuple[Observation, ActionResult]:
        """Execute an action and return the new observation + result."""
        spec = self.action_specs.get(action.action_id)
        if not spec:
            result = ActionResult(action=action, success=False, narration="",
                                  error=f"Unknown action {action.action_id}")
            return self._build_observation(), result

        ok, reason = evaluate_preconditions(spec, self.state)
        if not ok:
            result = ActionResult(
                action=action, success=False,
                narration=f"赵建国想做'{spec.name}'，但发现条件不够：{reason}",
                error=reason,
            )
            self._recent_events.append(result.narration)
            return self._build_observation(), result

        deps = HandlerDeps(case_data=self.case_data, npc_manager=self.npc_manager)
        handler = get_handler(action.action_id)
        result = handler(self.state, action, deps)

        # record
        self.state.actions_taken.append(ActionRecord(
            day=self.state.current_day,
            action_id=action.action_id,
            action_name=spec.name,
            parameters=action.parameters,
            success=result.success,
            narration=result.narration,
            state_changes=result.state_changes,
        ))
        for npc_id, npc_text in result.npc_interactions:
            self.state.npc_interactions.append(NpcInteraction(
                day=self.state.current_day,
                npc_id=npc_id,
                worker_message=action.parameters.get("message", ""),
                npc_response=npc_text,
            ))
        self._recent_events.append(result.narration)
        # keep recent_events bounded
        self._recent_events = self._recent_events[-8:]

        return self._build_observation(), result

    def adjudicate(self):
        """Run the judgment engine on the current (terminal) state."""
        from judge.judgment_engine import JudgmentEngine
        engine = JudgmentEngine()
        return engine.adjudicate(self.state, self.case_data)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> Observation:
        return Observation(
            day=self.state.current_day,
            date=self.state.current_date,
            procedural_stage=self.state.procedural_stage.value,
            recent_events=list(self._recent_events),
            last_action_result=(
                None if not self.state.actions_taken
                else None  # action result not stored on env, only via step return
            ),
            actions_taken_summary=[
                f"第{a.day}天 {a.action_id} {a.action_name}"
                for a in self.state.actions_taken
            ],
        )
