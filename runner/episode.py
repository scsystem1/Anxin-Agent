"""
Episode runner.

Drives one full simulation loop with one advisor. The loop:

    env.reset() → obs
    while not env.is_terminal:
        request = worker.formulate_request(obs)
        advice = advisor.give_advice(request)
        action_choice = worker.choose_action(advice, env.available_actions(),
                                             advice.suggested_action_hints)
        obs, result = env.step(action_choice.action)
    judgment = env.adjudicate()

The runner also keeps a transcript: every observation, request, advice,
action, NPC interaction. This transcript is what the demo UI displays.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from environment.env import Environment, Observation
from environment.actions import ActionResult
from worker.simulated_worker import SimulatedWorker, WorkerRequest, WorkerActionChoice
from advisor.base import Advisor, AdvisoryRequest, AdvisoryResponse
from judge.judgment_engine import Judgment


# ---------------------------------------------------------------------------
# Transcript types
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    """One full turn of the loop."""
    turn_index: int
    day: int
    observation_recent_events: list[str]
    worker_request_text: str
    advisor_response_text: str
    advisor_hints: list[str]
    chosen_action_id: str
    chosen_action_params: dict
    worker_reasoning: str
    action_narration: str
    action_success: bool
    new_evidence_ids: list[str] = field(default_factory=list)
    action_narration_full: str = ""
    npc_interactions: list[dict[str, str]] = field(default_factory=list)
    current_evidence_pool: list[dict[str, Any]] = field(default_factory=list)
    procedural_stage: str = ""
    is_final_turn: bool = False


@dataclass
class EpisodeResult:
    advisor_name: str
    transcript: list[TurnRecord] = field(default_factory=list)
    final_judgment: Judgment | None = None
    total_days: int = 0
    total_turns: int = 0
    terminal_reason: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EpisodeRunner:
    def __init__(
        self,
        env: Environment,
        worker: SimulatedWorker,
        advisor: Advisor,
        max_turns: int = 30,
        verbose: bool = True,
    ):
        self.env = env
        self.worker = worker
        self.advisor = advisor
        self.max_turns = max_turns
        self.verbose = verbose

    def run(self) -> EpisodeResult:
        """Run a full episode and return the result."""
        self.advisor.reset()
        obs = self.env.reset()
        result = EpisodeResult(advisor_name=self.advisor.name)
        conversation_history: list[dict] = []

        for turn_idx in range(self.max_turns):
            if self.env.is_terminal:
                break

            # Step 1 & 2: worker observes, formulates request
            request_obj = self.worker.formulate_request(obs)
            req = AdvisoryRequest(
                worker_message=request_obj.text,
                conversation_history=list(conversation_history),
                current_turn_index=turn_idx,
                max_turns=self.max_turns,
                remaining_turns=max(0, self.max_turns - turn_idx - 1),
                is_final_turn=(turn_idx == self.max_turns - 1),
            )
            self._log(f"\n══ Turn {turn_idx + 1} ── 第{obs.day}天 ──")
            self._log(f"[Worker → Advisor] {request_obj.text}")

            # Step 3: advisor responds
            response = self.advisor.give_advice(req)
            self._log(f"[Advisor → Worker] {response.text}")
            if response.suggested_action_hints:
                self._log(f"  (hints: {response.suggested_action_hints})")

            conversation_history.append({"role": "worker", "content": request_obj.text})
            conversation_history.append({"role": "advisor", "content": response.text})

            is_last_turn = req.is_final_turn
            if is_last_turn:
                from environment.actions import make_final_action

                channels = self.env.case_data.get("final_submission_actions", {}).get("channels", [])
                submission = self.worker.formulate_final_submission(
                    advice_text=response.text,
                    evidence_summary=self.env.state.evidence_summary(),
                    channels=channels,
                )
                choice = WorkerActionChoice(
                    action=make_final_action(
                        channel_id=submission.get("channel_id", "CH_GIVE_UP"),
                        channel_name=submission.get("channel_name", "放弃维权"),
                        advisor_reasoning=submission.get("advisor_reasoning", ""),
                        drafted_documents=submission.get("drafted_documents", []),
                        evidence_ids=submission.get("evidence_ids_submitted", list(self.env.state.evidence_pool.keys())),
                        respondents=submission.get("respondents", []),
                    ),
                    reasoning=submission.get("advisor_reasoning", ""),
                )
            else:
                # Step 4: worker chooses action
                available = self.env.available_actions()
                choice = self.worker.choose_action(
                    advice_text=response.text,
                    available_actions=available,
                    advisor_hints=response.suggested_action_hints,
                )
            self._log(
                f"[Worker → Env] {choice.action.action_id} "
                f"params={choice.action.parameters} :: {choice.reasoning}"
            )

            # Step 5: environment executes
            obs, action_result = self.env.step(choice.action)
            self._log(f"[Env → Worker] {action_result.narration}")
            if action_result.new_evidence_ids:
                self._log(f"  (new evidence: {action_result.new_evidence_ids})")

            # record
            result.transcript.append(TurnRecord(
                turn_index=turn_idx,
                day=obs.day,
                observation_recent_events=list(obs.recent_events),
                worker_request_text=request_obj.text,
                advisor_response_text=response.text,
                advisor_hints=list(response.suggested_action_hints),
                chosen_action_id=choice.action.action_id,
                chosen_action_params=dict(choice.action.parameters),
                worker_reasoning=choice.reasoning,
                action_narration=action_result.narration,
                action_success=action_result.success,
                new_evidence_ids=list(action_result.new_evidence_ids),
                action_narration_full=action_result.narration,
                npc_interactions=[
                    {"npc_id": npc_id, "text": text}
                    for npc_id, text in action_result.npc_interactions
                ],
                current_evidence_pool=[
                    {
                        "id": e.id,
                        "name": e.name,
                        "details": e.details,
                        "strength": e.evidentiary_strength,
                        "proves": e.proves,
                        "obtained_at_day": e.obtained_at_day,
                    }
                    for e in self.env.state.evidence_pool.values()
                ],
                procedural_stage=self.env.state.procedural_stage.value,
                is_final_turn=is_last_turn,
            ))
            if is_last_turn:
                break

        # Adjudicate
        self._log(f"\n══ Adjudicating {self.advisor.name} ══")
        judgment = self.env.finalize()
        # Patch the advisor name in (env doesn't know it)
        judgment.advisor_name = self.advisor.name

        result.final_judgment = judgment
        result.total_turns = len(result.transcript)
        result.total_days = self.env.state.current_day
        result.terminal_reason = (
            self.env.state.terminal_reason.value
            if self.env.state.terminal_reason else "max_turns"
        )
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
