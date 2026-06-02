"""FastAPI server for interactive Anxin demo sessions."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from enum import Enum
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from advisor.anxin_advisor import AnxinAdvisor
from advisor.base import AdvisoryRequest
from advisor.doubao_advisor import DoubaoAdvisor
from api.schemas import (
    FinalizeResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionStateResponse,
    TurnResponse,
)
from case_loader import load_case
from environment.actions import make_final_action
from environment.env import Environment, Observation
from worker.simulated_worker import SimulatedWorker


app = FastAPI(title="Anxin Sandbox API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, "DemoSession"] = {}


def _plain(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _plain(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _observation_to_dict(obs: Observation) -> dict[str, Any]:
    return {
        "day": obs.day,
        "date": obs.date.isoformat(),
        "procedural_stage": obs.procedural_stage,
        "recent_events": list(obs.recent_events),
        "actions_taken_summary": list(obs.actions_taken_summary),
    }


def _evidence_pool(env: Environment) -> list[dict[str, Any]]:
    return [
        {
            "id": e.id,
            "name": e.name,
            "details": e.details,
            "strength": e.evidentiary_strength,
            "proves": e.proves,
            "obtained_at_day": e.obtained_at_day,
        }
        for e in env.state.evidence_pool.values()
    ]


def _case_for_frontend(case_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_data["case_id"],
        "case_name": case_data["case_name"],
        "ground_truth": case_data["ground_truth"],
        "worker": case_data["worker"],
        "financial": case_data["financial"],
        "timeline": case_data["timeline"],
        "evidence_database": case_data["evidence_database"],
        "final_submission_actions": case_data.get("final_submission_actions", {}),
        "scoring_rubric": case_data.get("scoring_rubric", {}),
        "ui_assets": case_data.get("ui_assets", {}),
    }


class DemoSession:
    def __init__(self, advisor_type: str, max_turns: int, case_path: str):
        self.session_id = uuid.uuid4().hex
        self.advisor_type = advisor_type
        self.max_turns = max(1, min(int(max_turns), 30))
        self.case_path = case_path
        self.case_data = load_case(case_path)
        self.env = Environment(self.case_data)
        self.env._advisor_name = advisor_type
        self.worker = SimulatedWorker()
        self.advisor = AnxinAdvisor() if advisor_type == "anxin" else DoubaoAdvisor()
        self.advisor.reset()
        self.obs = self.env.reset()
        self.turn_index = 0
        self.conversation_history: list[dict[str, str]] = []
        self.turns: list[dict[str, Any]] = []
        self.finalized = False
        self.judgment = None

    def run_one_turn(self) -> TurnResponse:
        if self.finalized:
            raise HTTPException(status_code=409, detail="Session already finalized")
        if self.turn_index >= self.max_turns:
            raise HTTPException(status_code=409, detail="No turns remaining; finalize the session")

        is_last = self.turn_index == self.max_turns - 1
        request_obj = self.worker.formulate_request(self.obs)
        req = AdvisoryRequest(
            worker_message=request_obj.text,
            conversation_history=list(self.conversation_history),
            advisor_session_id=self.session_id,
            current_turn_index=self.turn_index,
            max_turns=self.max_turns,
            remaining_turns=max(0, self.max_turns - self.turn_index - 1),
            is_final_turn=is_last,
        )
        response = self.advisor.give_advice(req)
        self.conversation_history.append({"role": "worker", "content": request_obj.text})
        self.conversation_history.append({"role": "advisor", "content": response.text})

        worker_reasoning = ""
        if is_last:
            channels = self.case_data.get("final_submission_actions", {}).get("channels", [])
            sub = self.worker.formulate_final_submission(
                advice_text=response.text,
                evidence_summary=self.env.state.evidence_summary(),
                channels=channels,
            )
            action = make_final_action(
                channel_id=sub.get("channel_id", "CH_GIVE_UP"),
                channel_name=sub.get("channel_name", "放弃维权"),
                advisor_reasoning=sub.get("advisor_reasoning", ""),
                drafted_documents=sub.get("drafted_documents", []),
                evidence_ids=sub.get("evidence_ids_submitted", list(self.env.state.evidence_pool.keys())),
                respondents=sub.get("respondents", []),
            )
            worker_reasoning = sub.get("advisor_reasoning", "")
        else:
            choice = self.worker.choose_action(
                advice_text=response.text,
                available_actions=self.env.available_actions(),
                advisor_hints=response.suggested_action_hints,
            )
            action = choice.action
            worker_reasoning = choice.reasoning

        new_obs, result = self.env.step(action)
        self.obs = new_obs

        spec = self.env.action_specs.get(action.action_id)
        action_name = spec.name if spec else action.action_id
        npc_items = []
        for npc_id, text in result.npc_interactions:
            npc_data = next(
                (n for n in self.case_data.get("npcs", []) if n.get("id") == npc_id),
                self.case_data.get("judge_npcs", {}).get(npc_id, {"display_name": npc_id}),
            )
            npc_items.append({
                "npc_id": npc_id,
                "npc_name": npc_data.get("display_name", npc_id),
                "text": text,
            })

        payload = {
            "session_id": self.session_id,
            "advisor_type": self.advisor_type,
            "turn_index": self.turn_index,
            "day": new_obs.day,
            "worker_message": request_obj.text,
            "advisor_response": response.text,
            "advisor_hints": list(response.suggested_action_hints),
            "chosen_action_id": action.action_id,
            "chosen_action_name": action_name,
            "chosen_action_params": dict(action.parameters),
            "worker_reasoning": worker_reasoning,
            "action_narration": result.narration,
            "action_success": result.success,
            "new_evidence_ids": list(result.new_evidence_ids),
            "npc_interactions": npc_items,
            "current_evidence_pool": _evidence_pool(self.env),
            "procedural_stage": self.env.state.procedural_stage.value,
            "is_terminal": self.env.is_terminal or is_last,
            "is_final_turn": is_last,
            "remaining_turns": max(0, self.max_turns - self.turn_index - 1),
        }
        self.turns.append(payload)
        self.turn_index += 1
        return TurnResponse(**payload)

    def finalize(self) -> FinalizeResponse:
        if not self.finalized:
            self.judgment = self.env.finalize()
            self.finalized = True

        final_submission = self.env.state.final_submission
        channel_id = final_submission.channel_id if final_submission else "CH_GIVE_UP"
        channel = self._channel(channel_id)
        payload = {
            "session_id": self.session_id,
            "advisor_type": self.advisor_type,
            "judgment": _plain(self.judgment),
            "final_submission": _plain(final_submission) if final_submission else None,
            "channel_id": channel_id,
            "channel_name": (
                final_submission.channel_name
                if final_submission and final_submission.channel_name
                else channel.get("name", "放弃维权")
            ),
            "channel_background_key": channel.get("background_key", "court"),
            "score": compute_score(self.env, self.judgment, self.case_data),
        }
        return FinalizeResponse(**payload)

    def _channel(self, channel_id: str) -> dict[str, Any]:
        for c in self.case_data.get("final_submission_actions", {}).get("channels", []):
            if c.get("id") == channel_id:
                return c
        return {}

    def state_response(self) -> SessionStateResponse:
        return SessionStateResponse(
            session_id=self.session_id,
            advisor_type=self.advisor_type,
            max_turns=self.max_turns,
            turns_completed=self.turn_index,
            is_terminal=self.env.is_terminal or self.turn_index >= self.max_turns,
            finalized=self.finalized,
            evidence_pool=_evidence_pool(self.env),
            procedural_stage=self.env.state.procedural_stage.value,
            total_days=self.env.state.current_day,
            turns=list(self.turns),
            judgment=_plain(self.judgment) if self.judgment else None,
        )


def compute_score(env: Environment, judgment, case_data: dict[str, Any]) -> dict[str, Any]:
    state = env.state
    award = getattr(judgment, "monetary_award", None)
    total_owed = int(case_data["financial"]["total_owed"])
    actions = {a.action_id for a in state.actions_taken}
    liability_text = " ".join(
        f"{getattr(lf, 'party', '')} {getattr(lf, 'liability_type', '')}"
        for lf in getattr(judgment, "liability_findings", [])
    )
    ev_values = list(state.evidence_pool.values())
    evidence_quality = (
        sum(e.evidentiary_strength for e in ev_values) / (len(ev_values) * 5)
        if ev_values else 0
    )
    recovery = min(1.0, (getattr(award, "total", 0) if award else 0) / (total_owed * 1.75))
    scores = {
        "primary_respondent": 1.0 if "宏基" in liability_text else 0.2,
        "evidence_quality": round(evidence_quality, 3),
        "procedure_efficiency": round(max(0.0, 1.0 - max(0, state.current_day - 45) / 200), 3),
        "recovery_ratio": round(recovery, 3),
        "key_tools_used": round(sum(a in actions for a in ("A008", "A009", "A012")) / 3, 3),
        "trap_avoidance": 1.0 if not any("李大海" in str(r) and len(str(r)) < 6 for r in (state.final_submission.respondents if state.final_submission else [])) else 0.3,
    }
    weights = {
        d["id"]: d.get("weight", 0)
        for d in case_data.get("scoring_rubric", {}).get("dimensions", [])
    }
    total = sum(scores[k] * weights.get(k, 0) for k in scores)
    return {"dimensions": scores, "total": round(total, 3)}


@app.post("/sessions", response_model=SessionStartResponse)
def start_session(req: SessionStartRequest):
    if req.advisor_type not in ("anxin", "doubao"):
        raise HTTPException(status_code=400, detail="advisor_type must be anxin or doubao")
    session = DemoSession(req.advisor_type, req.max_turns, req.case_path)
    _sessions[session.session_id] = session
    return SessionStartResponse(
        session_id=session.session_id,
        advisor_type=session.advisor_type,
        max_turns=session.max_turns,
        case_data=_case_for_frontend(session.case_data),
        initial_observation=_observation_to_dict(session.obs),
    )


@app.post("/sessions/{session_id}/turn", response_model=TurnResponse)
def run_turn(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.run_one_turn()


@app.post("/sessions/{session_id}/finalize", response_model=FinalizeResponse)
def finalize_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.finalize()


@app.get("/sessions/{session_id}/state", response_model=SessionStateResponse)
def get_session_state(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.state_response()


try:
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
except RuntimeError:
    pass
