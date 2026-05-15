"""Pydantic schemas for the Anxin session API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class SessionStartRequest(BaseModel):
    advisor_type: str = "anxin"
    max_turns: int = 8
    case_path: str = "cases/tianjiao_mingyuan.json"


class SessionStartResponse(BaseModel):
    session_id: str
    advisor_type: str
    max_turns: int
    case_data: dict[str, Any]
    initial_observation: dict[str, Any]


class TurnResponse(BaseModel):
    session_id: str
    advisor_type: str
    turn_index: int
    day: int
    worker_message: str
    advisor_response: str
    advisor_hints: list[str]
    chosen_action_id: str
    chosen_action_name: str
    chosen_action_params: dict[str, Any]
    action_narration: str
    action_success: bool
    new_evidence_ids: list[str]
    npc_interactions: list[dict[str, Any]]
    current_evidence_pool: list[dict[str, Any]]
    procedural_stage: str
    is_terminal: bool
    is_final_turn: bool
    remaining_turns: int


class FinalizeResponse(BaseModel):
    session_id: str
    advisor_type: str
    judgment: dict[str, Any]
    final_submission: dict[str, Any] | None
    channel_id: str
    channel_name: str
    channel_background_key: str
    score: dict[str, Any]


class SessionStateResponse(BaseModel):
    session_id: str
    advisor_type: str
    max_turns: int
    turns_completed: int
    is_terminal: bool
    finalized: bool
    evidence_pool: list[dict[str, Any]]
    procedural_stage: str
    total_days: int
    turns: list[dict[str, Any]]
    judgment: dict[str, Any] | None = None
