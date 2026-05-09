"""
The Advisor interface.

This is the MOST IMPORTANT FILE in the project. It defines the boundary
between the environment and the things being compared (Anxin vs Doubao).

DESIGN PRINCIPLE — STRICT ISOLATION
-----------------------------------
The advisor receives ONLY:
  1. The worker's natural-language message (what the worker chose to share)
  2. The history of prior advisor↔worker exchanges in this conversation

The advisor receives NEVER:
  - The CaseState object
  - The evidence pool
  - The NPC objects
  - Any environment internals

This is what makes the comparison fair. Doubao is a vanilla LLM that knows
nothing about the case until the worker tells it. Anxin's value comes from
its OWN internal state-tracking (which the worker reveals through dialogue),
not from secretly reading environment state.

If you're tempted to add `case_state: CaseState` to AdvisoryRequest, STOP
and re-read this docstring.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AdvisoryRequest:
    """What the worker sends to the advisor."""
    worker_message: str
    # ↑ The most recent message in the worker's voice. This is the ONLY
    #   case-specific information the advisor receives.

    conversation_history: list[dict] = field(default_factory=list)
    # ↑ Prior turns of the conversation. Each dict has shape:
    #   {"role": "worker"|"advisor", "content": str}
    #   The advisor uses this for context within the conversation only.

    advisor_session_id: str = ""
    # ↑ For advisors that maintain server-side state (like the real Anxin
    #   agent), this id can be used to resume. Default "" means stateless.


@dataclass
class AdvisoryResponse:
    """What the advisor returns."""
    text: str
    # ↑ Free-form natural language. The simulated worker will read this and
    #   try to act on it. The text's specificity directly affects what the
    #   worker does next — see SimulatedWorker.choose_action().

    suggested_action_hints: list[str] = field(default_factory=list)
    # ↑ OPTIONAL. Anxin (the structured agent) may emit hints like
    #   ["A006:target=宏基建设", "A009"] alongside the text. The worker
    #   uses these as a tiebreaker when picking an action. Doubao normally
    #   leaves this empty.

    advisor_meta: dict = field(default_factory=dict)
    # ↑ For logging/debugging. Anxin can emit its internal state snapshot,
    #   confidence levels, etc. NOT used by the env or worker.


# ---------------------------------------------------------------------------
# The interface every advisor must implement
# ---------------------------------------------------------------------------

class Advisor(ABC):
    """
    Abstract advisor interface.

    Implement two concrete subclasses:
      - AnxinAdvisor    (advisor/anxin_advisor.py)
      - DoubaoAdvisor   (advisor/doubao_advisor.py)

    Both can default to LLM-backed implementations (different system prompts
    + different "memory" handling). When the real Anxin agent is ready,
    swap AnxinAdvisor.give_advice for an HTTP call to the agent service.
    """

    name: Literal["anxin", "doubao", "other"] = "other"

    @abstractmethod
    def give_advice(self, request: AdvisoryRequest) -> AdvisoryResponse:
        """Return advice in response to the worker's message."""
        ...

    def reset(self) -> None:
        """
        Reset any per-episode state inside the advisor.

        Default no-op. Anxin's real implementation should clear its case-state.
        Doubao's stateless implementation can leave this empty.
        """
        return None
