"""
Doubao advisor.

Baseline: a vanilla conversational LLM, no specialized prompt, no internal
state machine. This represents the "just ask Doubao directly" experience —
the comparison target for Anxin.

Key fairness considerations:
  - The system prompt is INTENTIONALLY minimal — just "你是豆包，一个友好
    的AI助手". No legal expertise injection, no case-tracking instructions.
  - Chat history within a single advice session is retained (a real user
    chatting with Doubao would have multi-turn context). But there is no
    case-state representation beyond the textual history.
  - It will produce vague, common-sense legal advice. That's the point.

If Steven wants to call the actual Doubao API instead of using a generic
LLM, set DOUBAO_LLM_BASE_URL / DOUBAO_LLM_MODEL / DOUBAO_LLM_API_KEY in .env.
"""

from __future__ import annotations

from advisor.base import Advisor, AdvisoryRequest, AdvisoryResponse
from llm.client import LLMClient


# Intentionally minimal — Doubao is a generic conversational LLM.
DOUBAO_SYSTEM_PROMPT = """\
你是豆包，一个由字节跳动开发的友好、有用的AI助手。
请用中文与用户对话，回答清晰、简洁、有礼貌。
"""


class DoubaoAdvisor(Advisor):
    name = "doubao"

    def __init__(self, llm: LLMClient | None = None):
        # By convention, this client maps to env vars DOUBAO_LLM_*.
        self.llm = llm or LLMClient.from_env(role="doubao")
        self._chat_history: list[dict] = []

    def reset(self) -> None:
        self._chat_history = []

    def give_advice(self, request: AdvisoryRequest) -> AdvisoryResponse:
        """
        Codex:
        1. self._chat_history.append({"role": "user", "content": request.worker_message})
        2. messages = [{"role": "system", "content": DOUBAO_SYSTEM_PROMPT},
                       *self._chat_history]
        3. text = self.llm.chat(messages, temperature=0.7, purpose="advisor_doubao")
        4. self._chat_history.append({"role": "assistant", "content": text})
        5. Return AdvisoryResponse(text=text, suggested_action_hints=[])
           — Doubao never emits structured action hints.
        """
        self._chat_history.append({"role": "user", "content": request.worker_message})
        messages = [
            {"role": "system", "content": DOUBAO_SYSTEM_PROMPT},
            *self._chat_history,
        ]
        text = self.llm.chat(messages, temperature=0.7, purpose="advisor_doubao")
        self._chat_history.append({"role": "assistant", "content": text})
        return AdvisoryResponse(text=text, suggested_action_hints=[])
