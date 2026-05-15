"""
Anxin advisor.

This is a STANDIN. The real Anxin agent (with its case state machine,
evidence gap analyzer, legal route scorer, etc.) lives in a separate
codebase. The default implementation here is an LLM call with an Anxin-
flavored system prompt — useful for end-to-end pipeline testing.

When the real Anxin agent is ready, replace `give_advice` with an HTTP
call to the agent service. The interface (AdvisoryRequest →
AdvisoryResponse) does not change.
"""

from __future__ import annotations

from advisor.base import Advisor, AdvisoryRequest, AdvisoryResponse
from llm.client import LLMClient


# ---------------------------------------------------------------------------
# Default Anxin system prompt
# ---------------------------------------------------------------------------

ANXIN_SYSTEM_PROMPT = """\
你是「安薪」，一个帮助农民工讨薪的AI助手。
请用简洁的中文给出建议，帮助工人一步一步解决欠薪问题。
"""


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

class AnxinAdvisor(Advisor):
    name = "anxin"

    def __init__(self, llm: LLMClient | None = None, system_prompt: str | None = None):
        self.llm = llm or LLMClient.from_env(role="advisor")
        self.system_prompt = system_prompt or ANXIN_SYSTEM_PROMPT
        # Per-conversation memory. The real Anxin agent has rich internal
        # state; this default impl uses just the chat history as memory.
        self._chat_history: list[dict] = []

    def reset(self) -> None:
        self._chat_history = []

    def give_advice(self, request: AdvisoryRequest) -> AdvisoryResponse:
        """
        Default: call the LLM with the Anxin system prompt + accumulated
        chat history. To swap to the real Anxin service, replace the body
        of this method with an HTTP call:

            resp = httpx.post(ANXIN_AGENT_URL, json={
                "session_id": request.advisor_session_id,
                "message": request.worker_message,
            })
            data = resp.json()
            return AdvisoryResponse(text=data["text"],
                                    suggested_action_hints=data.get("hints", []))

        Codex:
        1. Append {"role": "user", "content": request.worker_message} to
           self._chat_history.
        2. messages = [{"role": "system", ...self.system_prompt}, *self._chat_history]
        3. text = self.llm.chat(messages, temperature=0.6, purpose="advisor_anxin")
        4. Parse trailing <<actions: ...>> line into hints list. Strip it from
           the text shown to the worker.
        5. Append {"role": "assistant", "content": text} to history.
        6. Return AdvisoryResponse.
        """
        self._chat_history.append({"role": "user", "content": request.worker_message})
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._chat_history,
        ]
        raw = self.llm.chat(messages, temperature=0.6, purpose="advisor_anxin")
        cleaned, hints = self._parse_action_hints(raw)
        self._chat_history.append({"role": "assistant", "content": cleaned})
        return AdvisoryResponse(text=cleaned, suggested_action_hints=hints)

    @staticmethod
    def _parse_action_hints(text: str) -> tuple[str, list[str]]:
        """
        Extract a trailing `<<actions: A006(target=宏基建设); A009>>` line.
        Return (cleaned_text, hints_list).
        """
        import re
        m = re.search(r"<<actions:\s*(.*?)>>\s*$", text, flags=re.S)
        if not m:
            return text, []
        hints = [h.strip() for h in m.group(1).split(";") if h.strip()]
        cleaned = text[: m.start()].rstrip()
        return cleaned, hints
