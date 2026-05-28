"""
Anxin advisor — the structured case agent.

This is the core differentiator. Unlike Doubao (a vanilla LLM), Anxin:
  1. Has domain-specific legal knowledge embedded in its prompt
  2. Maintains internal case state inferred from worker messages
  3. Gives ultra-specific, actionable advice in plain language
  4. Emits structured action hints (<<actions: ...>>) to guide the worker
  5. Follows an optimal procedural strategy
"""

from __future__ import annotations

from advisor.base import Advisor, AdvisoryRequest, AdvisoryResponse
from advisor.anxin_knowledge import ANXIN_SYSTEM_PROMPT, STAGE_GUIDANCE
from advisor.anxin_state import AnxinInternalState
from llm.client import LLMClient


STRATEGY_PROMPT = """\
## 你现在的策略指导

根据当前案件阶段「{stage}」，你应该重点做这些事：

{stage_guidance}

如果赵建国告诉你刚才做了某个行动，你要根据结果判断下一步。
如果赵建国已经做了你需要他做的事，就推进到下一个阶段。
如果赵建国没理解，换一种更简单的方式再说一遍。
"""


class AnxinAdvisor(Advisor):
    name = "anxin"

    def __init__(self, llm: LLMClient | None = None, system_prompt: str | None = None):
        self.llm = llm or LLMClient.from_env(role="advisor")
        self.system_prompt = system_prompt or ANXIN_SYSTEM_PROMPT
        self._chat_history: list[dict] = []
        self._internal_state = AnxinInternalState()

    def reset(self) -> None:
        self._chat_history = []
        self._internal_state = AnxinInternalState()

    def give_advice(self, request: AdvisoryRequest) -> AdvisoryResponse:
        # 1. Update internal state from worker's message
        self._internal_state.update_from_worker_message(request.worker_message)

        # Also learn from conversation history (action results narrated by worker)
        for entry in request.conversation_history[-3:]:
            content = entry.get("content", "")
            if entry.get("role") == "worker":
                self._internal_state.update_from_action_result(content)

        # 2. Build strategy-aware system prompt
        stage = self._internal_state.infer_stage()
        guidance = STAGE_GUIDANCE.get(stage, STAGE_GUIDANCE["initial"])
        state_summary = self._internal_state.get_state_summary_for_prompt()
        strategy_block = STRATEGY_PROMPT.format(
            stage=stage,
            stage_guidance=guidance["advice_focus"],
        )

        full_system = self.system_prompt + "\n\n" + state_summary + "\n\n" + strategy_block

        # 3. Append worker message to history
        self._chat_history.append({"role": "user", "content": request.worker_message})

        # 4. Call LLM with enriched prompt
        messages = [
            {"role": "system", "content": full_system},
            *self._chat_history,
        ]
        raw = self.llm.chat(messages, temperature=0.4, purpose="advisor_anxin")

        # 5. Parse action hints and clean text
        cleaned, hints = self._parse_action_hints(raw)

        # 6. If LLM didn't produce hints but we know what to suggest, inject them
        if not hints:
            hints = self._generate_stage_hints(stage)

        # 7. Record assistant turn
        self._chat_history.append({"role": "assistant", "content": cleaned})

        # 8. Update state from our own advice (for next turn's context)
        self._internal_state.update_from_action_result(cleaned)

        return AdvisoryResponse(
            text=cleaned,
            suggested_action_hints=hints,
            advisor_meta={"inferred_stage": stage, "turn": self._internal_state.turn_count},
        )

    def _generate_stage_hints(self, stage: str) -> list[str]:
        """Generate action hints based on inferred stage when LLM doesn't provide them."""
        guidance = STAGE_GUIDANCE.get(stage, STAGE_GUIDANCE["initial"])
        return list(guidance.get("hints", []))

    @staticmethod
    def _parse_action_hints(text: str) -> tuple[str, list[str]]:
        """Extract trailing <<actions: ...>> hints from LLM output."""
        import re
        m = re.search(r"<<actions:\s*(.*?)>>\s*$", text, flags=re.S)
        if not m:
            return text, []
        hints = [h.strip() for h in m.group(1).split(";") if h.strip()]
        cleaned = text[:m.start()].rstrip()
        return cleaned, hints
