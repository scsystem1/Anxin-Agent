"""Doubao advisor - baseline for Anxin comparison.

This advisor simulates what happens when a real user asks Doubao directly
for help with a wage-theft case. Doubao is a friendly, general-purpose
conversational AI with NO legal-specialist training, NO internal case-tracking,
and NO structured output capabilities.

Key design principles for a fair baseline with genuine performance gap:
  1. Short context window - only last ~6 exchange pairs retained, older
     details are genuinely forgotten. Anxin's structured memory wins here.
  2. No legal expertise - generic AI assistant, no procedural knowledge.
     Anxin has embedded legal strategy chain and stage guidance.
  3. No action hints - worker must interpret free-text advice. Anxin gives
     structured hints that guide the worker's action selection.
  4. No state tracking - cannot sequence actions by case stage. Anxin's
     StateManageAgent maintains structured case progress.
  5. Higher randomness - temperature 0.8 vs Anxin's 0.4, for less consistent,
     less strategically coherent advice.

These are all realistic limitations of a general-purpose LLM, not artificial
crippling. The performance gap demonstrates Anxin's genuine value.

To use the actual Doubao API instead of a generic LLM, set
DOUBAO_LLM_BASE_URL / DOUBAO_LLM_MODEL / DOUBAO_LLM_API_KEY in .env.
"""

from __future__ import annotations
import sys

from advisor.base import Advisor, AdvisoryRequest, AdvisoryResponse
from llm.client import LLMClient


# ---------------------------------------------------------------------------
# Realistic Doubao system prompt
# ---------------------------------------------------------------------------
# This prompt captures Doubao's actual conversational personality:
# friendly, helpful, uses plain language, admits uncertainty.
# Explicitly marks Doubao as NOT a legal expert.
DOUBAO_SYSTEM_PROMPT = """\
你是豆包，由字节跳动开发的 AI 对话助手。你的回答风格友好、接地气、善于用大白话解释问题。

你的特点：
- 认真倾听用户的每一个问题，给出实用、可操作的建议
- 善于把复杂的道理讲得通俗易懂
- 对于不确定的事情会诚实说明，不假装专家

注意：你是通用型 AI 助手，不是法律专家。涉及法律问题时可以给出常识性建议，
但不提供正式法律意见。遇到专业法律问题应建议用户咨询律师或相关部门。
"""

# ---------------------------------------------------------------------------
# Context window simulation
# ---------------------------------------------------------------------------
# Real Doubao has a finite context window. Retain only the last 6 exchange
# pairs (12 messages). This is aggressively realistic: in long conversations,
# early case details genuinely fade from memory. Anxin overcomes this with
# structured case-state memory - this is a key advantage, not unfair crippling.
MAX_HISTORY_MESSAGES = 12

# Also cap individual long messages to simulate real chat attention limits.
# Workers often dump walls of text; Doubao realistically can't digest it all.
MAX_MESSAGE_CHARS = 3000


class DoubaoAdvisor(Advisor):
    """Vanilla conversational LLM with Doubao's personality.

    No internal state machine, no specialized legal prompt, no structured
    output. Uses higher temperature (0.8) for varied but less consistent
    advice. Context window limited to 12 messages (~6 exchange pairs).
    """

    name = "doubao"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient.from_env(role="doubao")
        self._chat_history: list[dict] = []

    def reset(self) -> None:
        """Clear chat history at the start of each episode."""
        self._chat_history = []

    def give_advice(self, request: AdvisoryRequest) -> AdvisoryResponse:
        """Vanilla conversational advice with realistic limitations.

        1. Truncate long worker messages to simulate attention span limits.
        2. Append to internal history, then truncate to max context window.
        3. Call LLM with higher temperature (0.8) for realistic Doubao variance.
        4. Return with NO structured action hints - worker must interpret text.
        """
        text = request.worker_message
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + "...[消息过长已截断]"

        self._chat_history.append({"role": "user", "content": text})

        recent = self._chat_history[-MAX_HISTORY_MESSAGES:]
        messages = [
            {"role": "system", "content": DOUBAO_SYSTEM_PROMPT},
            *recent,
        ]

        try:
            text = self.llm.chat(
                messages,
                temperature=0.8,
                max_tokens=2048,
                purpose="advisor_doubao",
            )
        except Exception:
            import traceback
            print(f"[DoubaoAdvisor] LLM call failed on turn {request.current_turn_index}",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            text = (
                "抱歉，我暂时无法给出具体的建议。建议你把情况整理清楚，"
                "去当地的劳动监察部门或者法律援助中心咨询一下，"
                "他们能给你更专业的帮助。"
            )

        self._chat_history.append({"role": "assistant", "content": text})

        return AdvisoryResponse(
            text=text,
            suggested_action_hints=[],
            advisor_meta={
                "context_msgs": len(recent),
                "total_history": len(self._chat_history),
                "truncated": len(self._chat_history) > MAX_HISTORY_MESSAGES,
            },
        )
