"""
Anxin advisor — double-agent architecture.

StateManageAgent (LLM + tool calls):
  - Reads worker messages and conversation history
  - Uses structured tool calls to update case state
  - No brittle keyword matching

AdvisorAgent (LLM with state context):
  - Receives state summary from StateManageAgent
  - Uses legal knowledge + stage guidance
  - Returns specific, actionable advice

This is Anxin's architectural advantage over Doubao (vanilla LLM).
"""

from __future__ import annotations
import re

from advisor.base import Advisor, AdvisoryRequest, AdvisoryResponse
from advisor.anxin_knowledge import ANXIN_SYSTEM_PROMPT, STAGE_GUIDANCE
from advisor.anxin_state import AnxinInternalState
from advisor.state_agent import StateManageAgent
from llm.client import LLMClient


STRATEGY_PROMPT = """\
## 当前策略

记忆系统推断案件处于「{stage}」阶段。

{stage_guidance}

{state_summary}

如果对方告诉你刚才做了某个行动，你要根据结果判断下一步。
如果对方已经做了你需要他做的事，就推进到下一个阶段。
如果对方没理解，换一种更简单的方式再说一遍。
绝对不要重复已经做过的行动！"""

FINAL_SUBMISSION_HINT = """\
## 即将进入最终提交！

你必须在建议中明确告诉对方最终提交时怎么选：
1. 渠道选「劳动仲裁」（CH_ARBITRATION），不要选「直接起诉」或「行政监察」！
2. 被申请人要把所有责任主体都写上（总包单位 + 分包单位）
3. 除了欠薪，还要主张加付50%赔偿金（因为限期整改令已下达且到期未支付）
4. 把所有证据都提交上去
用大白话反复强调，让对方记住！"""


class AnxinAdvisor(Advisor):
    name = "anxin"

    def __init__(self, llm: LLMClient | None = None, system_prompt: str | None = None):
        self.llm = llm or LLMClient.from_env(role="advisor")
        self.system_prompt = system_prompt or ANXIN_SYSTEM_PROMPT
        self._chat_history: list[dict] = []
        self._internal_state = AnxinInternalState()
        self._state_agent = StateManageAgent(self.llm)

    def reset(self) -> None:
        self._chat_history = []
        self._internal_state = AnxinInternalState()

    def give_advice(self, request: AdvisoryRequest) -> AdvisoryResponse:
        # 1. StateManageAgent: single call with all context
        env_context = ""
        for entry in request.conversation_history[-5:]:
            content = entry.get("content", "")
            if not content:
                continue
            role = entry.get("role", "")
            if role == "advisor":
                env_context += f"[军师回复] {content[:300]}\n"

        combined_message = request.worker_message
        if env_context:
            combined_message += f"\n\n---\n最近的对话历史：\n{env_context}"

        self._state_agent.process(
            state=self._internal_state,
            worker_message=combined_message,
            conversation_history=request.conversation_history,
        )

        # 2. Build strategy-aware prompt using state from StateManageAgent
        stage = self._internal_state.infer_stage()
        guidance = STAGE_GUIDANCE.get(stage, STAGE_GUIDANCE["initial"])
        state_summary = self._internal_state.get_state_summary_for_prompt()

        final_hint = ""
        if stage == "arbitration_and_preservation_done" and self._internal_state.turn_count >= 7:
            final_hint = FINAL_SUBMISSION_HINT

        strategy_block = STRATEGY_PROMPT.format(
            stage=stage,
            stage_guidance=guidance["advice_focus"],
            state_summary=state_summary,
        )
        if final_hint:
            strategy_block += "\n\n" + final_hint

        full_system = self.system_prompt + "\n\n" + strategy_block

        # 3. Append worker message to history
        self._chat_history.append({"role": "user", "content": request.worker_message})

        # 4. AdvisorAgent: call LLM with enriched prompt
        messages = [
            {"role": "system", "content": full_system},
            *self._chat_history,
        ]
        raw = self.llm.chat(messages, temperature=0.4, purpose="advisor_anxin")

        # 5. Parse action hints and clean text
        cleaned, hints = self._parse_action_hints(raw)

        # 6. Generate dynamic hints from state if LLM didn't produce them
        if not hints:
            hints = self._generate_dynamic_hints(stage)

        # 7. Record assistant turn
        self._chat_history.append({"role": "assistant", "content": cleaned})

        return AdvisoryResponse(
            text=cleaned,
            suggested_action_hints=hints,
            advisor_meta={"inferred_stage": stage, "turn": self._internal_state.turn_count},
        )

    def _generate_dynamic_hints(self, stage: str) -> list[str]:
        """Generate action hints dynamically based on state and stage."""
        guidance = STAGE_GUIDANCE.get(stage, STAGE_GUIDANCE["initial"])
        base_hints = list(guidance.get("hints", []))
        state = self._internal_state

        enhanced = []
        for hint in base_hints:
            if hint == "A006" and state.get_total_contractor():
                enhanced.append(f"A006(target_company={state.get_total_contractor()})")
            elif hint == "A007":
                respondents = state.get_respondent_list()
                if respondents:
                    enhanced.append(f"A007(respondents={respondents})")
                else:
                    enhanced.append("A007")
            elif hint == "A008" and state.milestones.get("asset_preservation_applied"):
                continue  # Already done
            else:
                enhanced.append(hint)

        return enhanced

    @staticmethod
    def _parse_action_hints(text: str) -> tuple[str, list[str]]:
        """Extract trailing <<actions: ...>> hints from LLM output."""
        m = re.search(r"<<actions:\s*(.*?)>>\s*$", text, flags=re.S)
        if not m:
            return text, []
        hints = [h.strip() for h in m.group(1).split(";") if h.strip()]
        cleaned = text[:m.start()].rstrip()
        return cleaned, hints
