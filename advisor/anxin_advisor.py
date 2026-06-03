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

## Turn Budget
- 当前是第{turn_number}/{max_turns}轮
- 本轮回复后剩余轮次：{remaining_turns}
- Is final advice turn: {is_final_turn}

记忆系统推断案件处于「{stage}」阶段。

{stage_guidance}

{state_summary}

请像法官审查案卷一样先判断：当前事实是否真的支持进入下一阶段，还是只是对方口头受理、要求补材料、或者工人误把分包当成总包。遇到这些陷阱时，要在建议里先纠偏，再指导下一步；不要把未经确认的程序进展当成已经完成。

在给建议前先完成三个内部判断，不要把判断过程原样输出：
1. 事实边界：工人实际做成了什么，哪些只是你上一轮建议过。
2. 主体边界：当前是否已经确认总包、分包、包工头各是谁。
3. 程序边界：当前是否有正式受理、整改令、立案、保全等可验证结果。

如果某个边界不清，下一步优先做“最小核实/补正动作”，而不是继续推进到更后面的法律程序。
主体边界不清时尤其要保守：不要把“工人找过某公司”“某公司推给另一家公司”“监察员口头说也加上某公司”“工人自己说好像某公司是总包”当成总包确认。
只有以下来源能锁定主体角色：项目公示牌、施工许可证、实名制台账、劳动监察书面材料、工商/住建查询结果。
如果工人把总包和分包说反了，先温和纠偏，并让他做一个很小的确认动作：拍清楚公示牌，或把公示牌上“施工总承包单位：____；劳务分包单位：____”逐字念给你。
在主体未锁定前，不要输出带具体公司的 A006(target_company=...)；可以建议先核实主体，避免 worker 顺着错误 hints 投诉错对象。
最终提交前必须做“主体-金额-渠道”三项复核：总包/分包角色有没有可靠来源，欠薪金额是否沿用已确认金额，被申请人是否覆盖总包和分包。

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
        if request.is_final_turn:
            final_hint = FINAL_SUBMISSION_HINT
        elif stage == "arbitration_and_preservation_done" and self._internal_state.turn_count >= 7:
            final_hint = FINAL_SUBMISSION_HINT

        strategy_block = STRATEGY_PROMPT.format(
            turn_number=request.current_turn_index + 1,
            max_turns=request.max_turns or "?",
            remaining_turns=request.remaining_turns,
            is_final_turn="YES" if request.is_final_turn else "NO",
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
            if hint == "A006":
                # Keep the fallback hint conservative. Subject identity can be
                # misreported by the worker, so only the AdvisorAgent's explicit
                # text/hints should bind A006 to a concrete company.
                enhanced.append("A006")
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
