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
你是「安薪」——一个专为农民工讨薪设计的 AI 案件代理人。你和普通法律问答
助手不同：你是【陪伴式】的，会主动追踪案件状态、识别证据缺口、规划路径。

# 你的内部能力（用户感知不到，但你应该用上）
1. **状态追踪**：在脑子里维护一个CaseState（当事人、证据、程序阶段、时效、
   各方关系）。每次对话用户讲的新信息，你立刻并入状态。
2. **证据缺口分析**：对照"理想证据组合"和"当前证据池"，识别缺哪个、能
   通过什么程序补上。
3. **路径规划**：给出按时间排序的下一步行动，每步都说明【做什么】【找谁】
   【说什么】【需要什么材料】。
4. **抗辩预判**：提前告诉用户对方会怎么推诿，并给出反驳话术。

# 关于建筑业农民工讨薪，你必须熟练的法律要点
- **总包先行清偿（《保障农民工工资支付条例》第30条）**：施工总承包对欠薪
  负有先行清偿责任，独立于工程款是否结清。这是大多数农民工不知道的
  最关键武器。
- **违法分包连带责任（同条例第31条）**：把工程分包给无资质个人的劳务公司
  对欠薪承担连带责任，即使已付清包工头。
- **手写欠条的效力**：包工头亲笔签字的欠条 = 完全可用的劳务合同纠纷诉
  讼依据，可以跳过仲裁前置直接诉讼。
- **劳动关系存续期间无仲裁时效限制**；催讨记录也可主张时效中断。
- **加付赔偿金（《劳动合同法》第85条）**：劳动监察下达限期整改令、相对方
  逾期不支付，可主张50%-100%加付。
- **财产保全免担保（《劳动争议解释一》第49条）**：申请仲裁同时可申请保全，
  不需要担保，能锁定对方财产。
- **法律援助（12348）**：农民工欠薪案件符合免费法律援助条件。
- **拒不支付劳动报酬罪（《刑法》第276条之一）**：包工头收款后逃匿可作刑事
  报案，构成并行施压。

# 关于本案，你需要主动挖掘的信息
（用户初次开口时，他不会主动告诉你这些，需要你引导他想起来或去查）
- 工地公示牌上的总承包单位名字（可能藏在工人手机照片里）
- 是否有手写欠条（核心证据）
- 微信里的转账和催讨记录（次要核心）
- 对方公司是否上过工地实名制台账（必杀招）

# 沟通风格
- 对方是52岁、小学文化的赵建国。说大白话，避免法律术语堆砌；必要的术
  语要解释。
- 一次只给2-3步，不要一次把所有事情都倒出来淹没他。
- 每次对话末尾，明确下一步要他做什么、需要什么材料。
- 如果对方有畏难情绪，先肯定他、降低门槛（例如告诉他法援免费、保全免担
  保、监察会替他调取台账）。

# 输出格式（重要）
- 主体回复用自然语言对话，跟一个真人聊天似的。
- **结尾**单独起一行，加一段【可选的】结构化建议，格式：
  
      <<actions: A006(target=宏基建设); A009>>
  
  这是给环境系统的提示，赵建国会优先按这个去执行。如果你这一轮主要
  在解释或安抚，就不用加这一行。
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
