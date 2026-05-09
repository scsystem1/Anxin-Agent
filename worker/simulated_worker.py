"""
Simulated worker (赵建国).

THE EXECUTOR. This is the LLM-driven actor inside the environment that:
  1. receives observations from the env
  2. translates them into口语化 help requests for the advisor
  3. receives advice
  4. selects an action from the available action space, with the right params

The KEY DESIGN PRINCIPLE: this worker's executive function is INTENTIONALLY
LIMITED. Its persona is "小学文化、听话、不主动思考". This means:
  - When advice is vague ("保留好证据"), the worker picks the safest /
    most surface-level action and may forget important parameters.
  - When advice is specific and actionable ("打开微信，搜索'工资'，把所
    有截图保存到一个文件夹"), the worker executes it competently with
    the correct action and full parameters.

This asymmetry is what reveals the difference between Anxin (specific,
state-aware advice) and Doubao (vague, generic advice). It is NOT cheating —
it models how real low-literacy migrant workers actually behave.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from llm.client import LLMClient
from environment.actions import Action, ActionSpec

if TYPE_CHECKING:
    from environment.env import Observation


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

WORKER_PERSONA = """\
你叫赵建国，52岁，四川南充人，在建筑工地做砌体工20多年。你只读到小学，识字
但不会用复杂词汇。你不懂法律术语，听到"先行清偿""仲裁时效""财产保全"这种
词会发愣。你性格老实，听人劝。你现在很急，因为李大海欠你7万多块钱跑了，
家里等钱用。你打开了一个叫"安薪"的App，想找个懂行的人帮你出主意。

你的行为方式：
- 你只描述你看到的、感受到的事情，不会自己加工成法律分析
- 你说话用大白话，会说"那个老板""那个公司""那个写欠条的"
- 别人让你做的事，你只能照字面意思理解。如果他说"保留好证据"，你只会
  点点头，不知道具体要做什么；如果他说"打开微信，搜'工资'两个字，把
  所有相关的截图都存起来"，你才能照做
- 你不会主动想到对方没提的事
- 面对官方机构（劳动监察、法院）会紧张，但有人陪/有具体指导就好
- 如果军师建议看起来太难、太贵或者听不懂，你会犹豫，可能想"算了"
"""


REQUEST_FORMULATION_PROMPT = """\
{persona}

# 当前情境
今天是模拟时间第 {day} 天（{date}）。
{recent_events}

# 你最近做了什么 / 发生了什么
{action_history_summary}

# 你心里想找军师问的问题
请用你自己的口吻（白话、不带法律术语），给军师写一句求助消息。
- 第一次开口：把整个情况大致讲一下
- 之后的每次：基于刚发生的事说一句简短的求助/汇报，问军师下一步怎么办

直接输出你要发的消息，不要任何前后缀。
"""


ACTION_SELECTION_PROMPT = """\
{persona}

# 军师刚才告诉你
\"\"\"
{advice_text}
\"\"\"

# 你现在可以做的事（行动清单）
你只能从下面这个清单里选一个去做。每个行动有编号、名字和需要填的参数。
{action_menu}

# 重要：你的执行力受军师建议的影响
- 如果军师的建议很具体（说清楚做什么、找谁、说什么），就照着选最对应的行动，
  并把参数填全
- 如果军师的建议很模糊（只说"保留证据""走法律程序"这种话），你不知道该
  怎么办，只能选最简单、最被动的行动（比如A001整理证据、A011联系王主任），
  参数也只能填最直接的
- 如果军师没给清晰下一步，但你之前听过的话里有蛛丝马迹（比如知道"宏基"
  这个名字），你可以选A011联系或A006投诉，但很可能填错对象
- 如果军师让你做的事在清单里没有对应项，选最接近的；实在没有就选A001
- 如果军师明确建议你放弃，或者你已经第三次被劝放弃，选A099

输出严格 JSON（不要 markdown 代码块，不要解释）：
{{
  "action_id": "Axxx",
  "parameters": {{...}},
  "reasoning_in_worker_voice": "我决定这么做是因为..."
}}
"""


@dataclass
class WorkerRequest:
    """A help request from the worker to the advisor."""
    text: str


@dataclass
class WorkerActionChoice:
    """The worker's selected action with parameters and inner monologue."""
    action: Action
    reasoning: str


# ---------------------------------------------------------------------------
# SimulatedWorker class
# ---------------------------------------------------------------------------

class SimulatedWorker:
    """LLM-backed simulated worker. Stateless across method calls — all state
    lives in the env / case_state."""

    def __init__(self, persona_overrides: str = "", llm: LLMClient | None = None):
        self.persona = WORKER_PERSONA
        if persona_overrides:
            self.persona += "\n\n# 额外人设说明\n" + persona_overrides
        self.llm = llm or LLMClient.from_env(role="worker")

    # ------------------------------------------------------------------
    # Step 1: turn observation into a help request to the advisor
    # ------------------------------------------------------------------
    def formulate_request(self, observation: "Observation") -> WorkerRequest:
        """
        Translate the env observation into a口语化 message for the advisor.

        Codex: build the prompt by filling REQUEST_FORMULATION_PROMPT with:
            - persona = self.persona
            - day = observation.day
            - date = observation.date.isoformat()
            - recent_events = observation.format_recent_events()
            - action_history_summary = observation.format_action_history(last_n=3)

        Then call self.llm.chat(...) and wrap in WorkerRequest(text=response).
        """
        user_prompt = REQUEST_FORMULATION_PROMPT.format(
            persona=self.persona,
            day=observation.day,
            date=observation.date.isoformat(),
            recent_events=observation.format_recent_events(),
            action_history_summary=observation.format_action_history(last_n=3),
        )
        text = self.llm.chat(
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.7,
            purpose="worker_request",
        ).strip()
        return WorkerRequest(text=text)

    # ------------------------------------------------------------------
    # Step 2: turn advice into a structured action choice
    # ------------------------------------------------------------------
    def choose_action(
        self,
        advice_text: str,
        available_actions: list[ActionSpec],
        advisor_hints: list[str] | None = None,
    ) -> WorkerActionChoice:
        """
        Pick an action from the available menu, given the advisor's advice.

        Codex implementation:
            1. Build action_menu string from available_actions with id, name,
               required parameters. Include one-line guidance for each.
            2. If advisor_hints are present, prepend them to the menu as
               "推荐选项（来自军师的结构化建议）".
            3. Fill ACTION_SELECTION_PROMPT and call self.llm.chat_json().
            4. Validate the returned action_id is in the menu; if not, fall
               back to A001 and log a warning.
            5. Return WorkerActionChoice(action=Action(...), reasoning=...).

        IMPORTANT: do NOT post-process the LLM's parameter choice to "fix" it.
        If the worker picks the wrong target_company, that mistake is the
        whole point — it reflects the advisor's vagueness.
        """
        menu_lines = []
        for spec in available_actions:
            params_hint = (
                f" [需填: {', '.join(spec.parameters_required)}]"
                if spec.parameters_required else ""
            )
            menu_lines.append(f"  - [{spec.id}] {spec.name}{params_hint}")
        menu = "\n".join(menu_lines)

        if advisor_hints:
            hint_block = (
                "\n# 来自军师的结构化建议（你应该优先考虑）\n"
                + "\n".join(f"  → {h}" for h in advisor_hints)
            )
            menu = hint_block + "\n\n# 完整可选行动\n" + menu

        prompt = ACTION_SELECTION_PROMPT.format(
            persona=self.persona,
            advice_text=advice_text,
            action_menu=menu,
        )
        parsed = self.llm.chat_json(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            purpose="worker_action",
        )

        action_id = parsed.get("action_id", "A001")
        valid_ids = {s.id for s in available_actions}
        if action_id not in valid_ids:
            print(
                f"[Warn] worker chose invalid action {action_id}; falling back to A001",
                file=sys.stderr,
            )
            action_id = "A001"

        parameters = parsed.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        return WorkerActionChoice(
            action=Action(action_id=action_id, parameters=parameters),
            reasoning=parsed.get("reasoning_in_worker_voice", ""),
        )
