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
import re
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


INCREMENTAL_REQUEST_FORMULATION_PROMPT = """\
{persona}

# 你和军师已经聊过前面的案情
军师能看到之前的聊天记录，所以你这次不要重复最开始的欠薪背景、欠条、李大海跑路这些旧话。

# 刚刚发生的新进展
今天是模拟时间第 {day} 天（{date}）。
{latest_event}

# 你最近做过的动作
{action_history_summary}

# 你这次要发给军师的话
只用赵建国的白话，说清楚“刚刚发生了什么 / 对方怎么回 / 我现在卡在哪里”，然后问下一步。
要求：
- 只说增量信息，不要从头复述案情
- 如果刚拿到新证据，就点名说拿到了什么
- 如果刚和NPC沟通，就说对方具体怎么推诿或答复
- 1到3句话即可

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


FINAL_SUBMISSION_PROMPT = """\
{persona}

# 军师刚才给你的最终建议
\"\"\"
{advice_text}
\"\"\"

# 你现在手头的证据
{evidence_summary}

# 可选的维权渠道
{channels_menu}

# 你的任务
根据军师的建议，选择一个最终渠道，并整理要提交的证据、被申请人/被告和一份白话文书。
如果军师没有说清楚，就选择最稳妥、你最能理解的渠道；不要编造自己没有的证据。

输出严格 JSON（不要 markdown 代码块）：
{{
  "channel_id": "CH_ARBITRATION",
  "channel_name": "劳动仲裁",
  "advisor_reasoning": "我为什么听军师选这个渠道",
  "respondents": ["宏基建设集团股份有限公司", "成都恒达劳务有限公司"],
  "evidence_ids_submitted": ["E001", "E002"],
  "drafted_documents": [
    {{
      "doc_type": "仲裁申请书",
      "content": "申请人赵建国……请求支付拖欠工资……"
    }}
  ]
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
        if observation.actions_taken_summary:
            latest_event = (
                f"- {observation.recent_events[-1]}"
                if observation.recent_events else "- （暂无新进展）"
            )
            user_prompt = INCREMENTAL_REQUEST_FORMULATION_PROMPT.format(
                persona=self.persona,
                day=observation.day,
                date=observation.date.isoformat(),
                latest_event=latest_event,
                action_history_summary=observation.format_action_history(last_n=3),
            )
        else:
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
        try:
            parsed = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                purpose="worker_action",
            )
        except Exception as e:
            print(f"[Warn] worker_action LLM failed; using heuristic fallback: {e}", file=sys.stderr)
            return self._fallback_action_choice(advice_text, available_actions, advisor_hints)

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

    def _fallback_action_choice(
        self,
        advice_text: str,
        available_actions: list[ActionSpec],
        advisor_hints: list[str] | None = None,
    ) -> WorkerActionChoice:
        """Deterministic backup when the worker action LLM call fails."""
        valid = {s.id for s in available_actions}
        text = advice_text or ""

        hint_text = "; ".join(advisor_hints or [])
        for candidate in re.findall(r"A\d{3}", hint_text):
            if candidate in valid:
                return WorkerActionChoice(
                    action=Action(candidate, parameters=self._default_params(candidate, text + hint_text)),
                    reasoning="行动选择模型暂时失败，我按军师给的结构化提示照做。",
                )

        rules = [
            (("财产保全", "冻结"), "A008"),
            (("仲裁", "仲裁委"), "A007"),
            (("劳动监察", "监察", "投诉"), "A006"),
            (("法律援助", "12348", "法援"), "A009"),
            (("法院", "起诉", "欠条"), "A013"),
            (("宏基", "总包", "张国华"), "A018"),
            (("恒达", "王主任", "王培"), "A011"),
            (("李大海", "包工头", "电话"), "A017"),
            (("公示牌", "拍照"), "A019"),
            (("证据", "截图", "微信", "转账", "整理"), "A001"),
        ]
        for keywords, action_id in rules:
            if action_id in valid and any(k in text for k in keywords):
                return WorkerActionChoice(
                    action=Action(action_id, parameters=self._default_params(action_id, text)),
                    reasoning="行动选择模型暂时失败，我按军师话里的关键词选择最接近的行动。",
                )

        fallback_id = "A001" if "A001" in valid else (available_actions[0].id if available_actions else "A099")
        return WorkerActionChoice(
            action=Action(fallback_id, parameters=self._default_params(fallback_id, text)),
            reasoning="行动选择模型暂时失败，我先做当前最稳妥的一步。",
        )

    def _default_params(self, action_id: str, text: str) -> dict[str, Any]:
        if action_id == "A006":
            target = "宏基建设集团股份有限公司" if "宏基" in text or "总包" in text else "成都恒达劳务有限公司"
            return {"target_company": target}
        if action_id == "A007":
            respondents = ["宏基建设集团股份有限公司", "成都恒达劳务有限公司"]
            if "李大海" in text:
                respondents.append("李大海")
            return {"respondents": respondents}
        if action_id == "A011":
            return {"message": "王主任，我是赵建国，李大海欠我工资，你们恒达是劳务公司，请帮我处理。"}
        if action_id == "A017":
            return {"contact_method": "old_phone", "message": "李哥，我是赵建国，你欠我的工资什么时候给？"}
        if action_id == "A018":
            return {"message": "张经理，我在天骄名苑干活被欠薪，公示牌上总包是宏基，请你们处理。"}
        return {}

    def formulate_final_submission(
        self,
        advice_text: str,
        evidence_summary: str,
        channels: list[dict],
    ) -> dict[str, Any]:
        """Turn the final advisor advice into structured submission params."""
        channels_menu = "\n".join(
            f"  [{c.get('id')}] {c.get('name')} - {c.get('description')}"
            for c in channels
        )
        prompt = FINAL_SUBMISSION_PROMPT.format(
            persona=self.persona,
            advice_text=advice_text,
            evidence_summary=evidence_summary,
            channels_menu=channels_menu,
        )
        try:
            parsed = self.llm.chat_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                purpose="worker_final_submission",
            )
        except Exception as e:
            print(f"[Warn] worker_final_submission LLM failed; using fallback: {e}", file=sys.stderr)
            evidence_ids = re.findall(r"E\d{3}", evidence_summary)
            return {
                "channel_id": "CH_ARBITRATION",
                "channel_name": "劳动仲裁",
                "advisor_reasoning": "最终提交生成模型暂时失败，我按已整理证据选择劳动仲裁。",
                "respondents": ["宏基建设集团股份有限公司", "成都恒达劳务有限公司"],
                "evidence_ids_submitted": evidence_ids,
                "drafted_documents": [
                    {
                        "doc_type": "仲裁申请书",
                        "content": "申请人赵建国，请求被申请人支付拖欠工资76600元，并依法承担相应责任。提交证据以现有证据清单为准。",
                    }
                ],
            }
        if not isinstance(parsed, dict):
            return {}
        return parsed
