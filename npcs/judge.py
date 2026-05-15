"""Court judge NPC for final lawsuit outcomes."""

from __future__ import annotations

from npcs.base import BaseNPC, NpcContext, NPC_OUTPUT_CONTRACT


JUDGE_PROMPT = """\
你是成都市双流区人民法院的民事法官，审理建筑业农民工欠薪案件。
你只在最终裁判阶段出现，根据起诉状、证据清单和被告范围作出判决意见。

关键规则：
- 劳动者以工资欠条直接起诉且请求只涉及拖欠劳动报酬的，可按普通民事纠纷受理。
- 施工总承包单位对分包单位拖欠农民工工资承担《保障农民工工资支付条例》第30条下的先行清偿责任。
- 对违法发包、分包给个人或无资质单位导致欠薪的，重点适用《保障农民工工资支付条例》第36条。
- 第31条是总包代发工资制度，不要把它写成违法分包连带责任依据。
- 财产保全符合条件时可减免担保；刑事报案部分另案处理。

性格：公正严肃，会用工人能听懂的话解释裁判结果。
"""


class JudgeNPC(BaseNPC):
    npc_id = "judge"
    display_name = "法官"

    def system_prompt(self, ctx: NpcContext) -> str:
        return (
            JUDGE_PROMPT
            + f"\n\n# 当前程序阶段：{ctx.procedural_stage}\n"
            + (
                "已知情况：\n  - " + "\n  - ".join(ctx.extra_facts_visible)
                if ctx.extra_facts_visible else ""
            )
            + "\n\n"
            + NPC_OUTPUT_CONTRACT
        )
