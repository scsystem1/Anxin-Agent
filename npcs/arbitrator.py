"""Arbitrator NPC for final labor-arbitration outcomes."""

from __future__ import annotations

from npcs.base import BaseNPC, NpcContext, NPC_OUTPUT_CONTRACT


ARBITRATOR_PROMPT = """\
你是成都市双流区劳动人事争议仲裁委员会的仲裁员。
你只在最终裁决阶段出现，根据申请书、证据清单和被申请人范围作出仲裁意见。

关键规则：
- 宏基作为施工总承包单位，分包单位拖欠农民工工资时，依《保障农民工工资支付条例》第30条先行清偿。
- 对违法发包、分包给个人或无资质单位导致欠薪的，重点适用《保障农民工工资支付条例》第36条。
- 第31条是总包代发工资制度，不要把它写成违法分包连带责任依据。
- 加付赔偿金须以劳动行政部门责令限期支付且逾期不支付为前提，比例为50%-100%。
- 手写工资结算单、转账记录、催讨记录、实名制台账可以组合证明欠薪事实和用工事实。

性格：中立、严谨，用简洁法律语言说明理由。
"""


class ArbitratorNPC(BaseNPC):
    npc_id = "arbitrator"
    display_name = "仲裁员"

    def system_prompt(self, ctx: NpcContext) -> str:
        return (
            ARBITRATOR_PROMPT
            + f"\n\n# 当前程序阶段：{ctx.procedural_stage}\n"
            + (
                "已知情况：\n  - " + "\n  - ".join(ctx.extra_facts_visible)
                if ctx.extra_facts_visible else ""
            )
            + "\n\n"
            + NPC_OUTPUT_CONTRACT
        )
