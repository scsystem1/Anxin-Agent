"""李大海 - 包工头, mostly unreachable."""

from __future__ import annotations
from npcs.base import BaseNPC, NpcContext, NpcResponse


LI_DAHAI_PROMPT = """\
你是李大海，52岁，四川南充人，某工地的包工头，没有任何资质。
你欠了赵建国（同乡，砌体工）7.66万元工资。

事实背景（你知道但绝不主动说）：
- 2023年12月5日，恒达劳务已经把14个工人的全部劳务款 1,427,000 元打到你
  账户上了，包括赵建国应得的那部分
- 你拿到钱后没有发给工人，跑到湖南长沙宁乡灰汤镇某工地继续做事
- 你换了新手机号 137-XXXX-8888，旧号 158-XXXX-3456 已经停机
- 你名下有一辆四川牌照的汽车

你现在的处境：
- 旧号停机，赵建国理论上联系不到你
- 如果赵建国通过新号或公安渠道找到你，你会推诿，不会承认拿到了钱
- 典型推诿话术："甲方工程款还没结，我哪有钱给你" / "等我这边工程结束就给"
- 施压后可能承认欠款事实，但坚决拒绝立刻支付

刑事风险（你心里清楚）：
- 你已收款后逃匿，符合《刑法》第276条之一"拒不支付劳动报酬罪"

性格：
- 油滑，会装可怜
- 对法律细节一知半解，怕公安和法院
- 听到"刑事报案"会紧张，但也不会立刻松口
"""


class LiDahaiNPC(BaseNPC):
    npc_id = "li_dahai"
    display_name = "李大海（包工头）"

    def system_prompt(self, ctx: NpcContext) -> str:
        from npcs.base import NPC_OUTPUT_CONTRACT
        return (
            LI_DAHAI_PROMPT
            + f"\n\n# 当前对话上下文\n压力等级：{ctx.pressure_level}/3\n"
            + f"程序阶段：{ctx.procedural_stage}\n"
            + (
                "已知特殊情况：\n  - " + "\n  - ".join(ctx.extra_facts_visible)
                if ctx.extra_facts_visible
                else ""
            )
            + "\n\n"
            + NPC_OUTPUT_CONTRACT
        )

    def respond(self, ctx: NpcContext) -> NpcResponse:
        """Override: if context indicates contact via OLD phone, return UNREACHABLE."""
        contacted_via = ctx.extra_facts_visible
        if "via_old_phone" in contacted_via:
            return NpcResponse(text="（电话提示音：您拨打的电话已停机，请稍后再拨）")
        if "via_new_phone_unknown" in contacted_via:
            return NpcResponse(text="（电话无法接通——你不知道他的新号码）")
        return super().respond(ctx)
