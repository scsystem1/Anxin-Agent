"""
State Management Agent — uses LLM tool calls to track case state.

This agent reads the worker's messages and uses structured tool calls
to update AnxinInternalState. It replaces brittle keyword matching
with LLM-powered information extraction.
"""

from __future__ import annotations
import json
from typing import Any, Callable

from advisor.anxin_state import AnxinInternalState
from advisor.anxin_knowledge import STATE_AGENT_PROMPT
from llm.client import LLMClient


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------
STATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_party",
            "description": "记录案件中出现的当事人（公司或个人）",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "当事人名称（如 宏基、恒达、李大海）",
                    },
                    "role": {
                        "type": "string",
                        "description": "角色：包工头/总包单位/分包单位/监察员/项目经理/工友",
                    },
                    "full_name": {
                        "type": "string",
                        "description": "完整法人名称（如 宏基建设集团股份有限公司），不确定则留空",
                    },
                },
                "required": ["name", "role"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_evidence",
            "description": "记录发现的证据",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "description": "证据编号（如 E001）",
                    },
                    "name": {
                        "type": "string",
                        "description": "证据名称（如 手写工资结算单）",
                    },
                    "proves": {
                        "type": "string",
                        "description": "该证据证明什么",
                    },
                },
                "required": ["evidence_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_action",
            "description": "记录已完成的行动",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "string",
                        "description": "行动编号（如 A001、A006、A007）",
                    },
                    "description": {
                        "type": "string",
                        "description": "行动结果简述",
                    },
                },
                "required": ["action_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_milestone",
            "description": "设置程序里程碑",
            "parameters": {
                "type": "object",
                "properties": {
                    "milestone": {
                        "type": "string",
                        "enum": [
                            "evidence_organized",
                            "queried_company_info",
                            "has_legal_aid",
                            "complained_to_inspection",
                            "limit_order_issued",
                            "arbitration_filed",
                            "asset_preservation_applied",
                            "criminal_report_filed",
                            "contacted_witness",
                        ],
                        "description": "里程碑类型",
                    },
                    "details": {
                        "type": "string",
                        "description": "补充说明（如投诉对象、被申请人等）",
                    },
                },
                "required": ["milestone"],
            },
        },
    },
]


class StateManageAgent:
    """State management agent that uses tool calls to update case state."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def process(
        self,
        state: AnxinInternalState,
        worker_message: str,
        conversation_history: list[dict],
    ) -> AnxinInternalState:
        """
        Process a worker message, update state via tool calls.

        Returns the updated state.
        """
        current_summary = state.get_state_summary_for_prompt()

        # Build recent context from conversation history
        recent = ""
        for entry in conversation_history[-4:]:
            role = entry.get("role", "")
            content = entry.get("content", "")
            if content:
                recent += f"[{role}] {content[:200]}\n"

        user_content = (
            f"## 当前状态\n{current_summary}\n\n"
            f"## 最近对话\n{recent}\n\n"
            f"## 工人最新消息\n{worker_message}\n\n"
            f"请分析以上信息，调用工具更新案件状态。"
        )

        messages = [
            {"role": "system", "content": STATE_AGENT_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # Define tool executor that updates state
        def tool_executor(name: str, args: dict) -> dict:
            if name == "record_party":
                return state.add_party(
                    args.get("name", ""),
                    args.get("role", ""),
                    args.get("full_name", ""),
                )
            elif name == "record_evidence":
                return state.add_evidence(
                    args.get("evidence_id", ""),
                    args.get("name", ""),
                    args.get("proves", ""),
                )
            elif name == "record_action":
                return state.record_action_done(
                    args.get("action_id", ""),
                    args.get("description", ""),
                )
            elif name == "set_milestone":
                return state.set_milestone(
                    args.get("milestone", ""),
                    args.get("details", ""),
                )
            return {"status": "unknown_tool"}

        try:
            self.llm.chat_with_tools(
                messages=messages,
                tools=STATE_TOOLS,
                tool_executor=tool_executor,
                temperature=0.2,
                purpose="state_agent",
            )
        except Exception as e:
            # If tool calling fails, just increment turn count
            state.turn_count += 1
            print(f"[Warn] state_agent tool call failed: {e}", file=__import__("sys").stderr)

        return state
