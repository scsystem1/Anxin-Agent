"""
Base NPC class.

Each NPC is an LLM-driven character with:
  - a persona / role
  - a knowledge boundary (what they know vs don't know)
  - default deflection patterns
  - rules for how their behavior changes under pressure
  - a list of evidence they CAN reveal (only under certain conditions)

NPCs do NOT see CaseState directly. The NPC manager passes them just the
relevant slice — the worker's question, the current pressure level, what
the worker has already extracted from them — formatted as a system prompt.
"""

from __future__ import annotations
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llm.client import LLMClient

if TYPE_CHECKING:
    from environment.state import CaseState


@dataclass
class NpcContext:
    """The minimal context an NPC sees when responding."""
    worker_message: str
    pressure_level: int                # 0 = no pressure; ↑ as procedures escalate
    procedural_stage: str
    prior_exchanges: list[dict] = field(default_factory=list)
    # ↑ [{"worker": "...", "npc": "..."}, ...]
    extra_facts_visible: list[str] = field(default_factory=list)
    # ↑ e.g. ["劳动监察已下达限期整改令"], computed by NpcManager


@dataclass
class NpcResponse:
    text: str
    new_evidence_ids: list[str] = field(default_factory=list)
    state_hints: dict = field(default_factory=dict)
    # ↑ semantic flags an NPC may surface, e.g.
    #   {"defense_raised": "已付清恒达工程款", "willing_to_settle": True}


class BaseNPC(ABC):
    """Abstract NPC."""

    npc_id: str = ""
    display_name: str = ""

    def __init__(self, npc_data: dict, llm: LLMClient | None = None):
        self.data = npc_data
        self.llm = llm or LLMClient.from_env(role="npc")

    @abstractmethod
    def system_prompt(self, ctx: NpcContext) -> str:
        """Build the NPC's system prompt for this turn."""
        ...

    def respond(self, ctx: NpcContext) -> NpcResponse:
        """
        Default response flow. Subclasses can override for special handling
        (e.g. li_dahai who is unreachable on the old phone).

        Codex implementation:
            1. system = self.system_prompt(ctx)
            2. messages = [
                 {"role": "system", "content": system},
                 ...flatten ctx.prior_exchanges into role/content turns...,
                 {"role": "user", "content": ctx.worker_message},
               ]
            3. text = self.llm.chat(messages=messages, temperature=0.7,
                                    purpose=f"npc_{self.npc_id}")
            4. Parse for any structured flags (see PROMPT_OUTPUT_CONTRACT below)
            5. Return NpcResponse(text=cleaned_text, ...)
        """
        system = self.system_prompt(ctx)
        history: list[dict[str, str]] = []
        for ex in ctx.prior_exchanges:
            history.append({"role": "user", "content": ex.get("worker", "")})
            history.append({"role": "assistant", "content": ex.get("npc", "")})

        messages = [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": ctx.worker_message},
        ]
        raw = self.llm.chat(
            messages=messages,
            temperature=0.7,
            purpose=f"npc_{self.npc_id}",
        )

        new_evidence = []
        m = re.search(r"<<reveal:\s*(E\d+)\s*>>", raw)
        if m:
            new_evidence = [m.group(1)]
            raw = re.sub(r"<<reveal:\s*E\d+\s*>>", "", raw).strip()

        return NpcResponse(text=raw, new_evidence_ids=new_evidence)


# ---------------------------------------------------------------------------
# Output contract for NPCs (used by every NPC's system prompt)
# ---------------------------------------------------------------------------

NPC_OUTPUT_CONTRACT = """\
你只输出一段对话回复（中文），不要任何解释或说明。
如果你想揭露某个事实（比如出示一份文件、承认一笔款项），在回复末尾追加
一行特殊标记：

    <<reveal: EXXX>>

EXXX 是证据编号。一次最多揭露一个。如果你不想揭露任何东西，不要加这一行。
"""
