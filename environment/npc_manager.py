"""
NPC manager.

Loads all NPC instances and routes worker interactions to the right one.
Computes per-NPC pressure level based on overall case state, and surfaces
the right "extra_facts_visible" slice for each NPC's prompt.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from npcs.base import BaseNPC, NpcContext, NpcResponse
from npcs.li_dahai import LiDahaiNPC
from npcs.wang_pei import WangPeiNPC
from npcs.zhang_guohua import ZhangGuohuaNPC
from npcs.chen_wei import ChenWeiNPC
from npcs.arbitrator import ArbitratorNPC
from npcs.judge import JudgeNPC

if TYPE_CHECKING:
    from environment.state import CaseState


NPC_REGISTRY: dict[str, type[BaseNPC]] = {
    "li_dahai": LiDahaiNPC,
    "wang_pei": WangPeiNPC,
    "zhang_guohua": ZhangGuohuaNPC,
    "chen_wei": ChenWeiNPC,
    "arbitrator": ArbitratorNPC,
    "judge": JudgeNPC,
}


class NpcManager:
    def __init__(self, npc_data_list: list[dict], legal_knowledge_pack: dict | None = None):
        # npc_data_list comes from case_json["npcs"]
        self.npcs: dict[str, BaseNPC] = {}
        self.legal_knowledge_pack = legal_knowledge_pack or {}
        for npc_data in npc_data_list:
            cls = NPC_REGISTRY.get(npc_data["id"])
            if cls is None:
                raise ValueError(f"Unknown NPC id: {npc_data['id']}")
            self.npcs[npc_data["id"]] = cls(npc_data)

    def get(self, npc_id: str) -> BaseNPC:
        if npc_id not in self.npcs:
            cls = NPC_REGISTRY.get(npc_id)
            if cls is None:
                raise KeyError(npc_id)
            self.npcs[npc_id] = cls({"id": npc_id})
        return self.npcs[npc_id]

    def interact(
        self,
        npc_id: str,
        worker_message: str,
        state: "CaseState",
        contact_method: str | None = None,
    ) -> NpcResponse:
        """
        Have the worker interact with an NPC. Builds the NpcContext from
        case state, calls the NPC, returns the response.

        Codex implementation:
            1. Compute pressure_level for this NPC based on state.
               See `_compute_pressure()` below.
            2. Compute extra_facts_visible (e.g. if state.flags.get(
               "labor_inspection_order_issued"), zhang_guohua and wang_pei
               see "劳动监察已下达限期整改令"; if contact_method ==
               "old_phone" for li_dahai, append "via_old_phone").
            3. Pull the last 2-3 prior_exchanges with this NPC from
               state.npc_interactions.
            4. Build NpcContext, call npc.respond(ctx).
            5. If response carries `<<reveal: EXXX>>`, parse and add the
               evidence to state via the action handler (this method only
               returns the NpcResponse; the caller updates state).
        """
        npc = self.get(npc_id)
        pressure = self._compute_pressure(npc_id, state)
        state.npc_pressure_level[npc_id] = pressure

        extras: list[str] = []
        if state.flags.get("limit_order_issued"):
            extras.append("劳动监察已下达限期整改令")
        if state.flags.get("asset_freeze_active"):
            extras.append("仲裁立案 + 财产保全已生效")
        if state.flags.get("criminal_case_filed"):
            extras.append("已对李大海提起拒不支付劳动报酬罪报案")

        if npc_id == "li_dahai" and contact_method == "old_phone":
            extras.append("via_old_phone")
        if npc_id == "li_dahai" and contact_method != "new_phone":
            extras.append("via_new_phone_unknown")
        extras.extend(self._legal_facts_for(npc_id))

        prior = [
            {"worker": inter.worker_message, "npc": inter.npc_response}
            for inter in state.npc_interactions
            if inter.npc_id == npc_id
        ][-3:]

        ctx = NpcContext(
            worker_message=worker_message,
            pressure_level=pressure,
            procedural_stage=state.procedural_stage.value,
            prior_exchanges=prior,
            extra_facts_visible=extras,
        )
        return npc.respond(ctx)

    def _legal_facts_for(self, npc_id: str) -> list[str]:
        role_map = {
            "zhang_guohua": "general_contractor",
            "wang_pei": "subcontractor",
            "chen_wei": "inspector",
            "arbitrator": "arbitrator",
            "judge": "judge",
        }
        role = role_map.get(npc_id)
        if not role:
            return []
        facts = []
        for source in self.legal_knowledge_pack.get("sources", []):
            if role in source.get("applies_to", []):
                facts.append(f"{source.get('title')}: {source.get('rule')}")
        return facts[:5]

    def _compute_pressure(self, npc_id: str, state: "CaseState") -> int:
        """
        Map case state → integer pressure level for an NPC.

        Suggested scale (0-3):
            wang_pei / zhang_guohua:
              0: no formal procedure
              1: labor inspection underway
              2: arbitration filed (or +asset freeze)
              3: civil judgment rendered against them

            li_dahai:
              0: just trying to contact
              1: labor inspection summoned
              2: criminal report filed
              3: arrest warrant issued

            chen_wei: no pressure concept (always procedural)
        """
        flags = state.flags
        stage_value = state.procedural_stage.value

        if npc_id in ("wang_pei", "zhang_guohua"):
            if "civil_judgment" in stage_value or "arbitration_awarded" in stage_value:
                return 3
            if flags.get("asset_freeze_active") or "arbitration" in stage_value:
                return 2
            if flags.get("limit_order_issued") or "labor_inspection" in stage_value:
                return 1
            return 0

        if npc_id == "li_dahai":
            if flags.get("arrest_warrant_issued"):
                return 3
            if flags.get("criminal_case_filed"):
                return 2
            if flags.get("limit_order_issued"):
                return 1
            return 0

        return 0
