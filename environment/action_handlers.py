"""
Action handlers.

For each action in the action space, define how it mutates CaseState and
what narration it produces. Handlers are pure-ish: they take (state, action,
deps) and return ActionResult.

Design pattern: a dict of handler functions keyed by action_id. The
Environment looks up the handler and calls it.

For MVP, handlers can be simple. Codex: implement at least:
  - A001 (整理证据), A002 (联系老乡), A006 (投诉), A007 (仲裁),
    A009 (法援), A010 (催告函), A011 (谈判), A099 (放弃)
  - The remaining actions can share a generic "apply effects from JSON" handler.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from environment.actions import Action, ActionResult
from environment.state import (
    CaseState,
    Evidence,
    ProceduralStage,
    TerminalReason,
    ActionRecord,
)

if TYPE_CHECKING:
    from environment.npc_manager import NpcManager


@dataclass
class HandlerDeps:
    """Dependencies passed to every handler."""
    case_data: dict                    # the parsed case JSON
    npc_manager: "NpcManager"


HandlerFn = Callable[[CaseState, Action, HandlerDeps], ActionResult]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _add_evidence_from_data(state: CaseState, ev_id: str, case_data: dict) -> Evidence | None:
    """Look up an evidence definition in case_data and add to the pool."""
    all_evidence = (
        case_data["evidence_database"]["initial_visible"]
        + case_data["evidence_database"]["discoverable"]
    )
    for ev in all_evidence:
        if ev["id"] == ev_id:
            evidence = Evidence(
                id=ev["id"],
                name=ev["name"],
                details=ev.get("details", ev.get("obtain_via", "")),
                evidentiary_strength=ev["evidentiary_strength"],
                proves=ev["proves"],
            )
            state.add_evidence(evidence)
            return evidence
    return None


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------

def handle_A001_organize_evidence(state, action, deps) -> ActionResult:
    """整理手头已有证据：把E001-E005添加进证据池。"""
    added = []
    for ev_id in ["E001", "E002", "E003", "E004", "E005"]:
        if not state.has_evidence(ev_id):
            ev = _add_evidence_from_data(state, ev_id, deps.case_data)
            if ev:
                added.append(ev_id)
    state.advance_days(1)
    return ActionResult(
        action=action,
        success=True,
        narration=(
            f"赵建国把手机里所有跟李大海相关的微信记录都翻了一遍，"
            f"把工资结算单（E001）拍了照备份，把转账记录和催讨记录"
            f"都整理好了。手头的证据：{', '.join(added) if added else '已是最全'}。"
        ),
        new_evidence_ids=added,
        days_elapsed=1,
    )


def handle_A006_complain_to_inspection(state, action, deps) -> ActionResult:
    """
    向劳动监察大队投诉。
    
    Codex: this is the FIRST critical decision point.
    - If parameters["target_company"] contains "宏基" → correct path.
      Set procedural_stage = LABOR_INSPECTION. Trigger chen_wei greeting.
      Auto-add E006 after 7 days of advancement (or queue a delayed event).
    - If target is only "李大海" or only "恒达" → still receivable but
      monitoring scope is narrower. Don't auto-add E006.
    """
    target = str(action.parameters.get("target_company", "")).strip()
    targets_hongji = "宏基" in target
    targets_hengda = "恒达" in target

    state.advance_days(5)
    state.procedural_stage = ProceduralStage.LABOR_INSPECTION

    worker_msg = (
        f"我是赵建国，来投诉拖欠工资。被投诉方：{target or '李大海（包工头）'}。"
    )
    action.parameters.setdefault("message", worker_msg)
    npc_resp = deps.npc_manager.interact("chen_wei", worker_msg, state)

    new_evidence = []
    state_changes = {}

    if targets_hongji:
        ev = _add_evidence_from_data(state, "E006", deps.case_data)
        if ev:
            new_evidence.append("E006")
        state.advance_days(7)
        state.flags["limit_order_issued"] = True
        state.flags["限期整改令已下达"] = True
        state.flags["limit_order_due_day"] = state.current_day + 15
        state_changes.update({
            "limit_order_issued": True,
            "限期整改令已下达": True,
            "limit_order_due_day": state.flags["limit_order_due_day"],
        })
        state.procedural_stage = ProceduralStage.LABOR_INSPECTION_ORDER_ISSUED
        state.worker_known_facts.add("宏基有先行清偿责任")
        if "宏基建设集团股份有限公司" not in state.liable_parties_identified:
            state.liable_parties_identified.append("宏基建设集团股份有限公司")
        narration = (
            "陈科长接受了赵建国对宏基建设的投诉，立即启动调查。"
            "数日后，监察大队向宏基调取了《农民工实名制管理台账》，"
            "证实赵建国在该项目工作的事实。监察大队向宏基下达了"
            "《劳动保障监察限期整改指令书》，限15日内支付。"
        )
    elif targets_hengda:
        narration = (
            "陈科长受理了对恒达劳务的投诉，启动调查。"
        )
    else:
        narration = (
            "陈科长说：投诉对象是个人，不属于劳动保障监察的常规受理范围。"
        )

    return ActionResult(
        action=action,
        success=True,
        narration=narration + "\n陈科长回复：" + npc_resp.text,
        new_evidence_ids=new_evidence,
        npc_interactions=[("chen_wei", npc_resp.text)],
        state_changes=state_changes,
        days_elapsed=12 if targets_hongji else 5,
    )


def handle_A007_file_arbitration(state, action, deps) -> ActionResult:
    """向劳动仲裁委提交申请。Codex: implement."""
    respondents = action.parameters.get("respondents", [])
    if isinstance(respondents, str):
        respondents = [respondents]
    if not isinstance(respondents, list):
        respondents = []

    state.advance_days(30)
    state.procedural_stage = ProceduralStage.ARBITRATION_FILED

    targets_hongji = any("宏基" in str(r) for r in respondents)
    targets_hengda = any("恒达" in str(r) for r in respondents)

    new_evidence = []
    if targets_hongji or targets_hengda:
        ev = _add_evidence_from_data(state, "E009", deps.case_data)
        if ev:
            new_evidence.append("E009")

    for r in respondents:
        r_text = str(r)
        if "宏基" in r_text:
            state.respondent_defenses.setdefault("宏基建设集团", []).extend([
                "已付清恒达工程款",
                "无直接合同关系",
            ])
        if "恒达" in r_text:
            state.respondent_defenses.setdefault("恒达劳务", []).extend([
                "已向李大海付清劳务款",
                "无直接劳动合同",
            ])

    narration = (
        "赵建国向双流区劳动人事争议仲裁委员会提交了仲裁申请。"
        f"被申请人：{', '.join(str(r) for r in respondents) if respondents else '（未明确）'}。"
        "仲裁委已立案。"
    )
    return ActionResult(
        action=action,
        success=True,
        narration=narration,
        new_evidence_ids=new_evidence,
        days_elapsed=30,
    )


def handle_A009_legal_aid(state, action, deps) -> ActionResult:
    """申请免费法律援助。"""
    state.flags["has_legal_aid_lawyer"] = True
    state.worker_known_facts.add("有法援律师协助")
    state.advance_days(5)
    return ActionResult(
        action=action,
        success=True,
        narration=(
            "赵建国拨打了12348热线，说明情况后，区法律援助中心受理了"
            "他的申请。几天后，一位姓周的律师联系了他，免费代理本案。"
        ),
        days_elapsed=5,
    )


def handle_A011_negotiate_with_wang_pei(state, action, deps) -> ActionResult:
    """联系恒达劳务王主任谈判 → triggers wang_pei NPC."""
    # Codex: build a worker_message from action.parameters or default,
    # call deps.npc_manager.interact("wang_pei", message, state)
    msg = action.parameters.get(
        "message",
        "王主任，我是赵建国，李大海欠我7万多块工资跑了，你们恒达必须负责。",
    )
    action.parameters.setdefault("message", msg)
    npc_resp = deps.npc_manager.interact("wang_pei", msg, state)
    state.advance_days(1)

    new_evidence = []
    for ev_id in npc_resp.new_evidence_ids:
        if not state.has_evidence(ev_id):
            ev = _add_evidence_from_data(state, ev_id, deps.case_data)
            if ev:
                new_evidence.append(ev_id)

    return ActionResult(
        action=action,
        success=True,
        narration=f"赵建国找到王主任，王主任回应：{npc_resp.text}",
        new_evidence_ids=new_evidence,
        npc_interactions=[("wang_pei", npc_resp.text)],
        days_elapsed=1,
    )


def handle_A014_ask_wang_payment(state, action, deps) -> ActionResult:
    """Ask Wang Pei specifically about Hengda's payment to Li Dahai."""
    msg = action.parameters.get(
        "message",
        "王主任，你们到底有没有把我们这批工人的钱打给李大海？能不能给我看凭证？",
    )
    action.parameters.setdefault("message", msg)
    npc_resp = deps.npc_manager.interact("wang_pei", msg, state)
    state.advance_days(1)

    new_evidence = []
    if state.npc_pressure_level.get("wang_pei", 0) >= 1 or "付款" in npc_resp.text:
        ev = _add_evidence_from_data(state, "E008", deps.case_data)
        if ev:
            new_evidence.append("E008")
    for ev_id in npc_resp.new_evidence_ids:
        if not state.has_evidence(ev_id):
            ev = _add_evidence_from_data(state, ev_id, deps.case_data)
            if ev and ev_id not in new_evidence:
                new_evidence.append(ev_id)

    return ActionResult(
        action=action,
        success=True,
        narration=f"赵建国追问恒达付款情况。王主任回应：{npc_resp.text}",
        new_evidence_ids=new_evidence,
        npc_interactions=[("wang_pei", npc_resp.text)],
        days_elapsed=1,
    )


def handle_A012_criminal_report(state, action, deps) -> ActionResult:
    """对李大海提起拒不支付劳动报酬罪刑事报案。"""
    result = handle_default(state, action, deps)
    state.flags["criminal_case_filed"] = True
    state.flags["已刑事报案"] = True
    result.state_changes.update({
        "criminal_case_filed": True,
        "已刑事报案": True,
    })
    result.narration = (
        result.narration
        or "赵建国向公安机关报案，反映李大海收款后逃匿、拒不支付劳动报酬。"
    )
    return result


def handle_A013_direct_lawsuit(state, action, deps) -> ActionResult:
    """持欠条直接向法院起诉。"""
    result = handle_default(state, action, deps)
    state.procedural_stage = ProceduralStage.CIVIL_LITIGATION_DIRECT
    result.state_changes["procedural_stage"] = ProceduralStage.CIVIL_LITIGATION_DIRECT.value
    return result


def handle_A015_find_li_dahai(state, action, deps) -> ActionResult:
    """通过公安渠道查找李大海下落。"""
    if not state.flags.get("criminal_case_filed"):
        return ActionResult(
            action=action,
            success=False,
            narration="赵建国想让公安帮忙查李大海下落，但还没有刑事报案，公安没有启动查询。",
            error="尚未刑事报案",
        )
    result = handle_default(state, action, deps)
    state.worker_known_facts.add("公安查到李大海在湖南宁乡某工地")
    return result


def handle_A017_call_li_dahai(state, action, deps) -> ActionResult:
    """Call Li Dahai through the known old phone number."""
    contact_method = action.parameters.get("contact_method", "old_phone")
    msg = action.parameters.get("message", "李哥，我是赵建国，你欠我的工资什么时候给？")
    action.parameters.setdefault("contact_method", contact_method)
    action.parameters.setdefault("message", msg)
    npc_resp = deps.npc_manager.interact("li_dahai", msg, state, contact_method=contact_method)
    state.advance_days(1)
    return ActionResult(
        action=action,
        success=True,
        narration=f"赵建国拨打李大海电话：{npc_resp.text}",
        npc_interactions=[("li_dahai", npc_resp.text)],
        days_elapsed=1,
    )


def handle_A018_contact_zhang_guohua(state, action, deps) -> ActionResult:
    """Contact Hongji project manager Zhang Guohua."""
    msg = action.parameters.get(
        "message",
        "张经理，我在天骄名苑干活，李大海欠我工资。公示牌上写总包是宏基，你们能不能处理？",
    )
    action.parameters.setdefault("message", msg)
    npc_resp = deps.npc_manager.interact("zhang_guohua", msg, state)
    state.advance_days(1)

    new_evidence = []
    for ev_id in npc_resp.new_evidence_ids:
        if not state.has_evidence(ev_id):
            ev = _add_evidence_from_data(state, ev_id, deps.case_data)
            if ev:
                new_evidence.append(ev_id)

    return ActionResult(
        action=action,
        success=True,
        narration=f"赵建国联系宏基项目经理张国华。张国华回应：{npc_resp.text}",
        new_evidence_ids=new_evidence,
        npc_interactions=[("zhang_guohua", npc_resp.text)],
        days_elapsed=1,
    )


def handle_A_FINAL(state, action, deps) -> ActionResult:
    """Record the final channel choice and document package."""
    from environment.state import FinalSubmission

    params = action.parameters or {}
    channel_id = str(params.get("channel_id") or "CH_GIVE_UP")
    evidence_ids = params.get("evidence_ids_submitted")
    if not isinstance(evidence_ids, list):
        evidence_ids = list(state.evidence_pool.keys())
    respondents = params.get("respondents")
    if isinstance(respondents, str):
        respondents = [respondents]
    if not isinstance(respondents, list):
        respondents = []
    docs = params.get("drafted_documents")
    if not isinstance(docs, list):
        docs = []

    state.final_submission = FinalSubmission(
        channel_id=channel_id,
        channel_name=str(params.get("channel_name") or ""),
        advisor_reasoning=str(params.get("advisor_reasoning") or ""),
        drafted_documents=docs,
        evidence_ids_submitted=[str(e) for e in evidence_ids],
        respondents=[str(r) for r in respondents],
    )

    stage_map = {
        "CH_INSPECTION_ONLY": ProceduralStage.LABOR_INSPECTION_ORDER_ISSUED,
        "CH_ARBITRATION": ProceduralStage.ARBITRATION_FILED,
        "CH_DIRECT_LAWSUIT": ProceduralStage.CIVIL_LITIGATION_DIRECT,
        "CH_CRIMINAL_CIVIL": ProceduralStage.ARBITRATION_FILED,
        "CH_GIVE_UP": ProceduralStage.ABANDONED,
    }
    state.procedural_stage = stage_map.get(channel_id, ProceduralStage.ARBITRATION_FILED)

    if channel_id == "CH_GIVE_UP":
        state.mark_terminal(TerminalReason.ABANDONED)

    state.advance_days(1 if channel_id != "CH_GIVE_UP" else 0)
    return ActionResult(
        action=action,
        success=True,
        narration=(
            f"赵建国按照军师建议，选择「{state.final_submission.channel_name or channel_id}」"
            f"作为最终渠道，并整理提交了 {len(state.final_submission.evidence_ids_submitted)} 份证据。"
        ),
        state_changes={
            "final_submission_channel": channel_id,
            "respondents": list(state.final_submission.respondents),
            "procedural_stage": state.procedural_stage.value,
        },
        days_elapsed=1 if channel_id != "CH_GIVE_UP" else 0,
    )


def handle_A099_give_up(state, action, deps) -> ActionResult:
    """放弃维权 → terminal."""
    state.mark_terminal(TerminalReason.ABANDONED)
    state.procedural_stage = ProceduralStage.ABANDONED
    return ActionResult(
        action=action,
        success=True,
        narration="赵建国叹了口气，关掉了App。",
        days_elapsed=0,
    )


def handle_default(state, action, deps) -> ActionResult:
    """
    Fallback: apply effects from the action's JSON spec mechanically.
    Used for actions Codex hasn't written a custom handler for.
    """
    spec_raw = next(
        (a for a in deps.case_data["action_space"] if a["id"] == action.action_id),
        None,
    )
    if not spec_raw:
        return ActionResult(
            action=action,
            success=False,
            narration="（无效行动）",
            error=f"Unknown action {action.action_id}",
        )
    days = spec_raw.get("duration_days", 1)
    state.advance_days(days)
    effects = spec_raw.get("effects", {})
    new_ev = []
    for ev_id in effects.get("evidence_pool_add", []):
        if not state.has_evidence(ev_id):
            ev = _add_evidence_from_data(state, ev_id, deps.case_data)
            if ev:
                new_ev.append(ev_id)
    for k, v in effects.get("state_changes", {}).items():
        state.flags[k] = v
    return ActionResult(
        action=action,
        success=True,
        narration=spec_raw.get("narration", spec_raw.get("name", "")),
        new_evidence_ids=new_ev,
        days_elapsed=days,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ACTION_HANDLERS: dict[str, HandlerFn] = {
    "A001": handle_A001_organize_evidence,
    "A006": handle_A006_complain_to_inspection,
    "A007": handle_A007_file_arbitration,
    "A009": handle_A009_legal_aid,
    "A011": handle_A011_negotiate_with_wang_pei,
    "A012": handle_A012_criminal_report,
    "A013": handle_A013_direct_lawsuit,
    "A014": handle_A014_ask_wang_payment,
    "A015": handle_A015_find_li_dahai,
    "A017": handle_A017_call_li_dahai,
    "A018": handle_A018_contact_zhang_guohua,
    "A_FINAL": handle_A_FINAL,
    "A099": handle_A099_give_up,
}


def get_handler(action_id: str) -> HandlerFn:
    return ACTION_HANDLERS.get(action_id, handle_default)
