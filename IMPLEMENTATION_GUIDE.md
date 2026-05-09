# 实现指南（Codex 专用）

这份文档是给 Codex 的施工说明书。整个项目骨架已经搭好，所有的接口、数据结构、prompt 模板都已写明，你需要做的是**填空**——把所有标记 `NotImplementedError` 的方法补全，让整个 pipeline 能在配置好 LLM API 后端到端运行。

请严格按本指南执行。**不要扩展架构、不要修改接口签名、不要重新设计模块边界**。架构是经过深度推敲的，三层解耦的隔离原则尤其重要。

---

## 0. 环境与开发约定

- **Python ≥ 3.10**（用了 `X | None` 语法、`dict[str, Y]` 内置泛型）
- 严格使用类型标注
- 所有用户可见文本（NPC 回复、worker 求助、判决书）用中文
- 调试日志用 stderr，正常 stdout 留给 transcript 输出
- 不要新增第三方依赖（已在 `requirements.txt` 列全）
- 任何随机性都要可关：温度参数从配置注入

---

## 1. 实现优先级（按这个顺序补全）

```
Tier 1 (跑通 pipeline 必需)
  1. llm/client.py            : LLMClient.chat() 实际 HTTP 调用
  2. advisor/anxin_advisor.py : give_advice() 
  3. advisor/doubao_advisor.py: give_advice()
  4. worker/simulated_worker.py: formulate_request(), choose_action()
  5. environment/actions.py    : evaluate_preconditions()

Tier 2 (使案件能推进)
  6. npcs/base.py              : BaseNPC.respond()
  7. npcs/li_dahai.py          : 旧号停机分支
  8. environment/npc_manager.py: NpcManager.interact(), _compute_pressure()
  9. environment/action_handlers.py: A006, A007, A011 三个核心 handler

Tier 3 (打磨)
  10. action_handlers.py 余下: A002, A005, A010, A012, A013, A014, A015, A016
  11. 错误处理、retry、限流
  12. 端到端 smoke test
```

---

## 2. Tier 1 详细规范

### 2.1 `llm/client.py` — `LLMClient.chat()`

**目标**：用 OpenAI 风格 SDK 实现一次 chat completion 调用。

```python
from openai import OpenAI
import sys, time

def chat(self, messages, *, temperature=0.7, max_tokens=2048,
         response_format=None, purpose="") -> str:
    client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
    
    kwargs = dict(model=self.config.model, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}
    
    last_err = None
    for attempt in range(3):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            print(f"[LLM/{purpose}] {self.config.model} | "
                  f"{int((time.time()-t0)*1000)}ms | "
                  f"in≈{sum(len(m['content']) for m in messages)}c "
                  f"out≈{len(text)}c", file=sys.stderr)
            return text
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after retries: {last_err}")
```

注意：

- 不要捕获 `KeyboardInterrupt`
- `chat_json` 在 base class 已实现，不要重写
- 某些非 OpenAI 后端不支持 `response_format`，可在 base_url 包含 "ark"/"deepseek" 等关键词时跳过该参数

### 2.2 `advisor/anxin_advisor.py` — `give_advice()`

```python
def give_advice(self, request):
    # 1. accumulate history
    self._chat_history.append({"role": "user", "content": request.worker_message})
    
    # 2. build messages
    messages = [
        {"role": "system", "content": self.system_prompt},
        *self._chat_history,
    ]
    
    # 3. call LLM (slightly higher temp for varied advice)
    raw = self.llm.chat(messages, temperature=0.6, purpose="advisor_anxin")
    
    # 4. parse trailing <<actions: ...>> hints
    cleaned, hints = self._parse_action_hints(raw)
    
    # 5. record assistant turn (use cleaned text — no need to leak the markup)
    self._chat_history.append({"role": "assistant", "content": cleaned})
    
    return AdvisoryResponse(text=cleaned, suggested_action_hints=hints)
```

`_parse_action_hints` 已在文件中实现，不要重写。

### 2.3 `advisor/doubao_advisor.py` — `give_advice()`

```python
def give_advice(self, request):
    self._chat_history.append({"role": "user", "content": request.worker_message})
    messages = [
        {"role": "system", "content": DOUBAO_SYSTEM_PROMPT},
        *self._chat_history,
    ]
    text = self.llm.chat(messages, temperature=0.7, purpose="advisor_doubao")
    self._chat_history.append({"role": "assistant", "content": text})
    return AdvisoryResponse(text=text, suggested_action_hints=[])
```

**绝对不要**给豆包加 case-tracking 指令、加结构化输出要求、或者用我们手写的法律 prompt 补强它。它的"差"是它本身的特性，而不是 prompt 工程的产物。

### 2.4 `worker/simulated_worker.py`

#### `formulate_request(observation)`

```python
def formulate_request(self, observation):
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
```

#### `choose_action(advice_text, available_actions, advisor_hints)`

```python
def choose_action(self, advice_text, available_actions, advisor_hints=None):
    # 1. build action menu string
    menu_lines = []
    for spec in available_actions:
        params_hint = (
            f" [需填: {', '.join(spec.parameters_required)}]"
            if spec.parameters_required else ""
        )
        menu_lines.append(f"  - [{spec.id}] {spec.name}{params_hint}")
    menu = "\n".join(menu_lines)
    
    # 2. prepend hints if Anxin gave any
    if advisor_hints:
        hint_block = "\n# 来自军师的结构化建议（你应该优先考虑）\n" + \
                     "\n".join(f"  → {h}" for h in advisor_hints)
        menu = hint_block + "\n\n# 完整可选行动\n" + menu
    
    prompt = ACTION_SELECTION_PROMPT.format(
        persona=self.persona,
        advice_text=advice_text,
        action_menu=menu,
    )
    
    parsed = self.llm.chat_json(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,                      # lower temp for action choice
        purpose="worker_action",
    )
    
    action_id = parsed.get("action_id", "A001")
    valid_ids = {s.id for s in available_actions}
    if action_id not in valid_ids:
        # fallback — log and default
        print(f"[Warn] worker chose invalid action {action_id}; falling back to A001",
              file=sys.stderr)
        action_id = "A001"
    
    return WorkerActionChoice(
        action=Action(action_id=action_id, parameters=parsed.get("parameters", {})),
        reasoning=parsed.get("reasoning_in_worker_voice", ""),
    )
```

**绝对不要**：

- 修改 worker 的 persona 让他变"聪明"。低执行力是设计的核心。
- 在工具层"修正"工人选错的目标公司。错的就是错的，那就是 advice 不到位的体现。

### 2.5 `environment/actions.py` — `evaluate_preconditions()`

实现一个支持以下语法的简单解释器：

```
""                                  → 永远 True
"E001 in evidence_pool"             → state.has_evidence("E001")
"evidence_pool_size>=2"             → len(state.evidence_pool) >= 2
"procedural_stage>=labor_inspection"→ ProceduralStage 序号比较
"state.限期整改令已下达 == true"     → state.flags.get("限期整改令已下达") is True
"worker_knows_wang_xinglin"         → 一个白名单常量映射，always True for MVP
"recent_npc_interaction.wang_pei"   → 最近 5 个 npc_interactions 里有 wang_pei
```

参考实现：

```python
def evaluate_preconditions(spec, state):
    for pre in spec.preconditions:
        ok, why = _eval_one(pre, state)
        if not ok:
            return False, why
    return True, ""

def _eval_one(pre: str, state) -> tuple[bool, str]:
    pre = pre.strip()
    if not pre:
        return True, ""
    
    # E001 in evidence_pool
    m = re.match(r"^(E\d+)\s+in\s+evidence_pool$", pre)
    if m:
        ev_id = m.group(1)
        return state.has_evidence(ev_id), f"缺少证据 {ev_id}"
    
    # evidence_pool_size>=N
    m = re.match(r"^evidence_pool_size\s*(>=|>|==|<|<=)\s*(\d+)$", pre)
    if m:
        op, n = m.group(1), int(m.group(2))
        return _cmp(len(state.evidence_pool), op, n), \
               f"证据数量不满足 {pre}（当前 {len(state.evidence_pool)}）"
    
    # procedural_stage>=stage_name
    m = re.match(r"^procedural_stage\s*(>=|>|==|<|<=)\s*([\w_]+)$", pre)
    if m:
        op, stage_name = m.group(1), m.group(2)
        ORDER = [s.value for s in ProceduralStage]
        cur = ORDER.index(state.procedural_stage.value)
        try:
            target = ORDER.index(stage_name)
        except ValueError:
            return False, f"未知阶段 {stage_name}"
        return _cmp(cur, op, target), f"程序阶段不满足 {pre}"
    
    # state.<key> == true|false
    m = re.match(r"^state\.([^\s]+)\s*==\s*(true|false)$", pre)
    if m:
        key, val = m.group(1), m.group(2) == "true"
        return state.flags.get(key) == val, f"标志 {key} 不为 {val}"
    
    # bare flag: worker_knows_wang_xinglin → always True for MVP
    if pre == "worker_knows_wang_xinglin":
        return True, ""
    
    # recent_npc_interaction.<id>
    m = re.match(r"^recent_npc_interaction\.([\w_]+)$", pre)
    if m:
        npc_id = m.group(1)
        recent = state.npc_interactions[-5:]
        return any(i.npc_id == npc_id for i in recent), \
               f"最近未与 {npc_id} 交互"
    
    # unknown — be permissive but warn
    print(f"[Warn] unknown precondition: {pre!r}", file=sys.stderr)
    return True, ""

def _cmp(a, op, b):
    return {">=": a >= b, ">": a > b, "==": a == b, "<": a < b, "<=": a <= b}[op]
```

---

## 3. Tier 2 详细规范

### 3.1 `npcs/base.py` — `BaseNPC.respond()`

```python
def respond(self, ctx):
    system = self.system_prompt(ctx)
    
    # flatten prior exchanges into role/content turns
    history = []
    for ex in ctx.prior_exchanges:
        history.append({"role": "user", "content": ex["worker"]})
        history.append({"role": "assistant", "content": ex["npc"]})
    
    messages = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": ctx.worker_message},
    ]
    
    raw = self.llm.chat(messages, temperature=0.7,
                        purpose=f"npc_{self.npc_id}")
    
    # parse <<reveal: EXXX>>
    new_evidence = []
    m = re.search(r"<<reveal:\s*(E\d+)\s*>>", raw)
    if m:
        new_evidence = [m.group(1)]
        raw = re.sub(r"<<reveal:\s*E\d+\s*>>", "", raw).strip()
    
    return NpcResponse(text=raw, new_evidence_ids=new_evidence)
```

### 3.2 `npcs/li_dahai.py` — 旧号停机

```python
def respond(self, ctx):
    if "via_old_phone" in ctx.extra_facts_visible:
        return NpcResponse(
            text="（电话提示音：您拨打的电话已停机，请稍后再拨）",
        )
    if "via_new_phone_unknown" in ctx.extra_facts_visible:
        # worker doesn't know the new phone yet
        return NpcResponse(
            text="（电话无法接通——你不知道他的新号码）",
        )
    return super().respond(ctx)
```

### 3.3 `environment/npc_manager.py`

#### `interact()`

```python
def interact(self, npc_id, worker_message, state, contact_method=None):
    npc = self.get(npc_id)
    
    # pressure
    pressure = self._compute_pressure(npc_id, state)
    
    # extra facts visible to this NPC
    extras = []
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
    
    # build prior_exchanges from state.npc_interactions, last 3 with this NPC
    prior = []
    for inter in state.npc_interactions:
        if inter.npc_id == npc_id:
            prior.append({"worker": inter.worker_message, "npc": inter.npc_response})
    prior = prior[-3:]
    
    ctx = NpcContext(
        worker_message=worker_message,
        pressure_level=pressure,
        procedural_stage=state.procedural_stage.value,
        prior_exchanges=prior,
        extra_facts_visible=extras,
    )
    return npc.respond(ctx)
```

#### `_compute_pressure()`

```python
def _compute_pressure(self, npc_id, state):
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
        if flags.get("criminal_case_filed"):
            return 2
        if flags.get("limit_order_issued"):
            return 1
        return 0
    
    return 0  # chen_wei: no pressure concept
```

### 3.4 `environment/action_handlers.py` — 关键 handler

#### `handle_A006_complain_to_inspection`

```python
def handle_A006_complain_to_inspection(state, action, deps):
    target = action.parameters.get("target_company", "").strip()
    
    # determine correctness
    targets_hongji = "宏基" in target
    targets_hengda = "恒达" in target
    targets_lidahai = "李大海" in target or not target
    
    state.advance_days(5)
    state.procedural_stage = ProceduralStage.LABOR_INSPECTION
    
    # always trigger chen_wei greeting
    npc_resp = deps.npc_manager.interact(
        "chen_wei",
        worker_message=f"我是赵建国，来投诉拖欠工资。被投诉方：{target or '李大海（包工头）'}。",
        state=state,
    )
    state.npc_interactions.append(NpcInteraction(
        day=state.current_day, npc_id="chen_wei",
        worker_message=action.parameters.get("message", "我要投诉欠薪"),
        npc_response=npc_resp.text,
    ))
    
    new_evidence = []
    state_changes = {}
    
    if targets_hongji:
        # Correct path: inspection has authority to pull the 实名台账
        # Schedule E006 to be added after a few more days
        ev = _add_evidence_from_data(state, "E006", deps.case_data)
        if ev:
            new_evidence.append("E006")
        state.advance_days(7)
        state.flags["limit_order_issued"] = True
        state_changes["limit_order_issued"] = True
        state.procedural_stage = ProceduralStage.LABOR_INSPECTION_ORDER_ISSUED
        state.worker_known_facts.add("宏基有先行清偿责任")
        state.liable_parties_identified = list(set(
            state.liable_parties_identified + ["宏基建设集团股份有限公司"]
        ))
        narration = (
            "陈科长接受了赵建国对宏基建设的投诉，立即启动调查。"
            "数日后，监察大队向宏基调取了《农民工实名制管理台账》，"
            "证实赵建国在该项目工作的事实。监察大队向宏基下达了"
            "《劳动保障监察限期整改指令书》，限15日内支付。"
        )
    elif targets_hengda:
        narration = (
            "陈科长受理了对恒达的投诉，但提示：建议同时把总包宏基"
            "也列入投诉对象，因为根据规定总包对农民工工资负有先行"
            "清偿责任。是否补充？（陈科长不会替工人改投诉书）"
        )
        # don't auto-add E006, just monitor
    else:
        narration = (
            "陈科长说：投诉对象是包工头李大海，但李大海是个人，"
            "不是劳动保障监察的常规执法对象。建议把恒达劳务和总包"
            "宏基建设列入投诉对象。"
        )
    
    return ActionResult(
        action=action, success=True,
        narration=narration + "\n陈科长回复：" + npc_resp.text,
        new_evidence_ids=new_evidence,
        npc_interactions=[("chen_wei", npc_resp.text)],
        state_changes=state_changes,
        days_elapsed=12 if targets_hongji else 5,
    )
```

#### `handle_A007_file_arbitration`

```python
def handle_A007_file_arbitration(state, action, deps):
    respondents = action.parameters.get("respondents", [])
    if isinstance(respondents, str):
        respondents = [respondents]
    
    state.advance_days(30)
    state.procedural_stage = ProceduralStage.ARBITRATION_FILED
    
    targets_hongji = any("宏基" in r for r in respondents)
    targets_hengda = any("恒达" in r for r in respondents)
    
    new_evidence = []
    if targets_hongji or targets_hengda:
        # court / arbitration commission may pull E009 (subcontract)
        ev = _add_evidence_from_data(state, "E009", deps.case_data)
        if ev:
            new_evidence.append("E009")
    
    # respondents file defenses
    for r in respondents:
        if "宏基" in r:
            state.respondent_defenses.setdefault("宏基建设集团", []).extend([
                "已付清恒达工程款", "无直接合同关系",
            ])
        if "恒达" in r:
            state.respondent_defenses.setdefault("恒达劳务", []).extend([
                "已向李大海付清劳务款", "无直接劳动合同",
            ])
    
    narration = (
        f"赵建国向双流区劳动人事争议仲裁委员会提交了仲裁申请。"
        f"被申请人：{', '.join(respondents) if respondents else '（未明确）'}。"
        f"仲裁委已立案。"
    )
    return ActionResult(
        action=action, success=True,
        narration=narration,
        new_evidence_ids=new_evidence,
        days_elapsed=30,
    )
```

#### `handle_A011_negotiate_with_wang_pei`

```python
def handle_A011_negotiate_with_wang_pei(state, action, deps):
    msg = action.parameters.get("message",
        "王主任，我是赵建国，李大海欠我7万多块工资跑了，你们恒达必须负责")
    
    npc_resp = deps.npc_manager.interact("wang_pei", msg, state)
    state.advance_days(1)
    state.npc_interactions.append(NpcInteraction(
        day=state.current_day, npc_id="wang_pei",
        worker_message=msg, npc_response=npc_resp.text,
    ))
    
    new_evidence = []
    for ev_id in npc_resp.new_evidence_ids:
        if not state.has_evidence(ev_id):
            ev = _add_evidence_from_data(state, ev_id, deps.case_data)
            if ev:
                new_evidence.append(ev_id)
    
    return ActionResult(
        action=action, success=True,
        narration=f"赵建国找到王主任，王主任回应：{npc_resp.text}",
        new_evidence_ids=new_evidence,
        npc_interactions=[("wang_pei", npc_resp.text)],
        days_elapsed=1,
    )
```

---

## 4. Tier 3 详细规范

### 4.1 余下的 action handler

A002, A005, A010, A012, A013, A014, A015, A016 都可以走 `handle_default`（已实现）即可。但额外建议自定义：

- **A012（刑事报案）**：除 `handle_default` 的副作用外，把 `flags["criminal_case_filed"] = True` 设上，并提升 li_dahai 的 pressure
- **A013（直接起诉）**：把 `procedural_stage = CIVIL_LITIGATION_DIRECT`，跳过仲裁
- **A015（公安查李大海下落）**：仅在 `flags["criminal_case_filed"]` 为 True 时才有效

### 4.2 错误处理

- LLM 调用包 try/except，失败重试 3 次
- 工人选了不存在的 action → fallback 到 A001
- NPC 输出格式错误 → 直接当 plain text 用，不报错

### 4.3 端到端 smoke test

新建 `tests/test_smoke.py`：

```python
def test_pipeline_runs_with_mock_llm(monkeypatch):
    # 用 mock 替换 LLMClient.chat，让它返回固定文本
    # 跑 EpisodeRunner，assert 拿到一个 Judgment 对象
    ...
```

---

## 5. 验证 checklist（Steven 接 LLM 后做这些）

按顺序验证：

1. `python -c "from llm.client import LLMClient; LLMClient.from_env().chat([{'role':'user','content':'你好'}], purpose='ping')"`
   → 能拿到 LLM 响应说明 LLM 通了

2. `python -c "from environment.env import Environment; e = Environment.from_case_file('cases/tianjiao_mingyuan.json'); print(e.reset())"`
   → 能正常加载案件、reset、得到 observation

3. `python run.py --advisor anxin --max-turns 5`
   → 跑 5 轮 Anxin，看 transcript 是否合理

4. `python run.py --advisor doubao --max-turns 5`
   → 跑 5 轮豆包，看是不是给的建议明显模糊

5. `python run.py --max-turns 30`
   → 完整跑双 advisor，输出 `out/comparison_report.md`

每一步若失败，只看那一步对应的模块。

---

## 6. 不要做的事（红线）

1. **不要在 advisor/base.py 的 AdvisoryRequest 里塞 case_state**——这破坏隔离原则。如果你觉得 advisor 必须知道更多，那应该是 worker 在求助文本里多说，而不是把 state 塞过去。

2. **不要把 worker 改"聪明"**。Worker 的低执行力是设计的核心，不是 bug。

3. **不要给豆包 advisor 加专业法律 prompt**。豆包的 baseline 必须是 vanilla。

4. **不要在 action_handlers 里偷偷修正工人选的错误参数**（"哦他选 target=李大海，但应该是宏基，我帮他改一下"——绝对不行）。错误的输入就要产生错误的结果，这才是对比的价值所在。

5. **不要新增模块**。当前模块划分覆盖了所有职责。如果你觉得需要新模块，说明你想错了路径。

6. **不要假设案件数据里的字段**——所有字段以 `cases/tianjiao_mingyuan.json` 为准，不要按记忆补字段。

---

## 7. 实现顺序提示

完整顺序（每步完成后做一次最小冒烟测试）：

1. `llm/client.py` → 用 ping 验证
2. `environment/actions.py:evaluate_preconditions` → 单元测试一两个 case
3. `worker/simulated_worker.py:formulate_request` → 跑一次空 env，看 worker 第一句话是否合理
4. `advisor/anxin_advisor.py:give_advice` + `doubao_advisor.py:give_advice` → 喂一句话，看双方各自的回答风格差异
5. `worker/simulated_worker.py:choose_action` → 喂一段建议 + 几个 action，看选的对不对
6. `npcs/base.py:respond` + `npcs/li_dahai.py` → 跑一次 li_dahai 旧号交互
7. `environment/npc_manager.py:interact + _compute_pressure`
8. `environment/action_handlers.py` 三个核心 handler
9. 跑 `python run.py --advisor anxin --max-turns 3`，看完整 loop 是否走得通
10. 跑双 advisor 对比

完成后回到 README 验证 checklist 走一遍。

---

## 8. 已知 trade-off 和未来工作

- **NPC 的"压力等级"是离散的 4 档**。更精细的话可以是连续值，但 MVP 不需要。
- **判决引擎是单次 LLM 调用**。更严肃的是 fact-finding / liability / monetary 三步分开调，互相 verify。MVP 不做。
- **时间推进是按 action 的 duration_days 累加**。没有"延迟事件"机制（比如对方N天后才回应）。MVP 不做。
- **多个 NPC 之间不互相通信**。比如 wang_pei 不知道 zhang_guohua 说了什么。MVP 不做。

这些都不影响 demo 价值。
