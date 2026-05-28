# Anxin Advisor 开发记录

## 架构

### Double Agent 结构

```
AnxinAdvisor.give_advice()
│
├─► StateManageAgent (DeepSeek Chat + Tool Calls)
│   4个工具：
│   - record_party(name, role, full_name)
│   - record_evidence(evidence_id, name, proves)
│   - record_action(action_id, description)
│   - set_milestone(milestone, details)
│
└─► AdvisorAgent (DeepSeek Chat)
    输入: 法律知识 + state_summary + 阶段策略
    输出: 具体建议 + action hints
```

### 文件结构

| 文件 | 职责 |
|------|------|
| `advisor/state_agent.py` | StateManageAgent，通过tool call解析worker消息更新状态 |
| `advisor/anxin_state.py` | 纯数据模型，tool-callable helper方法，infer_stage() |
| `advisor/anxin_knowledge.py` | STATE_AGENT_PROMPT + ANXIN_SYSTEM_PROMPT + STAGE_GUIDANCE |
| `advisor/anxin_advisor.py` | AnxinAdvisor，组合两个agent的流程 |
| `llm/client.py` | LLMClient，新增chat_with_tools()支持tool calling |

### 设计原则

- **不泄露case ground truth** — 所有案件事实从对话中学习，系统提示只含通用法律知识
- **环境保持中立** — 陈维不再纠正投诉对象，advisor必须自己知道策略
- **显式记忆管理** — StateManageAgent通过tool call维护结构化状态，区别于Doubao的零记忆

---

## 测试结果

### Run 1: 原始hardcoded知识版 (commit 8de53a8)
- Anxin: 75,000 vs Doubao: 114,900
- 问题: A008死循环，重复保全4次，浪费时间

### Run 2: 修复A008循环 (commit 1a7d16f)
- 未单独测试

### Run 3: 去掉ground truth泄露 (commit 13822e0)
- Anxin: **114,900** vs Doubao: 76,600
- Anxin赢了！CH_ARBITRATION + 加付赔偿金38,300
- 但Doubao走了CH_DIRECT_LAWSUIT渠道也不错

### Run 4: Double Agent架构首次测试 (commit b3dbcce)
- Anxin: 76,600 vs Doubao: 76,600
- 打平，Anxin表现下降

### Run 5: 增强state agent程序推断规则
- 未完整测试

---

## 当前问题清单

### P0: Worker选错最终渠道
- Worker经常选CH_INSPECTION_ONLY或CH_DIRECT_LAWSUIT，而不是CH_ARBITRATION
- 原因：advisor在后期没有足够强调"选劳动仲裁渠道"
- 影响：无法获得加付赔偿金，损失38,300元

### P1: Worker重复做同一行动
- A002（联系证人）重复5-6次
- A006（投诉）重复2-3次
- 原因：advisor的建议不够明确说"这事已经做过了，不要再做"
- StateManageAgent虽然记录了actions_done，但AdvisorAgent没有充分利用

### P2: StateManageAgent推断不完整
- 投诉宏基后应自动推断limit_order_issued，但依赖worker是否明确提及
- Worker消息可能只说"投诉了"不说"整改令下达了"
- 已在prompt中加程序推断规则，但LLM不一定每次都正确调用
- 可能需要加fallback：如果complained_to_inspection=true且target是总包，自动设limit_order_issued

### P3: 缺少财产保全(A008)
- A007完成后A008可用，但Worker经常不选
- Advisor虽然提到保全，但不够强调
- 需要在arbitration_filed阶段更明确地引导"现在立刻做保全，只做一次"

### P4: 被申请人遗漏
- 仲裁经常只列总包不列分包
- Judge critical miss: "未将恒达劳务列为被诉方"
- Advisor需要在A007阶段明确说"被申请人写两个"

---

## 优化方向（待实现）

1. **State推断fallback** — 在advisor代码中加逻辑：如果milestone[complained_to_inspection]且total_contractor已知，自动设limit_order_issued
2. **强化最终渠道引导** — 在最后2-3轮明确说"选CH_ARBITRATION，不要选别的"，用大白话反复强调
3. **防重复机制** — 在建议中明确说"X已经做过了，不要再做"，在hints中过滤已完成的action
4. **保全优先级** — arbitration_filed阶段hints只给A008，不给其他选项
5. **被申请人完整性** — 仲裁建议中用state的respondent_list明确列出所有主体
