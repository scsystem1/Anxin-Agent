# 安薪沙盒（Anxin Sandbox）

一个用于演示和评估「安薪」AI 案件代理人的虚拟讨薪环境。

通过让 Anxin 和豆包（Doubao）在**同一个案件、同一个模拟农民工、同一套环境规则**下指挥维权过程，沙盒能产生两份并排可比的"判决报告"，作为 demo 的核心展示。

## 设计哲学

三层严格解耦：

```
Advisor (被测对象)  ↔  Simulated Worker (传话筒+执行者)  ↔  Environment (状态+NPC+判决)
```

- **Advisor** 永远拿不到环境状态，只能从工人的对话中获取信息（这是公平对比的前提）
- **Simulated Worker** 是 LLM 驱动的赵建国，他的执行力随 advice 具体度变化
- **Environment** 维护完整 case state、调度 NPC、最终调用判决引擎

详见 `IMPLEMENTATION_GUIDE.md`。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM API
cp .env.example .env
# 编辑 .env，填入 ANXIN_LLM_API_KEY 等

# 3. 跑双 advisor 对比（命令行旧流程）
python run.py

# 4. 启动最终交互式 Demo API + 前端静态构建
python -m uvicorn api.server:app --reload --port 8000
# 打开 http://localhost:8000

# 5. 前端开发模式（可选）
cd frontend
npm install
npm run dev
# 打开 http://localhost:5173
```

## 项目结构

```
anxin_sandbox/
├── cases/                              # 案件 JSON
│   └── tianjiao_mingyuan.json
│
├── llm/client.py                       # 统一 LLM 客户端
│
├── environment/                        # 环境
│   ├── state.py                        #   CaseState 数据结构
│   ├── actions.py                      #   Action / ActionResult
│   ├── action_handlers.py              #   每个 action 的执行逻辑
│   ├── npc_manager.py                  #   NPC 调度
│   └── env.py                          #   环境主类（gym 风格 API）
│
├── npcs/                               # NPC 角色（每个一个 LLM prompt）
│   ├── base.py
│   ├── li_dahai.py                     #   包工头李大海
│   ├── wang_pei.py                     #   恒达 HR 王培
│   ├── zhang_guohua.py                 #   宏基项目经理张国华
│   └── chen_wei.py                     #   劳动监察员陈维
│
├── worker/simulated_worker.py          # 模拟农民工赵建国
│
├── advisor/                            # 军师接口（核心）
│   ├── base.py                         #   抽象基类 Advisor
│   ├── anxin_advisor.py                #   Anxin 实现
│   └── doubao_advisor.py               #   豆包实现
│
├── judge/judgment_engine.py            # 判决引擎
│
├── runner/                             # 流程编排
│   ├── episode.py                      #   单局 runner
│   └── comparison.py                   #   双局对比
│
├── case_loader.py
├── config.py
├── run.py                              # 入口
├── requirements.txt
├── .env.example
├── README.md                           # 你正在看的这个
└── IMPLEMENTATION_GUIDE.md             # 给 codex 的实现指南
```

## 当前状态

- ✅ 核心抽象（接口、数据结构、prompts）完整
- ✅ 流程编排骨架完整
- ⚠️ 多个方法标 `NotImplementedError`，等待 Codex 按 `IMPLEMENTATION_GUIDE.md` 补全

补全后接上 LLM API 即可端到端运行。
# Anxin-Agent
