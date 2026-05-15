import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const ADVISORS = {
  anxin: { name: '安薪 Agent', short: '安薪', tone: 'anxin' },
  doubao: { name: '豆包', short: '豆包', tone: 'doubao' },
};

const STAGE_LABELS = {
  initial_intake: '初始求助',
  evidence_gathering: '证据整理',
  negotiation: '协商沟通',
  labor_inspection: '劳动监察',
  labor_inspection_order_issued: '限期整改',
  labor_inspection_order_expired: '整改期满',
  arbitration_filed: '仲裁立案',
  arbitration_awarded: '仲裁裁决',
  civil_litigation_direct: '法院起诉',
  civil_judgment: '法院判决',
  execution: '执行',
  settlement: '和解',
  abandoned: '放弃',
};

function emptySession(advisorType) {
  return {
    advisorType,
    sessionId: null,
    caseData: null,
    maxTurns: 8,
    turns: [],
    evidencePool: [],
    judgment: null,
    finalSubmission: null,
    channel: null,
    loading: false,
    error: '',
  };
}

function App() {
  const [activeAdvisor, setActiveAdvisor] = useState('anxin');
  const [maxTurns, setMaxTurns] = useState(8);
  const [sessions, setSessions] = useState({
    anxin: emptySession('anxin'),
    doubao: emptySession('doubao'),
  });

  const active = sessions[activeAdvisor];
  const caseData = active.caseData || sessions.anxin.caseData || sessions.doubao.caseData;
  const evidenceCards = useEvidenceCards(caseData, active.evidencePool);

  async function startOrSwitch(advisorType) {
    setActiveAdvisor(advisorType);
    if (sessions[advisorType].sessionId) return;
    await startSession(advisorType);
  }

  async function startSession(advisorType) {
    updateSession(advisorType, { loading: true, error: '', maxTurns });
    try {
      const res = await fetch(`${API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ advisor_type: advisorType, max_turns: maxTurns }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      updateSession(advisorType, {
        sessionId: data.session_id,
        caseData: data.case_data,
        maxTurns: data.max_turns,
        turns: [],
        evidencePool: [],
        judgment: null,
        finalSubmission: null,
        channel: null,
        loading: false,
      });
    } catch (err) {
      updateSession(advisorType, { loading: false, error: String(err.message || err) });
    }
  }

  async function nextTurn() {
    if (!active.sessionId) {
      await startSession(activeAdvisor);
      return;
    }
    updateSession(activeAdvisor, { loading: true, error: '' });
    try {
      const res = await fetch(`${API_BASE}/sessions/${active.sessionId}/turn`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      const turn = await res.json();
      const nextTurns = [...active.turns, turn];
      updateSession(activeAdvisor, {
        turns: nextTurns,
        evidencePool: turn.current_evidence_pool || [],
        loading: false,
      });
      if (turn.is_final_turn) {
        await finalize(activeAdvisor, active.sessionId, nextTurns, turn.current_evidence_pool || []);
      }
    } catch (err) {
      updateSession(activeAdvisor, { loading: false, error: String(err.message || err) });
    }
  }

  async function finalize(advisorType, sessionId, turns, evidencePool) {
    updateSession(advisorType, { loading: true, error: '' });
    try {
      const res = await fetch(`${API_BASE}/sessions/${sessionId}/finalize`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      updateSession(advisorType, {
        turns,
        evidencePool,
        judgment: data.judgment,
        finalSubmission: data.final_submission,
        channel: data,
        loading: false,
      });
    } catch (err) {
      updateSession(advisorType, { loading: false, error: String(err.message || err) });
    }
  }

  function updateSession(advisorType, patch) {
    setSessions((prev) => ({
      ...prev,
      [advisorType]: { ...prev[advisorType], ...patch },
    }));
  }

  const canRunTurn = active.sessionId && !active.judgment && !active.loading;
  const nextLabel = !active.sessionId ? `启动${ADVISORS[activeAdvisor].short}` : active.turns.length >= active.maxTurns ? '等待裁决' : '推进下一轮';

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <div className="brand">安薪模拟环境</div>
          <div className="subtitle">虚拟案件 · Worker 执行 · Advisor 对接 · NPC 裁决</div>
        </div>
        <div className="turn-control">
          <label>总轮数</label>
          <input value={maxTurns} min="3" max="20" type="number" onChange={(e) => setMaxTurns(Number(e.target.value) || 8)} />
        </div>
      </header>

      <section className="case-panel">
        <CaseOverview caseData={caseData} maxTurns={active.maxTurns || maxTurns} />
        <EvidenceGrid cards={evidenceCards} />
      </section>

      <section className="stage">
        <WorkerFigure />
        <div className="advisor-switch">
          {Object.entries(ADVISORS).map(([key, item]) => (
            <button
              key={key}
              className={`advisor-card ${item.tone} ${activeAdvisor === key ? 'active' : ''}`}
              onClick={() => startOrSwitch(key)}
            >
              <span className="avatar">{key === 'anxin' ? '安' : '豆'}</span>
              <span>
                <strong>{item.name}</strong>
                <small>{sessionSummary(sessions[key])}</small>
              </span>
            </button>
          ))}
        </div>
      </section>

      <section className={`workspace ${ADVISORS[activeAdvisor].tone}`}>
        <div className="workspace-head">
          <div>
            <h2>{ADVISORS[activeAdvisor].name}</h2>
            <p>{active.sessionId ? `Session ${active.sessionId.slice(0, 8)} · ${active.turns.length}/${active.maxTurns}轮` : '点击启动后开始单会话模拟'}</p>
          </div>
          <button className="primary" disabled={active.loading || active.judgment || (active.sessionId && !canRunTurn)} onClick={nextTurn}>
            {active.loading ? '运行中...' : nextLabel}
          </button>
        </div>
        {active.error && <div className="error">{active.error}</div>}
        <TurnTimeline turns={active.turns} advisorType={activeAdvisor} />
        {active.judgment && <JudgmentPanel result={active.channel} />}
      </section>
    </main>
  );
}

function useEvidenceCards(caseData, evidencePool) {
  return useMemo(() => {
    if (!caseData) return [];
    const obtained = new Map((evidencePool || []).map((e) => [e.id, e]));
    const all = [
      ...(caseData.evidence_database?.initial_visible || []),
      ...(caseData.evidence_database?.discoverable || []),
    ];
    return all.map((ev) => ({
      ...ev,
      obtained: obtained.has(ev.id),
      obtained_at_day: obtained.get(ev.id)?.obtained_at_day,
    }));
  }, [caseData, evidencePool]);
}

function CaseOverview({ caseData, maxTurns }) {
  if (!caseData) {
    return (
      <div className="overview blank">
        <h1>天骄名苑住宅小区建设项目欠薪案</h1>
        <p>启动任一 advisor 后加载完整案件真相、证据库和流程设置。</p>
      </div>
    );
  }
  const gt = caseData.ground_truth;
  return (
    <div className="overview">
      <div className="case-title">
        <span className="case-id">{caseData.case_id}</span>
        <h1>{caseData.case_name}</h1>
        <p>{gt.worker_team_size}名工人在{gt.project.name}务工，赵建国被欠薪 ¥{caseData.financial.total_owed.toLocaleString()}。</p>
      </div>
      <div className="truth-grid">
        <TruthItem label="真相" value={`恒达违法分包给无资质包工头李大海，李大海收款后逃匿。`} />
        <TruthItem label="总包" value={`${gt.general_contractor.name}：第30条先行清偿责任`} />
        <TruthItem label="分包" value={`${gt.subcontractor.name}：第36条违法分包责任风险`} />
        <TruthItem label="流程" value={`当前演示 ${maxTurns} 轮，最后一轮提交最终渠道和文书`} />
      </div>
    </div>
  );
}

function TruthItem({ label, value }) {
  return (
    <div className="truth-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EvidenceGrid({ cards }) {
  return (
    <div className="evidence-grid">
      {cards.map((ev) => (
        <article key={ev.id} className={`evidence-card ${ev.obtained ? 'obtained' : ''}`}>
          <div className="evidence-top">
            <span>{ev.id}</span>
            <b>{ev.obtained ? '已取得' : '未取得'}</b>
          </div>
          <h3>{ev.name}</h3>
          <p>{ev.proves}</p>
          <small>{ev.obtained_at_day !== undefined ? `第${ev.obtained_at_day}天取得` : `强度 ${ev.evidentiary_strength}/5`}</small>
        </article>
      ))}
    </div>
  );
}

function WorkerFigure() {
  return (
    <div className="worker-figure">
      <div className="worker-avatar">赵</div>
      <div>
        <strong>赵建国</strong>
        <span>52岁 · 砌体工 · 执行 advisor 建议</span>
      </div>
    </div>
  );
}

function TurnTimeline({ turns, advisorType }) {
  if (!turns.length) {
    return <div className="empty-chat">这里会显示 worker 与 {ADVISORS[advisorType].short} 的微信式对话，以及 worker 每轮执行后环境返回的结果。</div>;
  }
  return (
    <div className="timeline">
      {turns.map((turn) => (
        <article key={turn.turn_index} className="turn">
          <div className="turn-meta">第 {turn.turn_index + 1} 轮 · 第 {turn.day} 天 · {STAGE_LABELS[turn.procedural_stage] || turn.procedural_stage}</div>
          <Bubble who="赵建国" side="left" text={turn.worker_message} />
          <Bubble who={ADVISORS[advisorType].short} side="right" text={turn.advisor_response} />
          <div className="env-event">
            <div className="action-line">
              <span>{turn.chosen_action_id}</span>
              <strong>{turn.chosen_action_name}</strong>
            </div>
            <p>{turn.action_narration}</p>
            {!!turn.new_evidence_ids?.length && <div className="new-evidence">新增证据：{turn.new_evidence_ids.join('、')}</div>}
            {turn.npc_interactions?.map((npc, idx) => (
              <div className="npc-dialog" key={`${npc.npc_id}-${idx}`}>
                <b>{npc.npc_name}</b>
                <span>{npc.text}</span>
              </div>
            ))}
          </div>
        </article>
      ))}
    </div>
  );
}

function Bubble({ who, side, text }) {
  return (
    <div className={`bubble ${side}`}>
      <small>{who}</small>
      <p>{text}</p>
    </div>
  );
}

function JudgmentPanel({ result }) {
  if (!result) return null;
  const j = result.judgment || {};
  const award = j.monetary_award || {};
  const fs = result.final_submission || {};
  return (
    <section className={`judgment ${result.channel_background_key}`}>
      <div className="judgment-banner">
        <span>{result.channel_name}</span>
        <strong>最终结果 ¥{Number(award.total || 0).toLocaleString()}</strong>
      </div>
      <div className="judgment-grid">
        <div className="paper">
          <h3>裁判/处理结果</h3>
          <p>{j.summary_in_plain_chinese || '暂无通俗说明'}</p>
          <div className="amounts">
            <span>本金 ¥{Number(award.principal || 0).toLocaleString()}</span>
            <span>加付 ¥{Number(award.additional_compensation || 0).toLocaleString()}</span>
            <span>利息 ¥{Number(award.interest || 0).toLocaleString()}</span>
          </div>
        </div>
        <div className="paper">
          <h3>最终提交</h3>
          <p>被申请人/被告：{(fs.respondents || []).join('、') || '未列明'}</p>
          <p>提交证据：{(fs.evidence_ids_submitted || []).join('、') || '无'}</p>
          <p>{fs.advisor_reasoning || ''}</p>
        </div>
        <div className="paper wide">
          <h3>文书摘要</h3>
          <pre>{j.formal_judgment_text || '（无正式文书）'}</pre>
        </div>
      </div>
    </section>
  );
}

function sessionSummary(session) {
  if (session.judgment) return '已完成裁决';
  if (session.sessionId) return `${session.turns.length}/${session.maxTurns}轮`;
  return '未启动';
}

createRoot(document.getElementById('root')).render(<App />);
