from __future__ import annotations

import unittest
from unittest.mock import patch

from advisor.anxin_advisor import AnxinAdvisor
from environment.actions import Action, ActionSpec, evaluate_preconditions
from environment.env import Environment
from environment.state import ProceduralStage
from npcs.base import NpcContext
from npcs.li_dahai import LiDahaiNPC
from runner.episode import EpisodeRunner
from worker.simulated_worker import SimulatedWorker


CASE_PATH = "cases/tianjiao_mingyuan.json"


class FakeLLM:
    action_index = 0
    actions = [
        {"action_id": "A001", "parameters": {}, "reasoning_in_worker_voice": "我先整理证据。"},
        {
            "action_id": "A006",
            "parameters": {"target_company": "宏基建设"},
            "reasoning_in_worker_voice": "军师说要投诉宏基。",
        },
        {
            "action_id": "A007",
            "parameters": {"respondents": ["宏基建设集团股份有限公司", "成都恒达劳务有限公司"]},
            "reasoning_in_worker_voice": "我把两个公司都写上。",
        },
    ]

    def chat(self, messages, *, temperature=0.7, max_tokens=2048, response_format=None, purpose=""):
        if purpose == "worker_request":
            return "我这个工资还没拿到，下一步咋办？"
        if purpose == "advisor_anxin":
            return "先按我说的做，材料要写清楚。"
        if purpose == "advisor_doubao":
            return "嗯，你这个情况确实挺急的。建议先把手上的材料整理好，然后去找劳动监察部门问问看。"
        if purpose.startswith("npc_chen"):
            return "你这个投诉我先登记，材料齐的话我们会向总包调取台账。"
        if purpose.startswith("npc_"):
            return "这个事情我知道了。"
        return "好的。"

    def chat_json(self, messages, *, temperature=0.3, purpose=""):
        if purpose == "worker_action":
            action = self.actions[min(self.action_index, len(self.actions) - 1)]
            FakeLLM.action_index += 1
            return action
        if purpose == "judge":
            return {
                "primary_respondent": "宏基建设集团股份有限公司",
                "fact_findings": [],
                "liability_findings": [],
                "monetary_award": {
                    "principal": 76600,
                    "additional_compensation": 0,
                    "interest": 0,
                    "legal_costs": 0,
                    "total": 76600,
                },
                "summary_in_plain_chinese": "已经走到仲裁记录。",
                "formal_judgment_text": "本院查明。本院认为。判决如下。",
                "critical_misses": [],
            }
        if purpose == "worker_final_submission":
            return {
                "channel_id": "CH_ARBITRATION",
                "channel_name": "劳动仲裁",
                "advisor_reasoning": "军师让我把两个公司都列上。",
                "respondents": ["宏基建设集团股份有限公司", "成都恒达劳务有限公司"],
                "evidence_ids_submitted": ["E001", "E002", "E003", "E006", "E009"],
                "drafted_documents": [
                    {"doc_type": "仲裁申请书", "content": "请求支付拖欠工资76600元。"}
                ],
            }
        return {}


class SandboxSmokeTests(unittest.TestCase):
    def setUp(self):
        FakeLLM.action_index = 0

    def test_precondition_dsl_and_flag_aliases(self):
        with patch("llm.client.LLMClient.from_env", return_value=FakeLLM()):
            env = Environment.from_case_file(CASE_PATH)
        env.reset()
        spec = ActionSpec("T", "测试", "procedure", ["evidence_pool_size>=2"], 1)
        ok, why = evaluate_preconditions(spec, env.state)
        self.assertFalse(ok)
        self.assertIn("证据数量", why)

        env.step(Action("A001"))
        ok, _ = evaluate_preconditions(spec, env.state)
        self.assertTrue(ok)

        env.state.procedural_stage = ProceduralStage.ARBITRATION_FILED
        spec_stage = ActionSpec("T2", "测试阶段", "procedure", ["procedural_stage>=arbitration"], 1)
        ok, _ = evaluate_preconditions(spec_stage, env.state)
        self.assertTrue(ok)

        env.state.flags["limit_order_issued"] = True
        spec_flag = ActionSpec("T3", "测试标志", "procedure", ["state.限期整改令已下达 == true"], 1)
        ok, _ = evaluate_preconditions(spec_flag, env.state)
        self.assertTrue(ok)

    def test_anxin_action_hint_parser(self):
        text, hints = AnxinAdvisor._parse_action_hints(
            "先投诉总包。\n<<actions: A006(target=宏基建设); A009>>"
        )
        self.assertEqual(text, "先投诉总包。")
        self.assertEqual(hints, ["A006(target=宏基建设)", "A009"])

    def test_worker_invalid_action_falls_back(self):
        class InvalidActionLLM(FakeLLM):
            def chat_json(self, messages, *, temperature=0.3, purpose=""):
                return {
                    "action_id": "BAD",
                    "parameters": {"target_company": "宏基"},
                    "reasoning_in_worker_voice": "我选错了。",
                }

        worker = SimulatedWorker(llm=InvalidActionLLM())
        spec = ActionSpec("A001", "整理证据", "evidence", [], 1)
        choice = worker.choose_action("随便做点什么", [spec])
        self.assertEqual(choice.action.action_id, "A001")
        self.assertEqual(choice.action.parameters["target_company"], "宏基")

    def test_worker_second_turn_uses_incremental_prompt(self):
        class CaptureLLM(FakeLLM):
            last_messages = None

            def chat(self, messages, *, temperature=0.7, max_tokens=2048, response_format=None, purpose=""):
                CaptureLLM.last_messages = messages
                return "我刚刚整理了证据，下一步咋办？"

        from environment.env import Observation
        from datetime import date

        worker = SimulatedWorker(llm=CaptureLLM())
        obs = Observation(
            day=1,
            date=date(2024, 4, 21),
            procedural_stage="initial_intake",
            recent_events=[
                "赵建国今天打开了'安薪'App，准备求助。",
                "赵建国把手机里所有跟李大海相关的微信记录都翻了一遍。",
            ],
            actions_taken_summary=["第1天 A001 整理手头已有证据"],
        )
        worker.formulate_request(obs)
        prompt = CaptureLLM.last_messages[0]["content"]
        self.assertIn("只说增量信息", prompt)
        self.assertIn("刚刚发生的新进展", prompt)
        self.assertIn("翻了一遍", prompt)
        self.assertNotIn("准备求助。", prompt)

    def test_li_dahai_old_phone_is_unreachable(self):
        npc = LiDahaiNPC({"id": "li_dahai"}, llm=FakeLLM())
        resp = npc.respond(NpcContext(
            worker_message="李哥你接电话",
            pressure_level=0,
            procedural_stage="initial_intake",
            extra_facts_visible=["via_old_phone"],
        ))
        self.assertIn("已停机", resp.text)

    def test_pipeline_runs_with_mock_llm(self):
        with patch("llm.client.LLMClient.from_env", return_value=FakeLLM()):
            env = Environment.from_case_file(CASE_PATH)
            worker = SimulatedWorker()
            advisor = AnxinAdvisor()
            result = EpisodeRunner(env, worker, advisor, max_turns=4, verbose=False).run()

        self.assertIsNotNone(result.final_judgment)
        self.assertIn("E006", env.state.evidence_pool)
        self.assertIn("E009", env.state.evidence_pool)
        self.assertIsNotNone(env.state.final_submission)
        self.assertEqual(env.state.final_submission.channel_id, "CH_ARBITRATION")
        self.assertTrue(env.state.flags["limit_order_issued"])
        self.assertEqual(env.state.procedural_stage, ProceduralStage.ARBITRATION_FILED)
        self.assertEqual(len(env.state.npc_interactions), 1)

    def test_final_submission_give_up_channel(self):
        with patch("llm.client.LLMClient.from_env", return_value=FakeLLM()):
            env = Environment.from_case_file(CASE_PATH)
        env.reset()
        from environment.actions import make_final_action

        _, action_result = env.step(make_final_action(
            channel_id="CH_GIVE_UP",
            channel_name="放弃维权",
            advisor_reasoning="太难了",
            drafted_documents=[],
            evidence_ids=[],
            respondents=[],
        ))

        self.assertTrue(action_result.success)
        self.assertTrue(env.state.is_terminal)
        self.assertEqual(env.state.procedural_stage, ProceduralStage.ABANDONED)

    def test_judge_bad_json_falls_back_to_rule_award(self):
        class BadJudgeLLM(FakeLLM):
            def chat_json(self, messages, *, temperature=0.3, max_tokens=2048, purpose=""):
                if purpose == "judge":
                    raise ValueError("Unterminated string starting at: line 20 column 20")
                return super().chat_json(messages, temperature=temperature, purpose=purpose)

        with patch("llm.client.LLMClient.from_env", return_value=FakeLLM()):
            env = Environment.from_case_file(CASE_PATH)
        env.reset()
        env.step(Action("A001"))
        env.step(Action("A006", {"target_company": "宏基建设集团股份有限公司"}))
        from environment.actions import make_final_action
        env.step(make_final_action(
            channel_id="CH_INSPECTION_ONLY",
            channel_name="行政监察",
            advisor_reasoning="先走监察",
            drafted_documents=[],
            evidence_ids=["E001", "E002", "E003", "E005", "E006"],
            respondents=["成都恒达劳务有限公司", "宏基建设集团股份有限公司"],
        ))

        from judge.judgment_engine import JudgmentEngine
        judgment = JudgmentEngine(llm=BadJudgeLLM()).adjudicate(
            env.state,
            env.case_data,
            advisor_name="anxin",
            channel_id="CH_INSPECTION_ONLY",
            final_submission=env.state.final_submission,
        )

        self.assertEqual(judgment.monetary_award.principal, 76600)
        self.assertGreater(judgment.monetary_award.total, 0)
        self.assertIn("规则兜底", " ".join(judgment.critical_misses))

    def test_fastapi_session_contract_with_mock_llm(self):
        from fastapi.testclient import TestClient
        from api.server import app

        with patch("llm.client.LLMClient.from_env", return_value=FakeLLM()):
            client = TestClient(app)
            start = client.post("/sessions", json={"advisor_type": "anxin", "max_turns": 1})
            self.assertEqual(start.status_code, 200)
            sid = start.json()["session_id"]

            turn = client.post(f"/sessions/{sid}/turn")
            self.assertEqual(turn.status_code, 200)
            self.assertTrue(turn.json()["is_final_turn"])

            final = client.post(f"/sessions/{sid}/finalize")
            self.assertEqual(final.status_code, 200)
            body = final.json()
            self.assertEqual(body["channel_id"], "CH_ARBITRATION")
            self.assertIn("judgment", body)

    def test_doubao_pipeline_runs_with_mock_llm(self):
        """Run a full Doubao pipeline and verify baseline characteristics."""
        from advisor.doubao_advisor import DoubaoAdvisor

        FakeLLM.action_index = 0
        with patch("llm.client.LLMClient.from_env", return_value=FakeLLM()):
            env = Environment.from_case_file(CASE_PATH)
            worker = SimulatedWorker()
            advisor = DoubaoAdvisor()
            result = EpisodeRunner(env, worker, advisor, max_turns=3, verbose=False).run()

        self.assertIsNotNone(result.final_judgment)
        self.assertEqual(result.advisor_name, "doubao")

        for turn in result.transcript:
            self.assertEqual(turn.advisor_hints, [],
                             f"Doubao no hints at turn {turn.turn_index}")
        self.assertEqual(result.total_turns, 3)


if __name__ == "__main__":
    unittest.main()
