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
            result = EpisodeRunner(env, worker, advisor, max_turns=3, verbose=False).run()

        self.assertIsNotNone(result.final_judgment)
        self.assertIn("E006", env.state.evidence_pool)
        self.assertIn("E009", env.state.evidence_pool)
        self.assertTrue(env.state.flags["limit_order_issued"])
        self.assertEqual(env.state.procedural_stage, ProceduralStage.ARBITRATION_FILED)
        self.assertEqual(len(env.state.npc_interactions), 1)


if __name__ == "__main__":
    unittest.main()
