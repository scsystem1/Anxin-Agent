#!/usr/bin/env python3
"""
Flask web server for Anxin Sandbox demo.

Usage (from project root):
    pip install flask flask-cors
    python server.py

Then open: http://localhost:8080
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from enum import Enum

try:
    from flask import Flask, jsonify, request, send_from_directory
    from flask_cors import CORS
except ImportError:
    print("Please install: pip install flask flask-cors")
    sys.exit(1)

# Ensure project root is on path so all sandbox imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import Environment
from worker.simulated_worker import SimulatedWorker
from advisor.anxin_advisor import AnxinAdvisor
from advisor.doubao_advisor import DoubaoAdvisor
from runner.episode import EpisodeRunner
from runner.comparison import run_comparison

app = Flask(__name__, static_folder="frontend", static_url_path="/static")
CORS(app)


# ---------------------------------------------------------------------------
# Serialization — converts dataclasses / Enums / dates to plain JSON-safe types
# ---------------------------------------------------------------------------

def _to_plain(obj):
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_plain(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(i) for i in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def serialize_episode(result) -> dict:
    return _to_plain(result)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/cases")
def list_cases():
    result = []
    cases_dir = os.path.join(os.path.dirname(__file__), "cases")
    for fname in sorted(os.listdir(cases_dir)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(cases_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            result.append({
                "id": data["case_id"],
                "name": data["case_name"],
                "file": f"cases/{fname}",
                "total_owed": data["financial"]["total_owed"],
                "worker": data["worker"]["name"],
                "location": data["ground_truth"]["project"]["location"],
            })
        except Exception:
            pass
    return jsonify(result)


@app.route("/api/run", methods=["POST"])
def run_sim():
    body = request.get_json(force=True) or {}
    case_file = body.get("case", "cases/tianjiao_mingyuan.json")
    advisor = body.get("advisor", "both")
    max_turns = min(int(body.get("max_turns", 20)), 30)

    print(f"\n[API] /run  advisor={advisor}  max_turns={max_turns}  case={case_file}", flush=True)

    try:
        if advisor == "both":
            report = run_comparison(case_file, max_turns=max_turns, verbose=True)
            return jsonify({
                "case_name": report.case_name,
                "anxin":  serialize_episode(report.anxin_result),
                "doubao": serialize_episode(report.doubao_result),
            })
        else:
            env = Environment.from_case_file(case_file)
            adv = AnxinAdvisor() if advisor == "anxin" else DoubaoAdvisor()
            ep = EpisodeRunner(env, SimulatedWorker(), adv, max_turns, verbose=True).run()
            return jsonify({
                "case_name": env.case_data["case_name"],
                advisor: serialize_episode(ep),
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"\n{'='*55}")
    print(f"  安薪沙盒 Demo  →  http://localhost:{port}")
    print(f"  确保 .env 已填写 ANXIN_LLM_API_KEY 等变量")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
