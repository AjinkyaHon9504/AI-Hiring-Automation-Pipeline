"""
GTHR Web Dashboard — Flask API Backend

Serves the dashboard UI and provides API endpoints for:
- Running the pipeline
- Viewing candidates, scores, anti-cheat results
- Uploading CSV files
- Real-time pipeline status
"""

import os
import sys
import json
import uuid
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from shared.schema import Candidate, ScoredCandidate, ModuleOutput
from shared.database import Database
from shared.utils import setup_logger
from module1_ingestion.ingestor import parse_csv, run_ingestion
from module2_scoring.scorer import score_all_candidates, run_scoring, DEFAULT_WEIGHTS
from module4_anticheat.detector import (
    run_anticheat, detect_ai_generated, SimilarityDetector, analyze_timing
)
from module3_engagement.engine import (
    run_engagement, classify_response, decide_next_action,
    generate_email, QUESTIONS_BY_ROUND
)
from module5_learning.learner import run_learning, BatchAnalyzer, FeedbackLoop

logger = setup_logger("web")

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
CORS(app)

# Upload folder
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Pipeline state (in-memory for demo)
pipeline_state = {
    "status": "idle",  # idle, running, completed, error
    "current_step": "",
    "progress": 0,
    "steps_completed": [],
    "results": {},
    "candidates": [],
    "scored": [],
    "anticheat": {},
    "engagement": {},
    "learning": {},
    "input_file": os.path.join(ROOT_DIR, "data", "sample_candidates.csv"),
    "run_id": None,
    "started_at": None,
    "completed_at": None,
    "error": None,
}


def reset_state():
    pipeline_state.update({
        "status": "idle",
        "current_step": "",
        "progress": 0,
        "steps_completed": [],
        "results": {},
        "candidates": [],
        "scored": [],
        "anticheat": {},
        "engagement": {},
        "learning": {},
        "run_id": None,
        "started_at": None,
        "completed_at": None,
        "error": None,
    })


def run_pipeline_async(input_file):
    """Run the full pipeline in a background thread."""
    try:
        pipeline_state["status"] = "running"
        pipeline_state["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_state["started_at"] = datetime.now().isoformat()
        pipeline_state["input_file"] = input_file

        db_path = os.path.join(ROOT_DIR, f"gthr_web_{pipeline_state['run_id']}.db")

        # Step 1: Ingestion
        pipeline_state["current_step"] = "ingestion"
        pipeline_state["progress"] = 10
        candidates = parse_csv(input_file)
        pipeline_state["candidates"] = [c.to_json() for c in candidates]
        pipeline_state["steps_completed"].append("ingestion")
        pipeline_state["progress"] = 20

        # Step 2: Scoring
        pipeline_state["current_step"] = "scoring"
        pipeline_state["progress"] = 30
        scored = score_all_candidates(candidates)
        pipeline_state["scored"] = [
            {
                "candidate_id": sc.candidate.candidate_id,
                "name": sc.candidate.name,
                "email": sc.candidate.email,
                "total_score": sc.total_score,
                "tier": sc.tier,
                "dimension_scores": sc.dimension_scores,
                "explanation": sc.explanation[:8],
                "flags": sc.flags,
                "cheat_probability": sc.cheat_probability,
                "github_url": sc.candidate.profile.github_url,
                "linkedin_url": sc.candidate.profile.linkedin_url,
                "response_time": sc.candidate.metadata.response_time_seconds,
                "answers": [
                    {
                        "question_id": a.question_id,
                        "question_text": a.question_text,
                        "answer_text": a.answer_text,
                    }
                    for a in sc.candidate.answers
                ],
            }
            for sc in scored
        ]
        pipeline_state["steps_completed"].append("scoring")
        pipeline_state["progress"] = 50

        # Step 3: Anti-Cheat
        pipeline_state["current_step"] = "anticheat"
        pipeline_state["progress"] = 60

        # AI detection per candidate
        ai_results = {}
        for c in candidates:
            all_text = " ".join(a.answer_text for a in c.answers if a.answer_text)
            ai_results[c.candidate_id] = detect_ai_generated(all_text)

        # Similarity
        detector = SimilarityDetector()
        similar_pairs = detector.find_similar_pairs(candidates, threshold=0.75)

        # Timing
        timing_results = {}
        for c in candidates:
            timing_results[c.candidate_id] = analyze_timing(c)

        pipeline_state["anticheat"] = {
            "ai_detection": {cid: r for cid, r in ai_results.items()},
            "similar_pairs": similar_pairs,
            "timing": {cid: r for cid, r in timing_results.items()},
            "method": detector.method,
        }
        pipeline_state["steps_completed"].append("anticheat")
        pipeline_state["progress"] = 75

        # Step 4: Engagement
        pipeline_state["current_step"] = "engagement"
        pipeline_state["progress"] = 80
        engagement_result = run_engagement(scored, simulate=True)
        pipeline_state["engagement"] = engagement_result.data
        pipeline_state["steps_completed"].append("engagement")
        pipeline_state["progress"] = 90

        # Step 5: Learning
        pipeline_state["current_step"] = "learning"
        pipeline_state["progress"] = 95
        scored_data = [
            {
                "candidate_id": sc.candidate.candidate_id,
                "total_score": sc.total_score,
                "tier": sc.tier,
                "answers": [a.model_dump() for a in sc.candidate.answers],
                "dimension_scores": sc.dimension_scores,
            }
            for sc in scored
        ]
        learning_result = run_learning(scored_data)
        pipeline_state["learning"] = learning_result.data
        pipeline_state["steps_completed"].append("learning")

        # Done
        pipeline_state["status"] = "completed"
        pipeline_state["progress"] = 100
        pipeline_state["current_step"] = "done"
        pipeline_state["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        pipeline_state["status"] = "error"
        pipeline_state["error"] = str(e)
        logger.error(f"Pipeline error: {e}", exc_info=True)


# ================================================================
# ROUTES
# ================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "status": pipeline_state["status"],
        "current_step": pipeline_state["current_step"],
        "progress": pipeline_state["progress"],
        "steps_completed": pipeline_state["steps_completed"],
        "run_id": pipeline_state["run_id"],
        "started_at": pipeline_state["started_at"],
        "completed_at": pipeline_state["completed_at"],
        "error": pipeline_state["error"],
        "candidate_count": len(pipeline_state["candidates"]),
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    if pipeline_state["status"] == "running":
        return jsonify({"error": "Pipeline already running"}), 400

    reset_state()

    # Check for uploaded file
    input_file = pipeline_state["input_file"]
    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            input_file = filepath

    # Run in background thread
    thread = threading.Thread(target=run_pipeline_async, args=(input_file,))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "message": "Pipeline started"})


@app.route("/api/run-sample", methods=["POST"])
def api_run_sample():
    if pipeline_state["status"] == "running":
        return jsonify({"error": "Pipeline already running"}), 400

    reset_state()
    sample = os.path.join(ROOT_DIR, "data", "sample_candidates.csv")

    thread = threading.Thread(target=run_pipeline_async, args=(sample,))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started"})


@app.route("/api/candidates")
def api_candidates():
    return jsonify(pipeline_state["candidates"])


@app.route("/api/scores")
def api_scores():
    return jsonify(pipeline_state["scored"])


@app.route("/api/anticheat")
def api_anticheat():
    return jsonify(pipeline_state["anticheat"])


@app.route("/api/engagement")
def api_engagement():
    return jsonify(pipeline_state["engagement"])


@app.route("/api/learning")
def api_learning():
    return jsonify(pipeline_state["learning"])


@app.route("/api/candidate/<candidate_id>")
def api_candidate_detail(candidate_id):
    """Get full details for a single candidate."""
    candidate = next((c for c in pipeline_state["candidates"] if c["candidate_id"] == candidate_id), None)
    scored = next((s for s in pipeline_state["scored"] if s["candidate_id"] == candidate_id), None)
    anticheat_ai = pipeline_state.get("anticheat", {}).get("ai_detection", {}).get(candidate_id)
    anticheat_timing = pipeline_state.get("anticheat", {}).get("timing", {}).get(candidate_id)
    similar = [
        p for p in pipeline_state.get("anticheat", {}).get("similar_pairs", [])
        if p.get("candidate_a") == candidate_id or p.get("candidate_b") == candidate_id
    ]
    return jsonify({
        "candidate": candidate,
        "scores": scored,
        "ai_detection": anticheat_ai,
        "timing_analysis": anticheat_timing,
        "similarity_flags": similar,
    })


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Live response classifier demo."""
    data = request.json or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    rtype, confidence = classify_response(text)
    action = decide_next_action(text, current_round=data.get("round", 1))
    ai = detect_ai_generated(text)

    return jsonify({
        "response_type": rtype.value,
        "confidence": confidence,
        "action": action,
        "ai_detection": ai,
    })


@app.route("/api/weights")
def api_weights():
    return jsonify(DEFAULT_WEIGHTS)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    reset_state()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  GTHR Dashboard — http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
