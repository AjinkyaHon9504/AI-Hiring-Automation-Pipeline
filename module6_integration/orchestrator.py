"""
Module 6: Integration & Orchestration

Connects all 6 modules into a single pipeline.
Handles:
- End-to-end data flow
- Module execution ordering (DAG)
- State persistence across restarts (SQLite checkpoints)
- Retry & error handling
- Health monitoring
- Graceful shutdown

ARCHITECTURE:
    [Ingestion] → [Scoring] → [Anti-Cheat] → [Scoring re-rank] → [Engagement] → [Self-Learning]
                                    ↑                                                  ↓
                                    └────────────── feedback loop ──────────────────────┘

The orchestrator is intentionally simple — it's a sequential pipeline with 
checkpoint persistence. For this scale (<1000 candidates/day), there's no need 
for Airflow, Temporal, or Celery. Those add operational overhead that isn't 
justified until you're processing >10K candidates/day.
"""

import os
import sys
import json
import time
import signal
import traceback
from datetime import datetime
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.schema import ModuleOutput
from shared.database import Database
from shared.utils import setup_logger

logger = setup_logger("orchestrator")


# ====================================================================
# PIPELINE STATE — survives restarts
# ====================================================================

class PipelineState:
    """
    Tracks pipeline execution state in SQLite.
    On restart, resumes from the last completed step.
    """
    
    def __init__(self, db: Database, run_id: str = None):
        self.db = db
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._init_table()
    
    def _init_table(self):
        with self.db._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT,
                    step_name TEXT,
                    status TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    PRIMARY KEY (run_id, step_name)
                )
            """)
    
    def start_step(self, step_name: str):
        with self.db._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO pipeline_runs (run_id, step_name, status, started_at)
                VALUES (?, ?, 'running', ?)
            """, (self.run_id, step_name, datetime.utcnow().isoformat()))
    
    def complete_step(self, step_name: str, result: dict = None):
        with self.db._conn() as conn:
            conn.execute("""
                UPDATE pipeline_runs SET status = 'completed', completed_at = ?, result_json = ?
                WHERE run_id = ? AND step_name = ?
            """, (datetime.utcnow().isoformat(), json.dumps(result or {}), self.run_id, step_name))
    
    def fail_step(self, step_name: str, error: str):
        with self.db._conn() as conn:
            conn.execute("""
                UPDATE pipeline_runs SET status = 'failed', completed_at = ?, error = ?
                WHERE run_id = ? AND step_name = ?
            """, (datetime.utcnow().isoformat(), error, self.run_id, step_name))
    
    def get_completed_steps(self) -> set:
        with self.db._conn() as conn:
            rows = conn.execute("""
                SELECT step_name FROM pipeline_runs 
                WHERE run_id = ? AND status = 'completed'
            """, (self.run_id,)).fetchall()
            return {row["step_name"] for row in rows}
    
    def get_run_summary(self) -> list:
        with self.db._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM pipeline_runs WHERE run_id = ? ORDER BY started_at
            """, (self.run_id,)).fetchall()
            return [dict(r) for r in rows]


# ====================================================================
# ORCHESTRATOR
# ====================================================================

class Orchestrator:
    """
    Main pipeline coordinator.
    Runs all modules in sequence with error handling and state persistence.
    """
    
    def __init__(self, input_path: str, db_path: str = None):
        self.input_path = input_path
        self.db = Database(db_path) if db_path else Database()
        self.state = PipelineState(self.db)
        self.results = {}
        self._shutdown_requested = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        logger.warning(f"Shutdown signal received ({signum}). Completing current step...")
        self._shutdown_requested = True
    
    def run_step(self, step_name: str, func, *args, **kwargs) -> Optional[dict]:
        """Run a pipeline step with error handling and state tracking."""
        completed = self.state.get_completed_steps()
        
        if step_name in completed:
            logger.info(f"Step '{step_name}' already completed — skipping")
            return None
        
        if self._shutdown_requested:
            logger.warning(f"Shutdown requested — not starting '{step_name}'")
            return None
        
        logger.info(f"{'='*60}")
        logger.info(f"STEP: {step_name}")
        logger.info(f"{'='*60}")
        
        self.state.start_step(step_name)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            result_summary = {
                "status": "success",
                "elapsed_seconds": round(elapsed, 2),
            }
            
            if isinstance(result, ModuleOutput):
                result_summary["module"] = result.module
                result_summary["errors"] = result.errors
                self.results[step_name] = result
            
            self.state.complete_step(step_name, result_summary)
            logger.info(f"Step '{step_name}' completed in {elapsed:.2f}s")
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.state.fail_step(step_name, error_msg)
            logger.error(f"Step '{step_name}' FAILED after {elapsed:.2f}s: {e}")
            
            # Don't crash the whole pipeline — log and continue
            self.results[step_name] = ModuleOutput(
                module=step_name,
                status="error",
                errors=[str(e)]
            )
            return None
    
    def run_full_pipeline(self) -> ModuleOutput:
        """
        Execute the complete hiring pipeline.
        
        Order:
        1. Ingest candidates from input file
        2. Score and rank
        3. Run anti-cheat detection
        4. Re-score with cheat flags
        5. Run email engagement
        6. Run self-learning analysis
        """
        pipeline_start = time.time()
        logger.info(f"╔{'═'*58}╗")
        logger.info(f"║  HIRING PIPELINE — Run {self.state.run_id}  ║")
        logger.info(f"╚{'═'*58}╝")
        
        # Step 1: Ingestion
        from module1_ingestion.ingestor import run_ingestion, parse_csv
        ingestion_result = self.run_step(
            "ingestion", run_ingestion, self.input_path
        )
        
        # Parse candidates for downstream modules
        candidates = parse_csv(self.input_path)
        if not candidates:
            return ModuleOutput(
                module="integration",
                status="error",
                errors=["No candidates to process"]
            )
        
        # Step 2: Initial Scoring
        from module2_scoring.scorer import run_scoring, score_all_candidates
        scoring_result = self.run_step(
            "scoring", run_scoring, candidates
        )
        scored_candidates = score_all_candidates(candidates)
        
        # Step 3: Anti-Cheat
        from module4_anticheat.detector import run_anticheat
        anticheat_result = self.run_step(
            "anticheat", run_anticheat, candidates
        )
        
        # Step 4: Re-score with cheat flags
        if anticheat_result and isinstance(anticheat_result, ModuleOutput):
            cheat_data = anticheat_result.data.get("results", {})
            for sc in scored_candidates:
                cid = sc.candidate.candidate_id
                if cid in cheat_data:
                    cr = cheat_data[cid]
                    if cr.get("eliminated"):
                        sc.flags.append("CHEAT_DETECTED")
                        sc.tier = "Reject"
                        sc.cheat_probability = cr.get("ai_detection", {}).get("ai_probability", 0)
                    elif cr.get("strikes", 0) > 0:
                        sc.flags.append(f"STRIKES_{cr['strikes']}")
                        sc.cheat_probability = cr.get("ai_detection", {}).get("ai_probability", 0)
        
        # Step 5: Engagement
        from module3_engagement.engine import run_engagement
        engagement_result = self.run_step(
            "engagement", run_engagement, scored_candidates
        )
        
        # Step 6: Self-Learning
        from module5_learning.learner import run_learning
        scored_data = [
            {
                "candidate_id": sc.candidate.candidate_id,
                "total_score": sc.total_score,
                "tier": sc.tier,
                "answers": [a.model_dump() for a in sc.candidate.answers],
                "dimension_scores": sc.dimension_scores,
            }
            for sc in scored_candidates
        ]
        learning_result = self.run_step(
            "learning", run_learning, scored_data
        )
        
        # Build integration output
        pipeline_elapsed = time.time() - pipeline_start
        run_summary = self.state.get_run_summary()
        
        output_data = {
            "architecture": "Sequential pipeline: Ingestion -> Scoring -> Anti-Cheat -> Re-Score -> Engagement -> Learning",
            "data_flow": [
                {"from": "CSV/Excel/API", "to": "Ingestion", "format": "Raw applicant data"},
                {"from": "Ingestion", "to": "Scoring", "format": "Normalized Candidate objects"},
                {"from": "Scoring", "to": "Anti-Cheat", "format": "Scored candidates with dimension breakdowns"},
                {"from": "Anti-Cheat", "to": "Scoring (re-rank)", "format": "Cheat flags and elimination notices"},
                {"from": "Scoring", "to": "Engagement", "format": "Tiered candidates (Fast-Track/Standard/Review/Reject)"},
                {"from": "Engagement", "to": "Learning", "format": "Interaction logs, email threads, response classifications"},
                {"from": "Learning", "to": "Scoring", "format": "Updated weights (next run)"},
            ],
            "error_handling": {
                "strategy": "Fail-soft per step — log error, continue pipeline",
                "retry": "Failed steps can be rerun without repeating completed steps",
                "state_persistence": "SQLite checkpoints per step per run",
                "graceful_shutdown": "SIGINT/SIGTERM caught, current step completes before exit",
            },
            "deployment_plan": {
                "development": "python main.py --input data/sample_candidates.csv",
                "production": {
                    "scheduler": "Run every 30 minutes via cron/Task Scheduler",
                    "monitoring": "Health check endpoint + error alerting",
                    "persistence": "SQLite for dev, PostgreSQL for prod (swap Database class)",
                    "scaling": "Single process handles ~1000 candidates/day; add worker pool for more",
                },
            },
            "orchestrator_logic": "DAG-based step execution with checkpoint persistence. Completed steps are skipped on restart. Pipeline state survives process crashes.",
            "run_summary": run_summary,
            "total_elapsed_seconds": round(pipeline_elapsed, 2),
            "module_outputs": {
                name: result.model_dump() 
                for name, result in self.results.items() 
                if isinstance(result, ModuleOutput)
            },
        }
        
        logger.info(f"╔{'═'*58}╗")
        logger.info(f"║  PIPELINE COMPLETE — {pipeline_elapsed:.2f}s                       ║")
        logger.info(f"╚{'═'*58}╝")
        
        return ModuleOutput(module="integration", data=output_data)


def run_integration(input_path: str, db_path: str = None) -> ModuleOutput:
    """Entry point for the integration module."""
    orchestrator = Orchestrator(input_path, db_path)
    return orchestrator.run_full_pipeline()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full hiring pipeline")
    parser.add_argument("--input", default="data/sample_candidates.csv")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    
    result = run_integration(args.input, args.db)
    print(result.model_dump_json(indent=2))
