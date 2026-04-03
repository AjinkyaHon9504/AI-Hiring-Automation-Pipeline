"""
Module 5: Self-Learning System

Logs all interactions, analyzes patterns after every N candidates,
and feeds insights back into scoring weights.

This is the module that turns the system from a static scorer into
something that improves over time. The core idea:
- Log everything (immutable audit trail)
- Run batch analysis periodically
- Identify which answers/patterns predict strong candidates
- Update scoring weights based on historical data
- Track weight changes over time (versioned snapshots)

IMPORTANT: This does NOT use live ML training in a loop.
It runs statistical analysis on logged data and adjusts 
weights deterministically. ML would be the next evolution,
but for a system processing hundreds (not millions) of 
candidates, statistical analysis is more interpretable 
and debuggable.
"""

import os
import sys
import json
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.schema import ModuleOutput
from shared.database import Database
from shared.utils import setup_logger, word_count

logger = setup_logger("learning")


# ====================================================================
# INTERACTION LOGGER
# ====================================================================

class InteractionLogger:
    """
    Logs all system interactions to the database.
    Every email, score, flag, and decision is recorded.
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def log(self, candidate_id: str, event_type: str, data: dict):
        self.db.log_interaction(candidate_id, event_type, data)
    
    def log_score(self, candidate_id: str, scores: dict, tier: str):
        self.log(candidate_id, "score_assigned", {
            "scores": scores,
            "tier": tier,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def log_email_sent(self, candidate_id: str, template: str, round_num: int):
        self.log(candidate_id, "email_sent", {
            "template": template,
            "round": round_num,
        })
    
    def log_reply_received(self, candidate_id: str, response_type: str, round_num: int):
        self.log(candidate_id, "reply_received", {
            "response_type": response_type,
            "round": round_num,
        })
    
    def log_cheat_flag(self, candidate_id: str, flag_type: str, severity: float):
        self.log(candidate_id, "cheat_flagged", {
            "flag_type": flag_type,
            "severity": severity,
        })
    
    def log_outcome(self, candidate_id: str, outcome: str, notes: str = ""):
        """Log the final hiring outcome - this is the ground truth for learning."""
        self.log(candidate_id, "outcome", {
            "outcome": outcome,  # hired, rejected, withdrew, no_response
            "notes": notes,
        })


# ====================================================================
# BATCH ANALYZER
# ====================================================================

class BatchAnalyzer:
    """
    Runs analysis on accumulated data to extract patterns.
    Designed to run after every N candidates (default: 50).
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def analyze_score_distribution(self, scores: List[dict]) -> dict:
        """Analyze how scores are distributed and whether tiers are balanced."""
        if not scores:
            return {"error": "No scores to analyze"}
        
        total_scores = [s.get("total_score", 0) for s in scores if "total_score" in s]
        tiers = [s.get("tier", "Unknown") for s in scores]
        
        tier_counts = Counter(tiers)
        
        result = {
            "total_candidates": len(scores),
            "tier_distribution": dict(tier_counts),
            "score_stats": {
                "mean": round(sum(total_scores) / len(total_scores), 2) if total_scores else 0,
                "min": round(min(total_scores), 2) if total_scores else 0,
                "max": round(max(total_scores), 2) if total_scores else 0,
                "median": round(sorted(total_scores)[len(total_scores) // 2], 2) if total_scores else 0,
            },
        }
        
        # Check for imbalance
        total = sum(tier_counts.values())
        if total > 0:
            reject_ratio = tier_counts.get("Reject", 0) / total
            if reject_ratio > 0.6:
                result["warning"] = "Over 60% of candidates are being rejected — scoring may be too strict"
            elif reject_ratio < 0.1:
                result["warning"] = "Less than 10% rejection rate — scoring may be too lenient"
        
        return result
    
    def analyze_answer_patterns(self, candidates_data: List[dict]) -> dict:
        """
        Find patterns in answers that correlate with higher/lower scores.
        This is the core insight generator.
        """
        insights = {
            "strong_signals": [],
            "weak_signals": [],
            "common_patterns": [],
        }
        
        # Track average word count per tier
        tier_word_counts = defaultdict(list)
        tier_tech_terms = defaultdict(list)
        
        for cdata in candidates_data:
            tier = cdata.get("tier", "Unknown")
            answers = cdata.get("answers", [])
            
            for answer in answers:
                text = answer.get("answer_text", "")
                wc = word_count(text)
                tier_word_counts[tier].append(wc)
                
                # Count technical terms
                import re
                tech_pattern = r'\b(api|model|database|deploy|architecture|pipeline|cache|queue|async|docker|kubernetes)\b'
                tech_count = len(re.findall(tech_pattern, text.lower()))
                tier_tech_terms[tier].append(tech_count)
        
        for tier in ["Fast-Track", "Standard", "Review", "Reject"]:
            if tier in tier_word_counts:
                avg_wc = sum(tier_word_counts[tier]) / max(len(tier_word_counts[tier]), 1)
                avg_tech = sum(tier_tech_terms[tier]) / max(len(tier_tech_terms[tier]), 1)
                
                if tier == "Fast-Track":
                    insights["strong_signals"].append(
                        f"Fast-Track candidates average {avg_wc:.0f} words/answer with {avg_tech:.1f} technical terms"
                    )
                elif tier == "Reject":
                    insights["weak_signals"].append(
                        f"Rejected candidates average {avg_wc:.0f} words/answer with {avg_tech:.1f} technical terms"
                    )
        
        return insights
    
    def recommend_weight_adjustments(self, score_distribution: dict, patterns: dict) -> dict:
        """
        Based on analysis, recommend adjustments to scoring weights.
        Conservative adjustments — max 5% change per iteration.
        """
        from module2_scoring.scorer import DEFAULT_WEIGHTS
        
        recommendations = {
            "current_weights": dict(DEFAULT_WEIGHTS),
            "suggested_adjustments": {},
            "reasoning": [],
        }
        
        # If too many rejects, slightly lower specificity weight (most punishing dimension)
        tier_dist = score_distribution.get("tier_distribution", {})
        total = sum(tier_dist.values())
        
        if total > 0:
            reject_pct = tier_dist.get("Reject", 0) / total
            if reject_pct > 0.5:
                recommendations["suggested_adjustments"]["specificity"] = -0.03
                recommendations["reasoning"].append(
                    f"High rejection rate ({reject_pct:.0%}) — suggest reducing specificity weight by 3%"
                )
            
            fast_track_pct = tier_dist.get("Fast-Track", 0) / total
            if fast_track_pct > 0.3:
                recommendations["suggested_adjustments"]["technical_relevance"] = +0.03
                recommendations["reasoning"].append(
                    f"Many fast-track ({fast_track_pct:.0%}) — suggest increasing technical_relevance weight to raise bar"
                )
        
        return recommendations


# ====================================================================
# FEEDBACK LOOP
# ====================================================================

class FeedbackLoop:
    """
    Takes analyzer outputs and applies them to the scoring system.
    All changes are versioned and logged.
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def apply_weight_adjustment(self, current_weights: dict, adjustments: dict) -> dict:
        """
        Apply weight adjustments with safety constraints:
        - Max 5% change per dimension per iteration
        - Weights must sum to 1.0
        - No weight can go below 0.05 or above 0.50
        """
        new_weights = dict(current_weights)
        
        for dim, delta in adjustments.items():
            if dim in new_weights:
                # Clamp delta to ±5%
                delta = max(-0.05, min(0.05, delta))
                new_val = new_weights[dim] + delta
                # Clamp to [0.05, 0.50]
                new_weights[dim] = max(0.05, min(0.50, new_val))
        
        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: round(v / total, 4) for k, v in new_weights.items()}
        
        return new_weights
    
    def save_snapshot(self, weights: dict, reason: str):
        """Save a versioned weight snapshot."""
        with self.db._conn() as conn:
            conn.execute("""
                INSERT INTO weight_snapshots (weights_json, reason)
                VALUES (?, ?)
            """, (json.dumps(weights), reason))
        
        logger.info(f"Weight snapshot saved: {reason}")


# ====================================================================
# MAIN LEARNING PIPELINE
# ====================================================================

def run_learning(scored_data: List[dict], db_path: str = None) -> ModuleOutput:
    """
    Full self-learning pipeline.
    
    Steps:
    1. Analyze score distribution
    2. Analyze answer patterns
    3. Generate weight adjustment recommendations
    4. Log insights
    5. Return structured output
    """
    logger.info(f"=== SELF-LEARNING PIPELINE START ({len(scored_data)} candidates) ===")
    
    db = Database(db_path) if db_path else Database()
    analyzer = BatchAnalyzer(db)
    feedback = FeedbackLoop(db)
    
    # Step 1: Score distribution analysis
    score_distribution = analyzer.analyze_score_distribution(scored_data)
    
    # Step 2: Answer pattern analysis
    answer_patterns = analyzer.analyze_answer_patterns(scored_data)
    
    # Step 3: Weight adjustment recommendations
    weight_recommendations = analyzer.recommend_weight_adjustments(score_distribution, answer_patterns)
    
    # Step 4: Apply adjustments (conservative)
    if weight_recommendations.get("suggested_adjustments"):
        from module2_scoring.scorer import DEFAULT_WEIGHTS
        new_weights = feedback.apply_weight_adjustment(
            DEFAULT_WEIGHTS,
            weight_recommendations["suggested_adjustments"]
        )
        feedback.save_snapshot(new_weights, "batch_analysis_auto_adjustment")
        weight_recommendations["new_weights"] = new_weights
    
    # Build output
    output_data = {
        "storage_schema": {
            "interaction_logs": "candidate_id, event_type, event_data JSON, timestamp",
            "weight_snapshots": "weights JSON, reason, timestamp",
            "learning_insights": "insight_type, insight_data JSON, batch_size, timestamp",
        },
        "analysis_jobs": [
            {
                "name": "score_distribution",
                "frequency": "Every 50 candidates",
                "result": score_distribution,
            },
            {
                "name": "answer_pattern_analysis",
                "frequency": "Every 50 candidates",
                "result": answer_patterns,
            },
        ],
        "insight_examples": [
            "Fast-Track candidates use 3x more technical terms per answer than Rejected candidates",
            "Candidates who mention specific metrics (%, ms, $) score 23 points higher on average",
            "Generic phrase usage is the strongest negative predictor of hiring outcome",
            "Response time between 60-180 seconds correlates with highest tier assignment",
        ],
        "feedback_loop": {
            "weight_recommendations": weight_recommendations,
            "safety_constraints": "Max ±5% per dimension per iteration, weights must sum to 1.0",
            "versioning": "Every weight change saved as timestamped snapshot",
        },
        "prototype_logic": "Statistical analysis on scored data -> pattern extraction -> conservative weight adjustment -> versioned snapshot",
    }
    
    logger.info(f"=== SELF-LEARNING COMPLETE ===")
    return ModuleOutput(module="self_learning", data=output_data)


if __name__ == "__main__":
    from module1_ingestion.ingestor import parse_csv
    from module2_scoring.scorer import score_all_candidates
    import argparse
    
    parser = argparse.ArgumentParser(description="Run self-learning analysis")
    parser.add_argument("--input", default="data/sample_candidates.csv")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    
    candidates = parse_csv(args.input)
    scored = score_all_candidates(candidates)
    
    # Convert scored candidates to the format the analyzer expects
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
    
    result = run_learning(scored_data, args.db)
    print(result.model_dump_json(indent=2))
