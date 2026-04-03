"""
Module 4: Anti-Cheat Detection System

Detects three types of cheating:
1. AI-generated responses (statistical analysis + optional embedding-based detection)
2. Cross-candidate plagiarism (cosine similarity on embeddings)
3. Timing anomalies (suspiciously fast responses)

Uses a strike system: 3 strikes = automatic elimination.

WHAT I TRIED:
- Attempt 1: GPTZero API for AI detection. Problem: rate limit of 10/min 
  on free tier, and accuracy was inconsistent on short text (<100 words).
  Many candidate answers are 50-80 words, so this was unreliable.
  Error: "429 Too Many Requests" after 10 calls. Also $0.01/request.

- Attempt 2: Used perplexity scoring with a local GPT-2 model (huggingface).
  Problem: GPT-2 tokenizer + model download was 600MB, startup took 45s,
  and the perplexity threshold was impossible to calibrate — AI text and 
  well-written human text had similar perplexity scores.
  
  Traceback:
    torch.cuda.OutOfMemoryError: CUDA out of memory. 
    Tried to allocate 20.00 MiB (GPU 0; 4.00 GiB total)
  
  (Local laptop GPU wasn't enough. Could work on a server.)

- Attempt 3 (current): Statistical heuristics + sentence-transformers for 
  similarity. No LLM needed for detection. Uses:
  * Vocabulary richness (hapax legomena ratio)
  * Sentence length variance (AI text is suspiciously uniform)
  * Burstiness score (human text has more varied sentence structure)
  * Cross-candidate cosine similarity using MiniLM embeddings
  
  This works well for catching copy-paste and very obvious AI text.
  Won't catch sophisticated AI-assisted answers — but that's an 
  acknowledged limitation.
"""

import os
import sys
import re
import math
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.schema import Candidate, ScoredCandidate, ModuleOutput
from shared.database import Database
from shared.utils import (
    setup_logger, normalize_text, word_count,
    unique_words_ratio, hapax_legomena_ratio
)

logger = setup_logger("anticheat")

# Try to import sentence-transformers; fall back to sklearn TF-IDF if unavailable
EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    logger.info("sentence-transformers available — using MiniLM for similarity")
except ImportError:
    logger.warning("sentence-transformers not installed — falling back to TF-IDF similarity")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("sklearn not available — similarity detection disabled")


# ====================================================================
# THRESHOLDS — tuned through experimentation on sample data
# ====================================================================
THRESHOLDS = {
    "similarity_flag": 0.92,       # cosine sim above this = likely copy
    "similarity_warn": 0.80,       # above this = suspicious
    "ai_sentence_var_max": 2.5,    # sentence length std dev below this = too uniform
    "ai_hapax_min": 0.25,          # hapax ratio below this in long text = suspicious
    "ai_hapax_max": 0.80,          # hapax ratio above this in short text = suspicious  
    "timing_min_seconds": 10,      # below this = instant paste
    "timing_suspicious": 30,       # below this for complex questions = suspicious
    "strike_limit": 3,             # strikes before elimination
}


# ====================================================================
# SIGNAL 1: AI-Generated Text Detection (Statistical)
# ====================================================================

def analyze_sentence_uniformity(text: str) -> Tuple[float, str]:
    """
    AI-generated text tends to have very uniform sentence lengths.
    Human text is "bursty" — some short sentences, some long ones.
    
    Returns: (score 0-1 where 1 = likely AI, explanation)
    """
    if not text or word_count(text) < 20:
        return 0.0, "Text too short for uniformity analysis"
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 3:
        return 0.0, "Too few sentences for analysis"
    
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    
    # Coefficient of variation — low = uniform = suspicious
    cv = std_len / mean_len if mean_len > 0 else 0
    
    if cv < 0.2:
        return 0.85, f"Very uniform sentence lengths (CV={cv:.2f}) — strong AI signal"
    elif cv < 0.35:
        return 0.5, f"Somewhat uniform sentences (CV={cv:.2f}) — mild AI signal"
    else:
        return 0.1, f"Natural sentence variation (CV={cv:.2f}) — likely human"


def analyze_vocabulary_patterns(text: str) -> Tuple[float, str]:
    """
    AI text patterns:
    - Moderate, consistent hapax legomena ratio
    - Limited use of contractions (I'm, don't, can't)
    - Avoids strong opinions/informal language
    - Over-uses transition words (however, moreover, furthermore)
    """
    if not text or word_count(text) < 20:
        return 0.0, "Text too short for vocabulary analysis"
    
    text_lower = text.lower()
    wc = word_count(text)
    score = 0.0
    signals = []
    
    # Check contractions (humans use more)
    contractions = re.findall(r"\b\w+'\w+\b", text_lower)
    contraction_rate = len(contractions) / max(wc, 1)
    if contraction_rate < 0.01 and wc > 50:
        score += 0.2
        signals.append("no contractions")
    
    # Check AI-typical transition words
    ai_transitions = ["moreover", "furthermore", "additionally", "consequently", 
                       "in conclusion", "it is important to note", "it's worth noting"]
    for phrase in ai_transitions:
        if phrase in text_lower:
            score += 0.15
            signals.append(f"AI-typical phrase: '{phrase}'")
    
    # Hapax legomena ratio
    hlr = hapax_legomena_ratio(text)
    if 0.40 <= hlr <= 0.55 and wc > 80:
        # This specific range is suspiciously "average" — AI text clusters here
        score += 0.15
        signals.append(f"hapax ratio in AI-typical range ({hlr:.2f})")
    
    # Check for overly formal language patterns
    formal_patterns = [
        r'\bthis ensures\b', r'\bthis allows\b', r'\bthis enables\b',
        r'\bit is essential\b', r'\bit is crucial\b',
    ]
    formal_count = sum(1 for p in formal_patterns if re.search(p, text_lower))
    if formal_count >= 2:
        score += 0.2
        signals.append(f"{formal_count} overly formal patterns")
    
    explanation = f"Vocabulary analysis: {', '.join(signals)}" if signals else "No AI vocabulary signals"
    return min(1.0, score), explanation


def detect_ai_generated(text: str) -> Dict:
    """
    Combine all AI detection signals into a composite score.
    """
    uniformity_score, uniformity_expl = analyze_sentence_uniformity(text)
    vocab_score, vocab_expl = analyze_vocabulary_patterns(text)
    
    # Weighted combination
    composite = (uniformity_score * 0.6) + (vocab_score * 0.4)
    
    return {
        "ai_probability": round(composite, 3),
        "signals": {
            "sentence_uniformity": {"score": uniformity_score, "detail": uniformity_expl},
            "vocabulary_patterns": {"score": vocab_score, "detail": vocab_expl},
        },
        "verdict": "likely_ai" if composite > 0.65 else "suspicious" if composite > 0.4 else "likely_human"
    }


# ====================================================================
# SIGNAL 2: Cross-Candidate Similarity (Plagiarism Detection)
# ====================================================================

class SimilarityDetector:
    """
    Detects copied/shared responses across candidates.
    
    Uses sentence-transformers (MiniLM) if available, otherwise TF-IDF.
    MiniLM is better because it catches semantic similarity, not just 
    lexical overlap. E.g., "I built a RAG pipeline" and "I created a 
    retrieval-augmented generation system" would be flagged.
    """
    
    def __init__(self):
        self.model = None
        self.method = "none"
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.method = "sentence-transformers"
                logger.info("Loaded MiniLM model for semantic similarity")
            except Exception as e:
                logger.warning(f"Failed to load MiniLM: {e}")
        
        if self.model is None and SKLEARN_AVAILABLE:
            self.method = "tfidf"
            logger.info("Using TF-IDF for similarity (fallback)")
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise similarity matrix for a list of texts."""
        if not texts or len(texts) < 2:
            return np.zeros((len(texts), len(texts)))
        
        if self.method == "sentence-transformers":
            embeddings = self.model.encode(texts, show_progress_bar=False)
            # Cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # avoid division by zero
            normalized = embeddings / norms
            sim_matrix = np.dot(normalized, normalized.T)
            return sim_matrix
            
        elif self.method == "tfidf":
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            return cosine_similarity(tfidf_matrix).toarray() if hasattr(cosine_similarity(tfidf_matrix), 'toarray') else cosine_similarity(tfidf_matrix)
        
        else:
            return np.zeros((len(texts), len(texts)))
    
    def find_similar_pairs(self, candidates: List[Candidate], 
                           threshold: float = None) -> List[Dict]:
        """
        Find pairs of candidates with suspiciously similar answers.
        Checks each question separately.
        """
        if threshold is None:
            threshold = THRESHOLDS["similarity_warn"]
        
        similar_pairs = []
        
        # Group answers by question_id
        question_groups = {}
        for c in candidates:
            for a in c.answers:
                if a.question_id not in question_groups:
                    question_groups[a.question_id] = []
                if a.answer_text.strip():
                    question_groups[a.question_id].append({
                        "candidate_id": c.candidate_id,
                        "name": c.name,
                        "text": a.answer_text,
                    })
        
        for q_id, entries in question_groups.items():
            if len(entries) < 2:
                continue
            
            texts = [e["text"] for e in entries]
            sim_matrix = self.compute_similarity_matrix(texts)
            
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    sim = sim_matrix[i][j]
                    if sim >= threshold:
                        pair = {
                            "question_id": q_id,
                            "candidate_a": entries[i]["candidate_id"],
                            "candidate_a_name": entries[i]["name"],
                            "candidate_b": entries[j]["candidate_id"],
                            "candidate_b_name": entries[j]["name"],
                            "similarity": round(float(sim), 4),
                            "severity": "high" if sim >= THRESHOLDS["similarity_flag"] else "medium",
                            "text_a_preview": entries[i]["text"][:100],
                            "text_b_preview": entries[j]["text"][:100],
                        }
                        similar_pairs.append(pair)
                        logger.warning(
                            f"SIMILARITY ALERT: {entries[i]['name']} ↔ {entries[j]['name']} "
                            f"on Q{q_id}: {sim:.4f}"
                        )
        
        return similar_pairs


# ====================================================================
# SIGNAL 3: Timing Analysis
# ====================================================================

def analyze_timing(candidate: Candidate) -> Dict:
    """
    Analyze response timing for anomalies.
    
    Suspicious patterns:
    - Response in <10 seconds (instant paste)
    - Response in <30 seconds for technical questions
    - Multiple candidates with identical response times
    """
    resp_time = candidate.metadata.response_time_seconds
    flags = []
    severity = 0.0
    
    if resp_time <= 0:
        return {"flags": [], "severity": 0.0, "detail": "No timing data"}
    
    # Check absolute speed
    if resp_time < THRESHOLDS["timing_min_seconds"]:
        flags.append(f"Response in {resp_time}s — likely copy-paste")
        severity = 0.9
    elif resp_time < THRESHOLDS["timing_suspicious"]:
        # Check answer complexity
        total_words = sum(word_count(a.answer_text) for a in candidate.answers)
        words_per_second = total_words / resp_time if resp_time > 0 else 0
        
        if words_per_second > 5:  # Average typing is 1-2 words/sec
            flags.append(f"Typed {words_per_second:.1f} words/sec — faster than average typing")
            severity = 0.6
    
    return {
        "flags": flags,
        "severity": severity,
        "response_time": resp_time,
        "detail": f"Response time: {resp_time}s"
    }


# ====================================================================
# STRIKE SYSTEM
# ====================================================================

class StrikeSystem:
    """
    Track strikes per candidate. 3 strikes = elimination.
    
    Strike triggers:
    - High similarity with another candidate (1-2 strikes based on severity)
    - AI detection above threshold (1 strike)
    - Timing anomaly (1 strike)
    - Multiple flags on same candidate compound
    """
    
    def __init__(self, db: Database = None):
        self.db = db
        self.strikes = {}  # candidate_id → list of strike reasons
    
    def add_strike(self, candidate_id: str, reason: str, severity: float):
        if candidate_id not in self.strikes:
            self.strikes[candidate_id] = []
        
        # Severity > 0.8 = 2 strikes (e.g., exact copy detected)
        strike_count = 2 if severity > 0.8 else 1
        
        for _ in range(strike_count):
            self.strikes[candidate_id].append({
                "reason": reason,
                "severity": severity,
            })
        
        # Persist to DB
        if self.db:
            self.db.save_cheat_flag(candidate_id, "strike", reason, severity)
        
        total = len(self.strikes[candidate_id])
        logger.info(f"Strike for {candidate_id}: {reason} (total: {total})")
        
        return total >= THRESHOLDS["strike_limit"]
    
    def get_strikes(self, candidate_id: str) -> int:
        return len(self.strikes.get(candidate_id, []))
    
    def is_eliminated(self, candidate_id: str) -> bool:
        return self.get_strikes(candidate_id) >= THRESHOLDS["strike_limit"]


# ====================================================================
# MAIN DETECTION ORCHESTRATOR
# ====================================================================

def run_anticheat(candidates: List[Candidate], db_path: str = None) -> ModuleOutput:
    """
    Full anti-cheat pipeline.
    
    Steps:
    1. Run AI detection on each candidate's answers
    2. Run cross-candidate similarity check
    3. Run timing analysis
    4. Apply strike system
    5. Return results with flags and eliminations
    """
    logger.info(f"=== ANTI-CHEAT PIPELINE START ({len(candidates)} candidates) ===")
    
    db = Database(db_path) if db_path else Database()
    strikes = StrikeSystem(db)
    similarity_detector = SimilarityDetector()
    
    # Results storage
    candidate_results = {}
    
    # Step 1: AI Detection per candidate
    logger.info("--- Phase 1: AI Detection ---")
    for c in candidates:
        all_text = " ".join(a.answer_text for a in c.answers if a.answer_text)
        ai_result = detect_ai_generated(all_text)
        
        candidate_results[c.candidate_id] = {
            "name": c.name,
            "ai_detection": ai_result,
            "similarity_flags": [],
            "timing": {},
            "strikes": 0,
            "eliminated": False,
        }
        
        if ai_result["verdict"] == "likely_ai":
            eliminated = strikes.add_strike(
                c.candidate_id, 
                f"AI-generated text detected (p={ai_result['ai_probability']:.2f})",
                ai_result["ai_probability"]
            )
            if eliminated:
                candidate_results[c.candidate_id]["eliminated"] = True
        
        logger.info(f"  {c.name}: AI verdict={ai_result['verdict']} (p={ai_result['ai_probability']:.3f})")
    
    # Step 2: Cross-candidate similarity
    logger.info("--- Phase 2: Cross-Candidate Similarity ---")
    similar_pairs = similarity_detector.find_similar_pairs(candidates)
    
    for pair in similar_pairs:
        # Add flags to both candidates
        for cid_key in ["candidate_a", "candidate_b"]:
            cid = pair[cid_key]
            if cid in candidate_results:
                candidate_results[cid]["similarity_flags"].append(pair)
        
        # Add strikes
        severity = pair["similarity"]
        reason = f"Similar answer to {pair['candidate_b_name']} on Q{pair['question_id']} (sim={pair['similarity']:.2f})"
        
        eliminated = strikes.add_strike(pair["candidate_a"], reason, severity)
        if eliminated:
            candidate_results[pair["candidate_a"]]["eliminated"] = True
        
        eliminated = strikes.add_strike(pair["candidate_b"], reason, severity)
        if eliminated:
            candidate_results[pair["candidate_b"]]["eliminated"] = True
    
    # Step 3: Timing analysis
    logger.info("--- Phase 3: Timing Analysis ---")
    for c in candidates:
        timing_result = analyze_timing(c)
        candidate_results[c.candidate_id]["timing"] = timing_result
        
        if timing_result["severity"] > 0.5:
            eliminated = strikes.add_strike(
                c.candidate_id,
                f"Timing anomaly: {timing_result['flags'][0]}",
                timing_result["severity"]
            )
            if eliminated:
                candidate_results[c.candidate_id]["eliminated"] = True
    
    # Update strike counts
    for cid in candidate_results:
        candidate_results[cid]["strikes"] = strikes.get_strikes(cid)
        candidate_results[cid]["eliminated"] = strikes.is_eliminated(cid)
    
    # Build output
    eliminated_count = sum(1 for r in candidate_results.values() if r["eliminated"])
    flagged_count = sum(1 for r in candidate_results.values() if r["strikes"] > 0)
    
    output_data = {
        "signals": {
            "ai_detection": "Statistical analysis: sentence uniformity, vocabulary patterns, formality markers",
            "similarity": f"Cross-candidate cosine similarity ({similarity_detector.method})",
            "timing": "Response time anomaly detection",
        },
        "similarity_method": similarity_detector.method,
        "thresholds": THRESHOLDS,
        "strike_system": {
            "strikes_to_eliminate": THRESHOLDS["strike_limit"],
            "high_severity_double_strike": True,
            "rules": "High similarity (>0.8) = 2 strikes, AI detection = 1 strike, timing = 1 strike"
        },
        "results": candidate_results,
        "similar_pairs": similar_pairs,
        "summary": {
            "total_candidates": len(candidates),
            "flagged": flagged_count,
            "eliminated": eliminated_count,
            "clean": len(candidates) - flagged_count,
        },
        "false_positive_risks": [
            "Candidates with similar backgrounds may use similar technical terms — similarity threshold should be high (>0.92 for flag)",
            "Well-written human text may trigger AI detection — use as one signal, not sole decider",
            "Fast typers or prepared candidates may trigger timing flags — should combine with other signals",
            "Short answers have less signal — both AI detection and similarity are less reliable on <20 words",
        ]
    }
    
    logger.info(f"=== ANTI-CHEAT COMPLETE — Flagged: {flagged_count}, Eliminated: {eliminated_count} ===")
    return ModuleOutput(module="anti_cheat", data=output_data)


if __name__ == "__main__":
    from module1_ingestion.ingestor import parse_csv
    import argparse
    
    parser = argparse.ArgumentParser(description="Run anti-cheat detection")
    parser.add_argument("--input", default="data/sample_candidates.csv")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()
    
    candidates = parse_csv(args.input)
    result = run_anticheat(candidates, args.db)
    print(result.model_dump_json(indent=2))
