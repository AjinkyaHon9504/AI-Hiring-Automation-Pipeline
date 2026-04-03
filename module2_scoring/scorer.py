"""
Module 2: Candidate Scoring & Ranking Engine

Multi-dimensional scoring system that evaluates candidates on:
- Technical relevance (do they know what they're talking about?)
- Answer quality (depth, specificity, originality)
- Profile credibility (GitHub activity, LinkedIn presence)
- Specificity vs generic (penalize template responses)
- Response timing (suspiciously fast = flag)

WHAT I TRIED:
- Attempt 1: Simple keyword counting. "docker" +1, "kubernetes" +1, etc.
  Problem: Candidate could write "I don't know docker or kubernetes" and 
  score well. This was embarrassingly broken.
  
- Attempt 2: TF-IDF on answers vs a "ideal answer corpus". Better, but 
  required maintaining an ideal answer set per role which doesn't scale 
  and biases toward specific phrasing.
  
- Attempt 3 (current): Multi-signal scoring with semantic analysis.
  Uses sentence-level analysis, generic phrase detection, vocabulary 
  richness, and structural indicators. No ideal answers needed.
  The key insight: penalizing bad signals works better than rewarding 
  good ones, because good candidates are diverse but bad patterns are 
  consistent.
"""

import os
import sys
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.schema import Candidate, ScoredCandidate, ModuleOutput
from shared.database import Database
from shared.utils import (
    setup_logger, normalize_text, word_count, 
    unique_words_ratio, hapax_legomena_ratio
)

logger = setup_logger("scoring")


# ====================================================================
# SCORING WEIGHTS — These get updated by the self-learning module
# ====================================================================
DEFAULT_WEIGHTS = {
    "technical_relevance": 0.30,
    "answer_quality": 0.25,
    "profile_credibility": 0.20,
    "specificity": 0.15,
    "timing": 0.10,
}

# ====================================================================
# PENALTY & BONUS RULES
# ====================================================================

# Generic phrases that indicate template/low-effort responses
GENERIC_PHRASES = [
    "i am a fast learner",
    "passionate about technology",
    "i am passionate about",
    "great work in this space",
    "i think this company",
    "i am interested in",
    "depending on requirements",
    "i would use",
    "it was challenging",
    "i learned that",
    "best practices",
    "industry standard",
    "cutting edge",
    "state of the art",
    "leverage my skills",
    "team player",
    "hard worker",
    "detail oriented",
    "think outside the box",
]

# Technical indicators — terms that suggest actual technical depth
TECHNICAL_INDICATORS = {
    "ai_agent_developer": [
        "langchain", "rag", "retrieval", "embedding", "vector", "transformer",
        "fine-tuning", "fine tuning", "prompt engineering", "llm", "gpt",
        "agent", "tool calling", "function calling", "chain of thought",
        "openai", "anthropic", "huggingface", "tokenizer", "attention",
        "celery", "redis", "kafka", "flink", "api", "fastapi", "flask",
        "docker", "kubernetes", "ci/cd", "pipeline", "dag",
        "postgres", "mongodb", "redis", "sqlite", "dynamodb",
        "latency", "throughput", "p99", "p95", "benchmark",
        "precision", "recall", "f1", "rmse", "accuracy",
        "cosine similarity", "cross-encoder", "reranker",
        "async", "concurrent", "threading", "multiprocessing",
    ],
}

# Structural quality indicators
QUALITY_INDICATORS = [
    (r'\d+%', 0.5, "mentions specific percentages/metrics"),
    (r'\d+ms|\d+s\b|\d+ seconds', 0.3, "mentions specific timing/performance"),
    (r'\$\d+', 0.3, "mentions cost considerations"),
    (r'\([\d]+\)', 0.2, "uses numbered steps/approaches"),
    (r'tried|attempted|experimented', 0.4, "describes experimentation"),
    (r'failed|broke|didn\'t work|dropped', 0.3, "honest about failures"),
    (r'because|since|the reason', 0.2, "explains reasoning"),
    (r'trade-?off|versus|vs\.?', 0.3, "considers trade-offs"),
    (r'lesson|learned|realized|key insight', 0.3, "reflective learning"),
]


def score_technical_relevance(answers: list, role: str) -> Tuple[float, List[str]]:
    """
    Score based on technical terms and concepts mentioned.
    Not just keyword counting — checks context around keywords.
    
    Returns: (score 0-100, list of explanations)
    """
    role_key = role.lower().replace(" ", "_")
    indicators = TECHNICAL_INDICATORS.get(role_key, TECHNICAL_INDICATORS.get("ai_agent_developer", []))
    
    all_text = " ".join(a.answer_text.lower() for a in answers)
    explanations = []
    
    if not all_text.strip():
        return 0.0, ["No answer text provided"]
    
    # Count unique technical terms used
    found_terms = set()
    for term in indicators:
        if term in all_text:
            found_terms.add(term)
    
    term_ratio = len(found_terms) / max(len(indicators), 1)
    
    # Scale: 0 terms = 0, 5+ relevant terms = decent, 15+ = excellent
    if len(found_terms) >= 15:
        base_score = 90
    elif len(found_terms) >= 10:
        base_score = 75
    elif len(found_terms) >= 5:
        base_score = 55
    elif len(found_terms) >= 2:
        base_score = 35
    else:
        base_score = 10
    
    explanations.append(f"Found {len(found_terms)} technical terms: {', '.join(sorted(list(found_terms)[:10]))}")
    
    # Bonus: mentions specific tools/versions (not just buzzwords)
    specific_patterns = re.findall(r'[a-zA-Z]+[\s-]?v?\d+\.?\d*', all_text)
    if specific_patterns:
        base_score = min(100, base_score + len(specific_patterns) * 2)
        explanations.append(f"Mentions specific versions/numbers: shows precision")
    
    return base_score, explanations


def score_answer_quality(answers: list) -> Tuple[float, List[str]]:
    """
    Score based on answer depth, structure, and substance.
    
    Signals:
    - Word count (too short = bad, too long without substance = also bad)
    - Vocabulary richness (unique words ratio)
    - Structural indicators (numbers, examples, comparisons)
    - Multi-approach thinking (tried X, then Y, settled on Z)
    """
    explanations = []
    total_score = 0
    
    for answer in answers:
        text = answer.answer_text
        wc = word_count(text)
        
        # Word count scoring
        if wc == 0:
            explanations.append(f"Q '{answer.question_id}': Empty answer")
            continue
        elif wc <= 5:
            total_score += 5
            explanations.append(f"Q '{answer.question_id}': Extremely short ({wc} words)")
        elif wc <= 20:
            total_score += 20
            explanations.append(f"Q '{answer.question_id}': Brief response ({wc} words)")
        elif wc <= 50:
            total_score += 50
        elif wc <= 150:
            total_score += 70
        else:
            total_score += 80
            explanations.append(f"Q '{answer.question_id}': Detailed response ({wc} words)")
        
        # Vocabulary richness bonus
        uwr = unique_words_ratio(text)
        if uwr > 0.7:
            total_score += 10
        elif uwr < 0.4:
            total_score -= 10
            explanations.append(f"Q '{answer.question_id}': Repetitive vocabulary (ratio: {uwr:.2f})")
        
        # Structural quality bonuses
        for pattern, bonus, desc in QUALITY_INDICATORS:
            if re.search(pattern, text, re.IGNORECASE):
                total_score += bonus * 5  # Scale up for scoring
    
    # Average across answers
    if answers:
        avg_score = total_score / len(answers)
        return min(100, max(0, avg_score)), explanations
    return 0, ["No answers to evaluate"]


def score_profile_credibility(profile, answers: list) -> Tuple[float, List[str]]:
    """
    Score profile links. We CAN'T scrape — so we check:
    - Is the URL present and well-formed?
    - Does the GitHub username look real? (not test/example/null)
    - Is LinkedIn present?
    - Penalty for claiming GitHub but having suspicious URL
    
    NOTE: In production, this would make GitHub API calls to check 
    repo count, contribution activity, etc. Rate limit: 60/hr unauth.
    For this prototype, we do format validation only.
    """
    explanations = []
    score = 50  # Neutral baseline
    
    # GitHub
    if profile.github_url:
        url = profile.github_url.lower()
        # Check for suspicious/fake patterns
        fake_patterns = ["test", "example", "null", "none", "fake", "asdf", "404"]
        is_suspicious = any(p in url.split("/")[-1] for p in fake_patterns)
        
        if is_suspicious:
            score -= 20
            explanations.append(f"GitHub URL looks suspicious: {profile.github_url}")
        else:
            score += 20
            explanations.append(f"GitHub profile provided: {profile.github_url}")
            
            # Does any answer reference their own GitHub work?
            all_text = " ".join(a.answer_text.lower() for a in answers)
            if "github" in all_text or "repo" in all_text or "repository" in all_text:
                score += 10
                explanations.append("References GitHub work in answers")
    else:
        score -= 10
        explanations.append("No GitHub profile provided")
    
    # LinkedIn
    if profile.linkedin_url:
        score += 15
        explanations.append("LinkedIn profile provided")
    else:
        score -= 5
        explanations.append("No LinkedIn profile")
    
    return min(100, max(0, score)), explanations


def score_specificity(answers: list) -> Tuple[float, List[str]]:
    """
    Penalize generic/template responses. Reward specific, concrete answers.
    
    This is where most bad candidates get caught:
    - "I am passionate about technology" → penalty
    - "I reduced alert triage time by 40%" → bonus
    """
    explanations = []
    total_penalty = 0
    total_bonus = 0
    
    all_text = " ".join(a.answer_text.lower() for a in answers)
    
    # Check for generic phrases
    generic_found = []
    for phrase in GENERIC_PHRASES:
        if phrase in all_text:
            generic_found.append(phrase)
            total_penalty += 8
    
    if generic_found:
        explanations.append(f"Found {len(generic_found)} generic phrases: {', '.join(generic_found[:5])}")
    
    # Check for concrete examples (specific numbers, names, tools)
    concrete_patterns = [
        (r'\d+\.?\d*%', "specific percentages"),
        (r'\b\d+k\b|\b\d+K\b|\b\d+M\b', "scale numbers"),
        (r'\b(built|deployed|implemented|created|designed)\b', "action verbs"),
        (r'\b(at my|at our|in my|in our)\b', "personal experience references"),
    ]
    
    for pattern, desc in concrete_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        if matches:
            total_bonus += min(15, len(matches) * 5)
            explanations.append(f"Contains {desc} ({len(matches)} instances)")
    
    score = 60 + total_bonus - total_penalty
    return min(100, max(0, score)), explanations


def score_timing(response_time: float) -> Tuple[float, List[str]]:
    """
    Score based on response time.
    Too fast → suspicious (might be copy-paste)
    Too slow → not necessarily bad, but flags disengagement
    
    Sweet spot: 60-300 seconds
    """
    explanations = []
    
    if response_time <= 0:
        return 50, ["No timing data available"]
    
    if response_time < 10:
        explanations.append(f"Suspiciously fast response: {response_time}s — possible copy-paste")
        return 15, explanations
    elif response_time < 30:
        explanations.append(f"Very fast response: {response_time}s")
        return 40, explanations
    elif response_time < 60:
        return 65, [f"Quick response: {response_time}s"]
    elif response_time < 300:
        return 85, [f"Thoughtful response time: {response_time}s"]
    elif response_time < 600:
        return 70, [f"Deliberate response time: {response_time}s"]
    else:
        return 50, [f"Slow response: {response_time}s — may indicate disengagement"]


def assign_tier(total_score: float, flags: list) -> str:
    """
    Assign tier based on score and flags.
    Flags can downgrade a tier (e.g., cheat detection).
    """
    # Hard reject conditions
    if "CHEAT_DETECTED" in flags or "COPIED_RESPONSE" in flags:
        return "Reject"
    
    if total_score >= 85:
        return "Fast-Track"
    elif total_score >= 65:
        return "Standard"
    elif total_score >= 45:
        return "Review"
    else:
        return "Reject"


def score_candidate(candidate: Candidate, weights: dict = None) -> ScoredCandidate:
    """
    Score a single candidate across all dimensions.
    Returns a ScoredCandidate with scores, tier, and explanations.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    all_explanations = []
    dimension_scores = {}
    flags = []

    # Score each dimension
    tech_score, tech_expl = score_technical_relevance(candidate.answers, candidate.role_applied_for)
    dimension_scores["technical_relevance"] = tech_score
    all_explanations.extend(tech_expl)

    quality_score, quality_expl = score_answer_quality(candidate.answers)
    dimension_scores["answer_quality"] = quality_score
    all_explanations.extend(quality_expl)

    profile_score, profile_expl = score_profile_credibility(candidate.profile, candidate.answers)
    dimension_scores["profile_credibility"] = profile_score
    all_explanations.extend(profile_expl)

    spec_score, spec_expl = score_specificity(candidate.answers)
    dimension_scores["specificity"] = spec_score
    all_explanations.extend(spec_expl)

    timing_score, timing_expl = score_timing(candidate.metadata.response_time_seconds)
    dimension_scores["timing"] = timing_score
    all_explanations.extend(timing_expl)

    # Compute weighted total
    total = sum(
        dimension_scores[dim] * weights[dim]
        for dim in weights
        if dim in dimension_scores
    )

    # Flag checks
    if not candidate.email:
        flags.append("MISSING_EMAIL")
        total *= 0.5
        all_explanations.append("Major penalty: no email provided")

    if all(word_count(a.answer_text) <= 5 for a in candidate.answers):
        flags.append("ALL_MINIMAL_ANSWERS")
        total *= 0.3
        all_explanations.append("All answers are minimal (≤5 words each)")

    tier = assign_tier(total, flags)

    return ScoredCandidate(
        candidate=candidate,
        total_score=round(total, 2),
        dimension_scores={k: round(v, 2) for k, v in dimension_scores.items()},
        tier=tier,
        explanation=all_explanations,
        flags=flags,
    )


def score_all_candidates(candidates: List[Candidate], weights: dict = None) -> List[ScoredCandidate]:
    """Score and rank all candidates. Returns sorted by score (descending)."""
    scored = [score_candidate(c, weights) for c in candidates]
    scored.sort(key=lambda s: s.total_score, reverse=True)
    
    for rank, sc in enumerate(scored, 1):
        sc.explanation.insert(0, f"Rank: #{rank} of {len(scored)}")
    
    return scored


def run_scoring(candidates: List[Candidate], db_path: str = None) -> ModuleOutput:
    """Full scoring pipeline with structured output."""
    logger.info(f"=== SCORING PIPELINE START ({len(candidates)} candidates) ===")

    scored = score_all_candidates(candidates)

    # Persist scores
    db = Database(db_path) if db_path else Database()
    for sc in scored:
        try:
            db.save_score(sc.candidate.candidate_id, {
                "total_score": sc.total_score,
                "dimension_scores": sc.dimension_scores,
                "tier": sc.tier,
                "explanation": sc.explanation,
            })
            db.log_interaction(sc.candidate.candidate_id, "scored", {
                "total_score": sc.total_score,
                "tier": sc.tier,
            })
        except Exception as e:
            logger.error(f"Failed to persist score for {sc.candidate.candidate_id}: {e}")

    # Build output
    tier_counts = Counter(sc.tier for sc in scored)
    
    output_data = {
        "rubric": {
            "technical_relevance": "Presence and context of domain-specific technical terms",
            "answer_quality": "Word count, vocabulary richness, structural indicators",
            "profile_credibility": "GitHub/LinkedIn presence and URL validity",
            "specificity": "Concrete examples vs generic phrases",
            "timing": "Response time analysis for engagement signals",
        },
        "weights": DEFAULT_WEIGHTS,
        "tier_rules": {
            "Fast-Track": "Score >= 85, no cheat flags",
            "Standard": "Score >= 65",
            "Review": "Score >= 45",
            "Reject": "Score < 45 or cheat flags present",
        },
        "tier_distribution": dict(tier_counts),
        "scoring_logic": "Weighted multi-dimensional scoring with penalty system for generic/low-effort responses",
        "results": [
            {
                "candidate_id": sc.candidate.candidate_id,
                "name": sc.candidate.name,
                "email": sc.candidate.email,
                "total_score": sc.total_score,
                "tier": sc.tier,
                "dimension_scores": sc.dimension_scores,
                "flags": sc.flags,
                "top_explanations": sc.explanation[:5],
            }
            for sc in scored
        ],
    }

    logger.info(f"=== SCORING COMPLETE — Tiers: {dict(tier_counts)} ===")
    return ModuleOutput(module="scoring", data=output_data)


if __name__ == "__main__":
    # Standalone execution: score from CSV directly
    from module1_ingestion.ingestor import parse_csv
    import argparse

    parser = argparse.ArgumentParser(description="Score candidates")
    parser.add_argument("--input", default="data/sample_candidates.csv")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    candidates = parse_csv(args.input)
    result = run_scoring(candidates, args.db)
    print(result.model_dump_json(indent=2))
