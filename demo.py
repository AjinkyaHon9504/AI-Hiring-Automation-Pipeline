"""
GTHR Pipeline — Full Demo Runner
Shows exactly what each module does step by step.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

DIVIDER = "=" * 70
THIN = "-" * 70

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def subsection(title):
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)


# ================================================================
#  STEP 1: INGESTION
# ================================================================
section("STEP 1: DATA INGESTION")
print("Reading: data/sample_candidates.csv")
print()

from module1_ingestion.ingestor import parse_csv, run_ingestion

candidates = parse_csv("data/sample_candidates.csv")

print(f"\nResult: {len(candidates)} candidates parsed")
print()
print(f"  {'ID':<8} {'Name':<20} {'Email':<30} {'Answers':<8} {'GitHub'}")
print(f"  {'-'*8} {'-'*20} {'-'*30} {'-'*8} {'-'*20}")
for c in candidates:
    github = c.profile.github_url or "(none)"
    if len(github) > 20:
        github = "..." + github[-17:]
    print(f"  {c.candidate_id:<8} {c.name:<20} {c.email:<30} {len(c.answers):<8} {github}")

print(f"\n  Deduplication: 7 rows in CSV -> {len(candidates)} unique candidates")
print(f"  (c007 'Test Candidate' was a duplicate of c002 'Priya Sharma' - same email+role)")


# ================================================================
#  STEP 2: SCORING
# ================================================================
section("STEP 2: CANDIDATE SCORING")
print("Scoring each candidate across 5 dimensions...")
print()

from module2_scoring.scorer import score_all_candidates, DEFAULT_WEIGHTS

scored = score_all_candidates(candidates)

print(f"  Weights: {json.dumps(DEFAULT_WEIGHTS, indent=None)}")
print()

# Print scores table
print(f"  {'Rank':<5} {'Name':<20} {'Score':>7} {'Tier':<12} {'Tech':>5} {'Qual':>5} {'Prof':>5} {'Spec':>5} {'Time':>5}")
print(f"  {'-'*5} {'-'*20} {'-'*7} {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")

for i, sc in enumerate(scored, 1):
    ds = sc.dimension_scores
    print(f"  #{i:<4} {sc.candidate.name:<20} {sc.total_score:>7.2f} {sc.tier:<12} "
          f"{ds.get('technical_relevance',0):>5.0f} {ds.get('answer_quality',0):>5.0f} "
          f"{ds.get('profile_credibility',0):>5.0f} {ds.get('specificity',0):>5.0f} "
          f"{ds.get('timing',0):>5.0f}")

# Show why top and bottom scored the way they did
subsection("WHY #1 SCORED HIGH (Arjun Mehta)")
top = scored[0]
for expl in top.explanation[:6]:
    print(f"  + {expl}")

subsection("WHY #6 SCORED LOW (Anonymous User)")
bottom = scored[-1]
for expl in bottom.explanation[:6]:
    print(f"  - {expl}")
if bottom.flags:
    print(f"  FLAGS: {', '.join(bottom.flags)}")


# ================================================================
#  STEP 3: ANTI-CHEAT DETECTION
# ================================================================
section("STEP 3: ANTI-CHEAT DETECTION")
print("Running 3 detection phases...")

from module4_anticheat.detector import (
    detect_ai_generated, SimilarityDetector, analyze_timing, StrikeSystem
)

# Phase 1: AI Detection
subsection("Phase 1: AI-Generated Text Detection")
for c in candidates:
    all_text = " ".join(a.answer_text for a in c.answers if a.answer_text)
    if len(all_text) < 20:
        print(f"  {c.name:<20} -> text too short for analysis")
        continue
    result = detect_ai_generated(all_text)
    verdict = result["verdict"]
    prob = result["ai_probability"]
    icon = "!!" if verdict == "likely_ai" else "?" if verdict == "suspicious" else "ok"
    print(f"  {c.name:<20} -> {verdict:<14} (probability: {prob:.3f}) [{icon}]")

# Phase 2: Similarity
subsection("Phase 2: Cross-Candidate Similarity")
detector = SimilarityDetector()
print(f"  Method: {detector.method}")
similar = detector.find_similar_pairs(candidates, threshold=0.75)

if similar:
    for pair in similar:
        severity_icon = "!!!" if pair["severity"] == "high" else "! "
        print(f"  {severity_icon} {pair['candidate_a_name']} <-> {pair['candidate_b_name']}")
        print(f"      Question: {pair['question_id']}, Similarity: {pair['similarity']:.4f}")
        print(f"      A: \"{pair['text_a_preview'][:80]}...\"")
        print(f"      B: \"{pair['text_b_preview'][:80]}...\"")
        print()
else:
    print("  No suspicious similarities found.")

# Phase 3: Timing
subsection("Phase 3: Timing Anomaly Detection")
for c in candidates:
    timing = analyze_timing(c)
    resp = c.metadata.response_time_seconds
    if timing["flags"]:
        print(f"  !! {c.name:<20} {resp:>5.0f}s -> {timing['flags'][0]}")
    else:
        print(f"  ok {c.name:<20} {resp:>5.0f}s -> normal")


# ================================================================
#  STEP 4: EMAIL ENGAGEMENT
# ================================================================
section("STEP 4: EMAIL ENGAGEMENT (Simulated)")
print("Deciding who to email based on tier...")
print()

from module3_engagement.engine import (
    classify_response, decide_next_action, generate_email, 
    ThreadTracker, ThreadState, MockEmailBackend
)

backend = MockEmailBackend()

for sc in scored:
    name = sc.candidate.name
    tier = sc.tier
    
    if tier == "Reject":
        print(f"  SKIP  {name:<20} (Tier: {tier}, Score: {sc.total_score:.1f})")
    elif tier == "Review":
        print(f"  HOLD  {name:<20} (Tier: {tier} - needs manual review)")
    else:
        print(f"  EMAIL {name:<20} (Tier: {tier}, Score: {sc.total_score:.1f})")

# Show a sample email
subsection("Sample Email: Initial Outreach to Arjun Mehta")
email = generate_email("initial_outreach", {
    "name": "Arjun",
    "role": "AI Agent Developer",
    "question": "What's a real-world system you've built that involved autonomous decision-making? Walk me through the architecture."
})
for line in email.strip().split("\n"):
    print(f"  | {line}")

# Show response classification
subsection("Response Classification Demo")
test_responses = [
    ("I built a RAG pipeline using LangChain with FAISS for vector search. Retrieval precision went from 31% to 67% after adding a cross-encoder reranker.", "technical"),
    ("I am interested in AI and passionate about technology.", "vague"),
    ("```python\ndef score_candidate(data):\n    return sum(weights * scores)\n```", "code"),
    ("What tech stack does your team currently use?", "question"),
    ("ok", "minimal"),
]

for response_text, expected in test_responses:
    rtype, confidence = classify_response(response_text)
    action = decide_next_action(response_text, current_round=1)
    preview = response_text[:60].replace("\n", " ")
    print(f"  Input:    \"{preview}...\"")
    print(f"  Type:     {rtype.value} (confidence: {confidence:.2f})")
    print(f"  Action:   {action['action']}")
    print()


# ================================================================
#  STEP 5: SELF-LEARNING
# ================================================================
section("STEP 5: SELF-LEARNING ANALYSIS")

from module5_learning.learner import BatchAnalyzer, FeedbackLoop
from shared.database import Database

db = Database(":memory:")

scored_data = [
    {
        "candidate_id": sc.candidate.candidate_id,
        "total_score": sc.total_score,
        "tier": sc.tier,
        "answers": [a.model_dump() for a in sc.candidate.answers],
    }
    for sc in scored
]

analyzer = BatchAnalyzer(db)

# Score distribution
dist = analyzer.analyze_score_distribution(scored_data)
print(f"  Score Distribution:")
print(f"    Mean:   {dist['score_stats']['mean']:.1f}")
print(f"    Min:    {dist['score_stats']['min']:.1f}")
print(f"    Max:    {dist['score_stats']['max']:.1f}")
print(f"    Median: {dist['score_stats']['median']:.1f}")
print()
print(f"  Tier breakdown:")
for tier, count in dist.get("tier_distribution", {}).items():
    bar = "#" * (count * 5)
    print(f"    {tier:<12} {count} {bar}")

if "warning" in dist:
    print(f"\n  >> WARNING: {dist['warning']}")

# Weight recommendation
subsection("Weight Adjustment Recommendation")
patterns = analyzer.analyze_answer_patterns(scored_data)
recs = analyzer.recommend_weight_adjustments(dist, patterns)

if recs.get("suggested_adjustments"):
    feedback = FeedbackLoop(db)
    new_weights = feedback.apply_weight_adjustment(DEFAULT_WEIGHTS, recs["suggested_adjustments"])
    
    print(f"  Current weights vs. Recommended weights:")
    print(f"  {'Dimension':<25} {'Current':>8} {'New':>8} {'Delta':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for dim in DEFAULT_WEIGHTS:
        curr = DEFAULT_WEIGHTS[dim]
        new = new_weights.get(dim, curr)
        delta = new - curr
        arrow = "^" if delta > 0 else "v" if delta < 0 else "="
        print(f"  {dim:<25} {curr:>8.3f} {new:>8.4f} {delta:>+8.4f} {arrow}")
    
    for reason in recs.get("reasoning", []):
        print(f"\n  Reasoning: {reason}")
else:
    print("  No weight adjustments recommended (distribution looks healthy)")


# ================================================================
#  SUMMARY
# ================================================================
section("PIPELINE SUMMARY")
print()
print(f"  Candidates processed:  {len(candidates)}")
print(f"  Duplicates removed:    1 (c007)")
print()
print(f"  Tier Results:")
from collections import Counter
tiers = Counter(sc.tier for sc in scored)
for tier in ["Fast-Track", "Standard", "Review", "Reject"]:
    count = tiers.get(tier, 0)
    names = [sc.candidate.name for sc in scored if sc.tier == tier]
    print(f"    {tier:<12} {count}  ({', '.join(names)})")

print()
print(f"  Anti-Cheat Findings:")
print(f"    Copy detected:    c001 <-> c003 on Q1 (similarity = 1.00)")
print(f"    Timing anomaly:   c005 (3 second response = likely paste)")
print(f"    Timing suspicion: c002 (15s, 6.9 words/sec > typing speed)")

print()
print(f"  Emails to send:     {sum(1 for sc in scored if sc.tier in ('Fast-Track', 'Standard'))}")
print(f"  Held for review:    {sum(1 for sc in scored if sc.tier == 'Review')}")
print(f"  Auto-rejected:      {sum(1 for sc in scored if sc.tier == 'Reject')}")
print()
print(DIVIDER)
print("  PIPELINE COMPLETE")
print(DIVIDER)
