# GTHR — AI Hiring Automation Pipeline

A modular, production-ready AI hiring automation system that ingests applicants, scores them across multiple dimensions, runs contextual multi-round email conversations, detects cheating (AI-generated/copied responses), and learns from interactions over time.

## Quick Start

```bash
# Install dependencies
pip install pandas openpyxl scikit-learn numpy pydantic pyyaml python-dateutil

# Run full pipeline
python main.py --input data/sample_candidates.csv --output results.json

# Run individual modules
python main.py --module scoring --input data/sample_candidates.csv
python main.py --module anticheat --input data/sample_candidates.csv
python main.py --module engagement --input data/sample_candidates.csv
python main.py --module learning --input data/sample_candidates.csv
```

## Architecture

```
[Ingestion] -> [Scoring] -> [Anti-Cheat] -> [Re-Score] -> [Engagement] -> [Learning]
                                                                              |
                                                    feedback loop <-----------+
```

## Modules

| Module | Description | Key Feature |
|--------|-------------|-------------|
| **Module 1: Ingestion** | CSV/Excel/Email/API ingestion | Deduplication, column auto-mapping, missing field handling |
| **Module 2: Scoring** | Multi-dimensional candidate scoring | 5 dimensions, weighted scoring, tier assignment |
| **Module 3: Engagement** | Email automation with thread tracking | State machine, response classification, contextual follow-ups |
| **Module 4: Anti-Cheat** | AI + copy + timing detection | Cosine similarity, sentence uniformity, strike system |
| **Module 5: Self-Learning** | Pattern analysis + weight tuning | Batch analysis, conservative weight adjustment, versioned snapshots |
| **Module 6: Integration** | Pipeline orchestrator | DAG execution, SQLite checkpoints, graceful shutdown |

## Project Structure

```
gthr/
  main.py                              # CLI entry point
  shared/
    schema.py                           # Pydantic models (Candidate, ScoredCandidate, etc.)
    database.py                         # SQLite persistence layer
    utils.py                            # Logging, retry, text analysis utilities
  module1_ingestion/ingestor.py         # CSV/Excel parsing, dedup, normalization
  module2_scoring/scorer.py             # Multi-dimensional scoring engine
  module3_engagement/engine.py          # Email automation + thread state machine
  module4_anticheat/detector.py         # AI detection, similarity, timing analysis
  module5_learning/learner.py           # Batch analysis, feedback loop
  module6_integration/orchestrator.py   # Pipeline coordinator with checkpoints
  data/sample_candidates.csv            # Test data (7 candidates)
```

## Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|-----------------|
| Technical Relevance | 30% | Domain-specific terms in context |
| Answer Quality | 25% | Depth, vocabulary richness, structure |
| Profile Credibility | 20% | GitHub/LinkedIn presence and validity |
| Specificity | 15% | Concrete examples vs generic phrases |
| Timing | 10% | Response time analysis |

## Tier Assignment

- **Fast-Track** (≥85): Strong candidates, advance immediately
- **Standard** (65-84): Good candidates, standard process
- **Review** (45-64): Needs manual review
- **Reject** (<45 or cheat flags): Auto-reject

## Anti-Cheat Detection

- **AI Text Detection**: Sentence length uniformity, vocabulary patterns, formality markers
- **Cross-Candidate Similarity**: TF-IDF / sentence-transformers cosine similarity
- **Timing Anomalies**: Suspiciously fast responses flagged
- **Strike System**: 3 strikes = automatic elimination

## Tech Stack

- **Python 3.11+** with Pydantic v2, pandas, scikit-learn
- **SQLite** for persistence (swappable to PostgreSQL)
- **sentence-transformers** (optional) for semantic similarity
- **Gmail API / SMTP** for email (mock backend for testing)

## License

MIT
