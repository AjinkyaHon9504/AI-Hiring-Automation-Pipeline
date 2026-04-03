"""
GTHR — AI Hiring Automation Pipeline
Main entry point.

Usage:
    python main.py --input data/sample_candidates.csv              # Full pipeline
    python main.py --input data/sample_candidates.csv --module scoring  # Single module
    python main.py --input data/sample_candidates.csv --output results.json  # Save to file
"""

import argparse
import json
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.utils import setup_logger

logger = setup_logger("main")


def main():
    parser = argparse.ArgumentParser(
        description="GTHR — AI Hiring Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/sample_candidates.csv
  python main.py --input data/sample_candidates.csv --module scoring
  python main.py --input data/sample_candidates.csv --module anticheat
  python main.py --input data/sample_candidates.csv --output results.json
        """
    )
    parser.add_argument("--input", default="data/sample_candidates.csv", help="Input CSV/Excel file")
    parser.add_argument("--module", default="all", 
                        choices=["all", "ingestion", "scoring", "engagement", "anticheat", "learning"],
                        help="Run a specific module or all")
    parser.add_argument("--db", default=None, help="Database path (default: gthr_hiring.db)")
    parser.add_argument("--output", default=None, help="Output JSON file (optional)")
    parser.add_argument("--mode", default="run", choices=["run", "test"], help="Run mode")
    
    args = parser.parse_args()
    
    # Resolve input path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(script_dir, input_path)
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"GTHR Pipeline — Input: {input_path}, Module: {args.module}")
    
    result = None
    
    if args.module == "all":
        from module6_integration.orchestrator import run_integration
        result = run_integration(input_path, args.db)
    
    elif args.module == "ingestion":
        from module1_ingestion.ingestor import run_ingestion
        result = run_ingestion(input_path, args.db)
    
    elif args.module == "scoring":
        from module1_ingestion.ingestor import parse_csv
        from module2_scoring.scorer import run_scoring
        candidates = parse_csv(input_path)
        result = run_scoring(candidates, args.db)
    
    elif args.module == "anticheat":
        from module1_ingestion.ingestor import parse_csv
        from module4_anticheat.detector import run_anticheat
        candidates = parse_csv(input_path)
        result = run_anticheat(candidates, args.db)
    
    elif args.module == "engagement":
        from module1_ingestion.ingestor import parse_csv
        from module2_scoring.scorer import score_all_candidates
        from module3_engagement.engine import run_engagement
        candidates = parse_csv(input_path)
        scored = score_all_candidates(candidates)
        result = run_engagement(scored, args.db)
    
    elif args.module == "learning":
        from module1_ingestion.ingestor import parse_csv
        from module2_scoring.scorer import score_all_candidates
        from module5_learning.learner import run_learning
        candidates = parse_csv(input_path)
        scored = score_all_candidates(candidates)
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
    
    # Output
    if result:
        output_json = result.model_dump_json(indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_json)
            logger.info(f"Results written to {args.output}")
        else:
            print(output_json)
    
    return result


if __name__ == "__main__":
    main()
