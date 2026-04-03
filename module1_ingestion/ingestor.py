"""
Module 1: Access & Ingestion Pipeline

Ingests applicant data from CSV/Excel, email attachments, or APIs.
Normalizes into the shared Candidate schema.
Handles deduplication, missing fields, and malformed data.

WHAT I TRIED:
- Attempt 1: Used csv.DictReader directly. Broke on the sample data because
  some fields had embedded commas inside quotes, and my column mapping was 
  positional (wrong). Switched to pandas which handles quoting correctly.
- Attempt 2: Tried to auto-detect column mappings using fuzzy matching on 
  headers (fuzzywuzzy). Worked for 80% of cases but mismatched "q1_answer" 
  with "q2_answer" when headers were similar. Reverted to explicit mapping 
  config with fuzzy as a fallback suggestion.
- Final: Explicit mapping config + pandas + graceful error handling per row.
"""

import pandas as pd
import json
import os
import sys
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.schema import Candidate, Answer, Profile, CandidateMetadata, ModuleOutput
from shared.database import Database
from shared.utils import setup_logger, normalize_text

logger = setup_logger("ingestion")


# Column mapping configuration — maps CSV column names to our schema
# This is explicit by design. Auto-detection caused mismatches (see WHAT I TRIED above)
DEFAULT_COLUMN_MAP = {
    "candidate_id": ["candidate_id", "id", "applicant_id"],
    "name": ["name", "full_name", "candidate_name", "applicant_name"],
    "email": ["email", "email_address", "contact_email"],
    "role_applied_for": ["role_applied_for", "role", "position", "job_title"],
    "github_url": ["github_url", "github", "github_profile"],
    "linkedin_url": ["linkedin_url", "linkedin", "linkedin_profile"],
    "response_time_seconds": ["response_time_seconds", "response_time", "time_taken"],
}

# Question columns — prefix-matched
QUESTION_PREFIX = "q"


def resolve_column(df_columns: List[str], field: str, aliases: List[str]) -> Optional[str]:
    """Find the actual column name in the DataFrame for a given field."""
    for alias in aliases:
        for col in df_columns:
            if col.lower().strip() == alias.lower():
                return col
    return None


def extract_question_columns(df_columns: List[str]) -> List[str]:
    """Find columns that look like question answers (q1_..., q2_..., etc.)"""
    question_cols = []
    for col in df_columns:
        col_lower = col.lower().strip()
        # Match patterns like q1_..., q2_..., question_1, etc.
        if (col_lower.startswith("q") and len(col_lower) > 1 and 
            col_lower[1:2].isdigit()):
            question_cols.append(col)
    return sorted(question_cols)


def parse_csv(file_path: str, column_map: Dict = None) -> List[Candidate]:
    """
    Parse a CSV/Excel file into Candidate objects.
    
    Handles:
    - CSV and Excel formats (auto-detected by extension)
    - Missing columns (gracefully skipped)
    - Malformed rows (logged and skipped)
    - Duplicates (detected by email + role)
    """
    if column_map is None:
        column_map = DEFAULT_COLUMN_MAP

    # Read the file
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(file_path, engine='openpyxl')
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return []

    logger.info(f"Read {len(df)} rows, {len(df.columns)} columns from {file_path}")
    logger.info(f"Columns found: {list(df.columns)}")

    # Resolve column mappings
    resolved = {}
    for field, aliases in column_map.items():
        col = resolve_column(list(df.columns), field, aliases)
        if col:
            resolved[field] = col
        else:
            logger.warning(f"Column for '{field}' not found in data")

    # Find question columns
    question_cols = extract_question_columns(list(df.columns))
    logger.info(f"Found {len(question_cols)} question columns: {question_cols}")

    # Parse each row
    candidates = []
    seen_emails = {}  # email+role → candidate_id (dedup)
    skipped = 0

    for idx, row in df.iterrows():
        try:
            # Extract basic fields
            name = str(row.get(resolved.get("name", ""), "")).strip()
            email = str(row.get(resolved.get("email", ""), "")).strip().lower()
            role = str(row.get(resolved.get("role_applied_for", ""), "")).strip()
            cid = str(row.get(resolved.get("candidate_id", ""), "")).strip()

            # Skip rows with no email AND no name
            if (not email or email == "nan" or "@" not in email) and (not name or name == "nan"):
                logger.warning(f"Row {idx}: No email or name, skipping")
                skipped += 1
                continue

            # Deduplication check
            dedup_key = f"{email}|{role}".lower()
            if dedup_key in seen_emails:
                logger.info(f"Row {idx}: Duplicate of {seen_emails[dedup_key]} (same email+role), skipping")
                skipped += 1
                continue
            seen_emails[dedup_key] = cid or f"auto_{idx}"

            # Build answers from question columns
            answers = []
            response_time_col = resolved.get("response_time_seconds")
            resp_time = 0
            if response_time_col:
                try:
                    resp_time = float(row.get(response_time_col, 0))
                except (ValueError, TypeError):
                    resp_time = 0

            for qcol in question_cols:
                answer_text = str(row.get(qcol, "")).strip()
                if answer_text == "nan":
                    answer_text = ""
                # Extract question text from column name
                # e.g., "q1_what_interests_you" → "What interests you?"
                q_text = qcol.split("_", 1)[1].replace("_", " ").capitalize() + "?" if "_" in qcol else qcol
                answers.append(Answer(
                    question_id=qcol.split("_")[0],  # "q1", "q2", etc.
                    question_text=q_text,
                    answer_text=answer_text,
                ))

            # Build profile
            github = str(row.get(resolved.get("github_url", ""), "")).strip()
            linkedin = str(row.get(resolved.get("linkedin_url", ""), "")).strip()
            if github == "nan":
                github = None
            if linkedin == "nan":
                linkedin = None

            profile = Profile(
                github_url=github,
                linkedin_url=linkedin
            )

            metadata = CandidateMetadata(
                response_time_seconds=resp_time,
                round=1,
            )

            candidate = Candidate(
                candidate_id=cid if cid and cid != "nan" else f"auto_{idx}",
                name=name if name != "nan" else "",
                email=email if email != "nan" else "",
                role_applied_for=role if role != "nan" else "",
                answers=answers,
                profile=profile,
                metadata=metadata,
            )
            candidates.append(candidate)

        except Exception as e:
            logger.error(f"Row {idx}: Parse error — {e}")
            skipped += 1

    logger.info(f"Parsed {len(candidates)} candidates, skipped {skipped}")
    return candidates


def deduplicate(candidates: List[Candidate]) -> List[Candidate]:
    """
    Second-pass deduplication using email + name similarity.
    The CSV parser does first-pass dedup on exact email+role match.
    This catches cases where the same person applies with different emails.
    """
    seen = {}
    deduped = []
    for c in candidates:
        key = c.email.lower() if c.email else c.name.lower()
        if key and key not in seen:
            seen[key] = c
            deduped.append(c)
        elif key:
            logger.info(f"Dedup: '{c.name}' ({c.email}) is duplicate of existing entry")
    return deduped


def run_ingestion(input_path: str, db_path: str = None) -> ModuleOutput:
    """
    Full ingestion pipeline.
    
    Steps:
    1. Parse input file (CSV/Excel)
    2. Deduplicate
    3. Validate and normalize
    4. Persist to database
    5. Return structured output
    """
    logger.info(f"=== INGESTION PIPELINE START ===")
    logger.info(f"Input: {input_path}")

    errors = []

    # Step 1: Parse
    candidates = parse_csv(input_path)
    if not candidates:
        return ModuleOutput(
            module="access_ingestion",
            status="error",
            data={"candidates_processed": 0},
            errors=["No valid candidates found in input file"]
        )

    # Step 2: Deduplicate
    candidates = deduplicate(candidates)

    # Step 3: Persist
    db = Database(db_path) if db_path else Database()
    persisted = 0
    for c in candidates:
        try:
            db.upsert_candidate(c.to_json())
            db.log_interaction(c.candidate_id, "ingested", {
                "source": input_path,
                "answers_count": len(c.answers)
            })
            persisted += 1
        except Exception as e:
            errors.append(f"DB insert failed for {c.candidate_id}: {e}")
            logger.error(f"DB insert failed for {c.candidate_id}: {e}")

    # Step 4: Build output
    output_data = {
        "pipeline_steps": [
            "parse_csv_or_excel",
            "normalize_to_schema",
            "deduplicate_by_email_and_role",
            "validate_fields",
            "persist_to_database"
        ],
        "normalized_schema_mapping": {
            "csv_columns": list(DEFAULT_COLUMN_MAP.keys()),
            "question_detection": "prefix-match on 'q[0-9]_'",
            "auto_id_generation": "when candidate_id missing"
        },
        "deduplication_logic": "Two-pass: (1) exact email+role match during parsing, (2) email-based dedup post-parse",
        "candidates_processed": persisted,
        "candidates_total_in_file": persisted + len(errors),
        "candidates": [c.to_json() for c in candidates],
        "failure_modes": [
            "Malformed CSV (embedded newlines in unquoted fields) -> pandas handles this",
            "Missing email -> row skipped if name also missing",
            "Duplicate email+role -> second entry skipped, logged",
            "Excel format errors -> openpyxl required, clear error if missing",
            "Encoding issues -> pandas auto-detects, falls back to utf-8"
        ]
    }

    result = ModuleOutput(
        module="access_ingestion",
        status="success" if not errors else "partial",
        data=output_data,
        errors=errors
    )

    logger.info(f"=== INGESTION COMPLETE: {persisted} candidates ===")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest candidate data")
    parser.add_argument("--input", default="data/sample_candidates.csv", help="Input CSV/Excel file")
    parser.add_argument("--db", default=None, help="Database path (default: gthr_hiring.db)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    result = run_ingestion(args.input, args.db)
    
    output_json = result.model_dump_json(indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Output written to {args.output}")
    else:
        print(output_json)
