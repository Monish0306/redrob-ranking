"""
ingest.py  —  Phase 1
---------------------
Streaming JSONL reader + schema validator.
Reads candidates.jsonl line by line (never loads all 100K into RAM at once).

Key design decisions:
  - Generator pattern → memory-safe for 465MB file
  - Validates required fields + types on every record
  - Returns clean dicts that every downstream module can trust
  - Logs malformed records instead of crashing the whole pipeline
"""

import json
import logging
from pathlib import Path
from typing import Generator, Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# ─── Required top-level fields (from candidate_schema.json) ──────────────────
REQUIRED_TOP_LEVEL = {"candidate_id", "profile", "career_history",
                      "education", "skills", "redrob_signals"}

REQUIRED_PROFILE = {
    "anonymized_name", "headline", "summary", "location", "country",
    "years_of_experience", "current_title", "current_company",
    "current_company_size", "current_industry",
}

REQUIRED_SIGNALS = {
    "profile_completeness_score", "signup_date", "last_active_date",
    "open_to_work_flag", "recruiter_response_rate", "avg_response_time_hours",
    "notice_period_days", "preferred_work_mode", "willing_to_relocate",
    "github_activity_score", "interview_completion_rate",
    "offer_acceptance_rate", "verified_email", "verified_phone",
    "linkedin_connected", "applications_submitted_30d",
    "profile_views_received_30d", "saved_by_recruiters_30d",
    "search_appearance_30d", "connection_count", "endorsements_received",
    "skill_assessment_scores", "expected_salary_range_inr_lpa",
}

REQUIRED_CAREER_ENTRY = {
    "company", "title", "start_date", "duration_months",
    "is_current", "industry", "company_size", "description",
}

VALID_PROFICIENCY  = {"beginner", "intermediate", "advanced", "expert"}
VALID_COMPANY_SIZE = {"1-10","11-50","51-200","201-500","501-1000",
                      "1001-5000","5001-10000","10001+"}
VALID_WORK_MODE    = {"remote", "hybrid", "onsite", "flexible"}

# ─── Candidate ID pattern ─────────────────────────────────────────────────────
import re
CANDIDATE_ID_RE = re.compile(r"^CAND_\d{7}$")


def validate_candidate(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single candidate record.
    Returns (is_valid: bool, errors: List[str]).
    Soft errors (missing optional fields) are logged, not rejected.
    """
    errors = []

    # ── Top-level keys ────────────────────────────────────────────────────────
    for key in REQUIRED_TOP_LEVEL:
        if key not in record:
            errors.append(f"missing top-level key: {key}")

    if errors:
        return False, errors  # Can't validate sub-fields if top-level is broken

    # ── candidate_id format ───────────────────────────────────────────────────
    cid = record.get("candidate_id", "")
    if not CANDIDATE_ID_RE.match(cid):
        errors.append(f"invalid candidate_id format: {cid!r}")

    # ── profile fields ────────────────────────────────────────────────────────
    profile = record.get("profile", {})
    for key in REQUIRED_PROFILE:
        if key not in profile:
            errors.append(f"profile missing: {key}")

    yoe = profile.get("years_of_experience", -1)
    if not (0 <= yoe <= 50):
        errors.append(f"years_of_experience out of range: {yoe}")

    # ── career_history ────────────────────────────────────────────────────────
    career = record.get("career_history", [])
    if not isinstance(career, list) or len(career) == 0:
        errors.append("career_history must be a non-empty list")
    else:
        for i, entry in enumerate(career):
            for key in REQUIRED_CAREER_ENTRY:
                if key not in entry:
                    errors.append(f"career_history[{i}] missing: {key}")
            if entry.get("duration_months", -1) < 0:
                errors.append(f"career_history[{i}] negative duration_months")
            cs = entry.get("company_size", "")
            if cs and cs not in VALID_COMPANY_SIZE:
                errors.append(f"career_history[{i}] invalid company_size: {cs}")

    # ── skills ────────────────────────────────────────────────────────────────
    skills = record.get("skills", [])
    if isinstance(skills, list):
        for i, s in enumerate(skills):
            if s.get("proficiency") not in VALID_PROFICIENCY:
                errors.append(f"skills[{i}] invalid proficiency: {s.get('proficiency')}")
            if s.get("endorsements", 0) < 0:
                errors.append(f"skills[{i}] negative endorsements")

    # ── redrob_signals ────────────────────────────────────────────────────────
    signals = record.get("redrob_signals", {})
    for key in REQUIRED_SIGNALS:
        if key not in signals:
            errors.append(f"redrob_signals missing: {key}")

    rr = signals.get("recruiter_response_rate", -1)
    if not (-0.01 <= rr <= 1.01):
        errors.append(f"recruiter_response_rate out of range [0,1]: {rr}")

    gh = signals.get("github_activity_score", 0)
    if not (-1 <= gh <= 100):
        errors.append(f"github_activity_score out of range [-1,100]: {gh}")

    wm = signals.get("preferred_work_mode", "")
    if wm and wm not in VALID_WORK_MODE:
        errors.append(f"invalid preferred_work_mode: {wm}")

    return (len(errors) == 0), errors


def stream_candidates(
    filepath: Path,
    validate: bool = True,
    max_records: Optional[int] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream candidates one at a time from a JSONL file.
    Memory usage stays flat regardless of file size.

    Args:
        filepath:    Path to candidates.jsonl
        validate:    If True, runs schema validation on each record.
                     Malformed records are logged and skipped.
        max_records: Stop after this many valid records (for dev/testing).

    Yields:
        Clean candidate dicts.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Candidates file not found: {filepath}")

    valid_count   = 0
    skipped_count = 0

    with open(filepath, "r", encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            # ── Parse JSON ────────────────────────────────────────────────────
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON parse error — {e}")
                skipped_count += 1
                continue

            # ── Schema validation ─────────────────────────────────────────────
            if validate:
                is_valid, errs = validate_candidate(record)
                if not is_valid:
                    cid = record.get("candidate_id", f"line_{line_num}")
                    logger.warning(f"{cid}: schema errors — {errs}")
                    skipped_count += 1
                    continue

            yield record
            valid_count += 1

            if max_records and valid_count >= max_records:
                logger.info(f"Reached max_records={max_records}, stopping early.")
                break

    logger.info(
        f"Ingestion complete: {valid_count} valid, {skipped_count} skipped "
        f"(from {line_num} total lines)"
    )


def load_all_candidates(
    filepath: Path,
    validate: bool = True,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load all candidates into a list.
    Only use this when you need random access (e.g., scoring step).
    For exploration, prefer stream_candidates to keep memory flat.

    Returns:
        List of clean candidate dicts.
    """
    return list(stream_candidates(filepath, validate=validate,
                                  max_records=max_records))


# ─── Quick stats helper ───────────────────────────────────────────────────────

def dataset_stats(filepath: Path, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Compute quick stats on the dataset without loading everything.
    Useful for sanity-checking the file before a full run.
    """
    from collections import Counter
    titles   = Counter()
    yoe_vals = []
    countries = Counter()
    skills_all = Counter()

    for i, c in enumerate(stream_candidates(filepath, validate=False)):
        titles[c["profile"]["current_title"]] += 1
        yoe_vals.append(c["profile"]["years_of_experience"])
        countries[c["profile"]["country"]] += 1
        for s in c.get("skills", []):
            skills_all[s["name"]] += 1
        if i + 1 >= sample_size:
            break

    return {
        "sample_size":    sample_size,
        "top_10_titles":  titles.most_common(10),
        "top_10_skills":  skills_all.most_common(10),
        "top_5_countries": countries.most_common(5),
        "avg_yoe":        round(sum(yoe_vals) / len(yoe_vals), 2) if yoe_vals else 0,
        "min_yoe":        min(yoe_vals) if yoe_vals else 0,
        "max_yoe":        max(yoe_vals) if yoe_vals else 0,
    }


if __name__ == "__main__":
    # Quick smoke test — run: python src/ingest.py
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")

    data_path = Path(__file__).parent.parent / "data" / "candidates.jsonl"
    if not data_path.exists():
        print(f"❌  File not found: {data_path}")
        print("    Symlink or copy candidates.jsonl into data/ first.")
        sys.exit(1)

    print("Running ingestion smoke test on first 200 records...")
    count = 0
    for c in stream_candidates(data_path, validate=True, max_records=200):
        count += 1

    print(f"✅  Ingested {count} valid records without errors.")

    print("\nDataset quick stats (first 500 records):")
    stats = dataset_stats(data_path, sample_size=500)
    print(f"  avg_yoe : {stats['avg_yoe']}")
    print(f"  top titles: {stats['top_10_titles'][:5]}")
    print(f"  top skills: {stats['top_10_skills'][:5]}")