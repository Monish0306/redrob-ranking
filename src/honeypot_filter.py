"""
honeypot_filter.py  —  Phase 2
-------------------------------
Rule-based honeypot detection engine.
Runs BEFORE any scoring. Honeypots are a hard exclusion, not a graded penalty.

Rules are derived from:
  1. The JD's own stated examples ("expert proficiency in 10 skills with 0 years used")
  2. Our empirical analysis of the 100K dataset
  3. Basic career-history logic / date arithmetic

Key design decisions:
  - Rule-based, NOT learned → honeypots are logic/consistency problems
  - Each rule is an independent boolean check
  - We require ≥ 1 rule violation to EXCLUDE (calibrated against ~80 stated honeypots)
  - Every flagged candidate is logged with specific violation details
  - False positives (aggressive rules) risk removing real candidates → tune carefully
"""

import logging
from datetime import date, datetime
from typing import Dict, Any, List, Tuple

from src.config import (
    HONEYPOT_YOE_MISMATCH_THRESHOLD_YEARS,
    HONEYPOT_MIN_FLAGS_FOR_EXCLUSION,
)

logger = logging.getLogger(__name__)

TODAY = date(2026, 6, 18)   # competition reference date


# ─── Individual Rule Functions ────────────────────────────────────────────────
# Each returns (triggered: bool, detail: str)

def rule_expert_zero_duration(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    JD's own stated example: "expert proficiency in a skill with 0 months used."
    A real expert has months of actual usage. Zero duration = fabricated.
    """
    zero_expert_skills = [
        s["name"] for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 1) == 0
    ]
    if zero_expert_skills:
        return True, f"expert_zero_duration:{zero_expert_skills}"
    return False, ""


def rule_yoe_vs_career_mismatch(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Stated years_of_experience vs sum of career_history duration_months.
    A discrepancy > 5 years is suspicious — impossible timelines.
    Real candidates may have career gaps (0-2yr gap normal), but ±5yr is a flag.
    """
    stated_yoe = candidate["profile"].get("years_of_experience", 0)
    total_career_months = sum(
        h.get("duration_months", 0) for h in candidate.get("career_history", [])
    )
    total_career_years = total_career_months / 12.0

    diff = abs(total_career_years - stated_yoe)
    if diff > HONEYPOT_YOE_MISMATCH_THRESHOLD_YEARS:
        return True, (
            f"yoe_mismatch: stated={stated_yoe:.1f}yr "
            f"history_total={total_career_years:.1f}yr diff={diff:.1f}yr"
        )
    return False, ""


def rule_end_before_start(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    end_date < start_date in any career history entry.
    Physically impossible — clear data fabrication.
    """
    for h in candidate.get("career_history", []):
        try:
            if h.get("end_date"):
                start = datetime.strptime(h["start_date"], "%Y-%m-%d").date()
                end   = datetime.strptime(h["end_date"],   "%Y-%m-%d").date()
                if end < start:
                    return True, (
                        f"end_before_start: {h['company']} "
                        f"start={h['start_date']} end={h['end_date']}"
                    )
        except (ValueError, KeyError):
            pass
    return False, ""


def rule_duration_date_mismatch(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    stated duration_months vs actual (end_date - start_date) mismatch > 18 months.
    Small rounding is OK; 18+ months off in a single role is suspicious.
    """
    for h in candidate.get("career_history", []):
        try:
            start = datetime.strptime(h["start_date"], "%Y-%m-%d").date()
            end   = (
                datetime.strptime(h["end_date"], "%Y-%m-%d").date()
                if h.get("end_date")
                else TODAY
            )
            actual_months  = (end.year - start.year) * 12 + (end.month - start.month)
            stated_months  = h.get("duration_months", 0)
            diff = abs(actual_months - stated_months)
            if diff > 18:
                return True, (
                    f"duration_date_mismatch: {h['company']} "
                    f"stated={stated_months}mo actual={actual_months}mo diff={diff}mo"
                )
        except (ValueError, KeyError):
            pass
    return False, ""


def rule_overlapping_full_time_roles(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Two simultaneous full-time roles at different companies overlapping > 3 months.
    Note: Freelance/consulting careers can legitimately overlap — we only flag
    clearly non-overlappable situations (e.g., two 10001+ company full-time roles).
    """
    history = candidate.get("career_history", [])
    parsed  = []
    for h in history:
        try:
            s = datetime.strptime(h["start_date"], "%Y-%m-%d").date()
            e = (datetime.strptime(h["end_date"], "%Y-%m-%d").date()
                 if h.get("end_date") else TODAY)
            parsed.append((s, e, h["company"], h.get("company_size", "")))
        except (ValueError, KeyError):
            pass

    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            s1, e1, co1, sz1 = parsed[i]
            s2, e2, co2, sz2 = parsed[j]
            if co1 == co2:
                continue  # same company (promotion overlap) is fine
            overlap_start = max(s1, s2)
            overlap_end   = min(e1, e2)
            if overlap_start < overlap_end:
                overlap_months = (overlap_end - overlap_start).days / 30
                if overlap_months > 3:
                    return True, (
                        f"overlap: {co1}({sz1}) & {co2}({sz2}) "
                        f"overlap={overlap_months:.0f}mo"
                    )
    return False, ""


def rule_massive_skill_count_zero_evidence(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    20+ skills but near-zero average duration_months AND 5+ claimed as 'expert'.
    Catches mass keyword-stuffed profiles that didn't bother filling in durations.
    """
    skills = candidate.get("skills", [])
    if len(skills) < 20:
        return False, ""

    durations    = [s.get("duration_months", 0) for s in skills]
    avg_duration = sum(durations) / len(durations) if durations else 0
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")

    if avg_duration < 3 and expert_count >= 5:
        return True, (
            f"mass_skill_zero_evidence: skills={len(skills)} "
            f"experts={expert_count} avg_duration={avg_duration:.1f}mo"
        )
    return False, ""


def rule_future_career_dates(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    start_date or end_date in the future (beyond today).
    Real profiles don't have future employment dates.
    """
    for h in candidate.get("career_history", []):
        try:
            start = datetime.strptime(h["start_date"], "%Y-%m-%d").date()
            if start > TODAY:
                return True, f"future_start_date: {h['company']} start={h['start_date']}"
            if h.get("end_date"):
                end = datetime.strptime(h["end_date"], "%Y-%m-%d").date()
                if end > TODAY and not h.get("is_current", False):
                    return True, (f"future_end_date_non_current: "
                                  f"{h['company']} end={h['end_date']}")
        except (ValueError, KeyError):
            pass
    return False, ""


# ─── Rule Registry ────────────────────────────────────────────────────────────
# Order matters: cheapest (fastest) rules run first.
ALL_RULES = [
    ("expert_zero_duration",              rule_expert_zero_duration),
    ("yoe_vs_career_mismatch",            rule_yoe_vs_career_mismatch),
    ("end_before_start",                  rule_end_before_start),
    ("duration_date_mismatch",            rule_duration_date_mismatch),
    ("overlapping_full_time_roles",       rule_overlapping_full_time_roles),
    ("massive_skill_count_zero_evidence", rule_massive_skill_count_zero_evidence),
    ("future_career_dates",               rule_future_career_dates),
]


# ─── Main Filter Function ─────────────────────────────────────────────────────

def get_honeypot_flags(candidate: Dict[str, Any]) -> List[str]:
    """
    Run all rules against one candidate.
    Returns a list of triggered rule descriptions (empty = clean candidate).
    """
    triggered = []
    for rule_name, rule_fn in ALL_RULES:
        fired, detail = rule_fn(candidate)
        if fired:
            triggered.append(f"{rule_name}: {detail}")
    return triggered


def is_honeypot(candidate: Dict[str, Any]) -> bool:
    """
    Returns True if the candidate should be EXCLUDED from scoring.
    Threshold = HONEYPOT_MIN_FLAGS_FOR_EXCLUSION (default: 1)
    """
    flags = get_honeypot_flags(candidate)
    return len(flags) >= HONEYPOT_MIN_FLAGS_FOR_EXCLUSION


def filter_candidates(
    candidates: List[Dict[str, Any]],
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split candidates into (clean, flagged) lists.

    Args:
        candidates: List of validated candidate dicts from ingest.py
        verbose:    If True, log every flagged candidate + its violations

    Returns:
        (clean_candidates, flagged_candidates)
    """
    clean   = []
    flagged = []

    for c in candidates:
        cid   = c["candidate_id"]
        title = c["profile"]["current_title"]
        flags = get_honeypot_flags(c)

        if len(flags) >= HONEYPOT_MIN_FLAGS_FOR_EXCLUSION:
            flagged.append({
                "candidate_id": cid,
                "current_title": title,
                "flags": flags,
                "candidate": c,
            })
            if verbose:
                logger.warning(f"HONEYPOT EXCLUDED | {cid} | {title} | {flags}")
        else:
            if flags:
                # Single flag: suspicious but not excluded — annotate and keep
                c["_honeypot_warning"] = flags
                logger.debug(f"HONEYPOT WARNING  | {cid} | {title} | {flags}")
            clean.append(c)

    total    = len(clean) + len(flagged)
    hp_rate  = len(flagged) / total * 100 if total else 0

    logger.info(
        f"Honeypot filter: {len(clean)} clean, {len(flagged)} excluded "
        f"({hp_rate:.2f}% of input) from {total} total"
    )

    # Safety check: if we're excluding way more than expected, warn loudly
    if len(flagged) > 500:
        logger.warning(
            f"⚠️  Honeypot filter excluded {len(flagged)} candidates. "
            f"Expected ~80. Check rule thresholds — may be over-aggressive."
        )

    return clean, flagged


def honeypot_summary(flagged: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize what was flagged and why.
    Useful for the Stage 4 methodology documentation.
    """
    from collections import Counter
    rule_counts   = Counter()
    title_counts  = Counter()

    for entry in flagged:
        title_counts[entry["current_title"]] += 1
        for flag in entry["flags"]:
            rule_name = flag.split(":")[0]
            rule_counts[rule_name] += 1

    return {
        "total_excluded":       len(flagged),
        "rules_triggered":      dict(rule_counts),
        "top_excluded_titles":  title_counts.most_common(10),
    }


if __name__ == "__main__":
    # Smoke test — run: python src/honeypot_filter.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.ingest import load_all_candidates

    data_path = Path(__file__).parent.parent / "data" / "candidates.jsonl"
    print("Loading first 5000 candidates for honeypot test...")
    candidates = load_all_candidates(data_path, validate=True, max_records=5000)

    clean, flagged = filter_candidates(candidates, verbose=True)
    summary = honeypot_summary(flagged)

    print(f"\n✅ Filter complete:")
    print(f"   Clean    : {len(clean)}")
    print(f"   Flagged  : {len(flagged)}")
    print(f"   Summary  : {summary}")