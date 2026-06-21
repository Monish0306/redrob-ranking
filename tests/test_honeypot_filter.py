"""
tests/test_honeypot_filter.py
Unit tests for every honeypot detection rule.
Run: pytest tests/test_honeypot_filter.py -v

Each test is named for the exact scenario it catches.
"""

import pytest
import sys
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honeypot_filter import (
    rule_expert_zero_duration,
    rule_yoe_vs_career_mismatch,
    rule_end_before_start,
    rule_duration_date_mismatch,
    rule_overlapping_full_time_roles,
    rule_massive_skill_count_zero_evidence,
    rule_future_career_dates,
    get_honeypot_flags,
    is_honeypot,
    filter_candidates,
)


# ─── Base clean candidate fixture ────────────────────────────────────────────

def clean_candidate():
    """A perfectly valid, non-honeypot candidate."""
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": "ML Engineer",
            "years_of_experience": 5.0,
        },
        "career_history": [
            {
                "company": "Acme AI",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": "2025-01-01",
                "duration_months": 60,
                "is_current": False,
                "company_size": "51-200",
            },
            {
                "company": "BetaCorp",
                "title": "Software Engineer",
                "start_date": "2018-01-01",
                "end_date": "2019-12-31",
                "duration_months": 24,
                "is_current": False,
                "company_size": "201-500",
            },
        ],
        "skills": [
            {"name": "Python",  "proficiency": "expert",   "endorsements": 20, "duration_months": 48},
            {"name": "PyTorch", "proficiency": "advanced",  "endorsements": 10, "duration_months": 30},
        ],
        "redrob_signals": {
            "profile_completeness_score": 85,
            "signup_date": "2020-02-01",
        },
    }


# ─── Rule 1: Expert + Zero Duration ──────────────────────────────────────────

class TestExpertZeroDuration:

    def test_clean_candidate_not_flagged(self):
        fired, detail = rule_expert_zero_duration(clean_candidate())
        assert not fired

    def test_expert_with_zero_duration_flagged(self):
        """The JD's own stated example: expert skill, 0 months."""
        c = clean_candidate()
        c["skills"].append({
            "name": "Pinecone", "proficiency": "expert",
            "endorsements": 5, "duration_months": 0
        })
        fired, detail = rule_expert_zero_duration(c)
        assert fired
        assert "Pinecone" in detail

    def test_advanced_with_zero_duration_not_flagged(self):
        """Only 'expert' + 0 months triggers. Advanced with 0 months is suspicious but not this rule."""
        c = clean_candidate()
        c["skills"].append({
            "name": "SomeSkill", "proficiency": "advanced",
            "endorsements": 2, "duration_months": 0
        })
        fired, _ = rule_expert_zero_duration(c)
        assert not fired

    def test_expert_with_nonzero_duration_not_flagged(self):
        c = clean_candidate()
        c["skills"].append({
            "name": "RAG", "proficiency": "expert",
            "endorsements": 15, "duration_months": 24
        })
        fired, _ = rule_expert_zero_duration(c)
        assert not fired

    def test_multiple_expert_zero_skills_all_captured(self):
        c = clean_candidate()
        for skill in ["RAG", "Pinecone", "MLflow"]:
            c["skills"].append({
                "name": skill, "proficiency": "expert",
                "endorsements": 5, "duration_months": 0
            })
        fired, detail = rule_expert_zero_duration(c)
        assert fired
        assert "RAG" in detail

    def test_no_skills_not_flagged(self):
        c = clean_candidate()
        c["skills"] = []
        fired, _ = rule_expert_zero_duration(c)
        assert not fired


# ─── Rule 2: YoE vs Career History Mismatch ──────────────────────────────────

class TestYoEMismatch:

    def test_clean_candidate_not_flagged(self):
        fired, _ = rule_yoe_vs_career_mismatch(clean_candidate())
        assert not fired

    def test_stated_yoe_far_exceeds_history(self):
        """Stated 15yr, but only 8 months of career history recorded."""
        c = clean_candidate()
        c["profile"]["years_of_experience"] = 15.0
        c["career_history"] = [
            {
                "company": "FakeCorp", "title": "Manager",
                "start_date": "2025-10-01", "end_date": None,
                "duration_months": 8, "is_current": True,
                "company_size": "51-200",
            }
        ]
        fired, detail = rule_yoe_vs_career_mismatch(c)
        assert fired
        assert "yoe_mismatch" in detail

    def test_history_far_exceeds_stated_yoe(self):
        """Career history implies 25+ years but stated YoE = 10."""
        c = clean_candidate()
        c["profile"]["years_of_experience"] = 10.0
        c["career_history"][0]["duration_months"] = 228  # 19 years at one company
        c["career_history"][1]["duration_months"] = 120  # 10 years at another
        fired, detail = rule_yoe_vs_career_mismatch(c)
        assert fired

    def test_small_gap_not_flagged(self):
        """2-year gap between stated YoE and history is normal (career breaks)."""
        c = clean_candidate()
        c["profile"]["years_of_experience"] = 7.0
        # history = 60+24 = 84 months = 7 years → matches
        fired, _ = rule_yoe_vs_career_mismatch(c)
        assert not fired


# ─── Rule 3: End Before Start ─────────────────────────────────────────────────

class TestEndBeforeStart:

    def test_clean_candidate_not_flagged(self):
        fired, _ = rule_end_before_start(clean_candidate())
        assert not fired

    def test_end_date_before_start_date_flagged(self):
        c = clean_candidate()
        c["career_history"][0]["start_date"] = "2022-06-01"
        c["career_history"][0]["end_date"]   = "2021-01-01"  # before start!
        fired, detail = rule_end_before_start(c)
        assert fired
        assert "Acme AI" in detail

    def test_null_end_date_not_flagged(self):
        """Current role has end_date=None → skip this check."""
        c = clean_candidate()
        c["career_history"][0]["end_date"] = None
        fired, _ = rule_end_before_start(c)
        assert not fired

    def test_same_start_end_not_flagged(self):
        """Same day start and end — edge case, shouldn't flag."""
        c = clean_candidate()
        c["career_history"][0]["start_date"] = "2022-01-01"
        c["career_history"][0]["end_date"]   = "2022-01-01"
        fired, _ = rule_end_before_start(c)
        assert not fired


# ─── Rule 4: Duration vs Date Mismatch ────────────────────────────────────────

class TestDurationDateMismatch:

    def test_clean_candidate_not_flagged(self):
        fired, _ = rule_duration_date_mismatch(clean_candidate())
        assert not fired

    def test_large_duration_mismatch_flagged(self):
        """Start 2020-01, End 2021-01 → 12 actual months. Stated 36. Diff=24 > 18."""
        c = clean_candidate()
        c["career_history"][0]["start_date"]      = "2020-01-01"
        c["career_history"][0]["end_date"]        = "2021-01-01"
        c["career_history"][0]["duration_months"] = 36  # should be 12
        fired, detail = rule_duration_date_mismatch(c)
        assert fired

    def test_small_rounding_not_flagged(self):
        """6-month discrepancy is fine (rounding, part-month counting)."""
        c = clean_candidate()
        c["career_history"][0]["start_date"]      = "2020-01-01"
        c["career_history"][0]["end_date"]        = "2025-01-01"
        c["career_history"][0]["duration_months"] = 66   # actual=60, diff=6 < 18
        fired, _ = rule_duration_date_mismatch(c)
        assert not fired


# ─── Rule 5: Overlapping Full-Time Roles ──────────────────────────────────────

class TestOverlappingRoles:

    def test_clean_sequential_career_not_flagged(self):
        fired, _ = rule_overlapping_full_time_roles(clean_candidate())
        assert not fired

    def test_significant_overlap_flagged(self):
        """Two different companies, 12-month overlap → flag."""
        c = clean_candidate()
        c["career_history"] = [
            {
                "company": "CompanyA", "title": "Engineer",
                "start_date": "2020-01-01", "end_date": "2023-01-01",
                "duration_months": 36, "is_current": False, "company_size": "51-200",
            },
            {
                "company": "CompanyB", "title": "Engineer",
                "start_date": "2021-06-01", "end_date": "2024-01-01",
                "duration_months": 31, "is_current": False, "company_size": "51-200",
            },
        ]
        fired, detail = rule_overlapping_full_time_roles(c)
        assert fired

    def test_same_company_overlap_not_flagged(self):
        """Same company, overlapping dates = promotion / role transition. Fine."""
        c = clean_candidate()
        c["career_history"] = [
            {
                "company": "BigCorp", "title": "Senior Engineer",
                "start_date": "2022-01-01", "end_date": "2024-01-01",
                "duration_months": 24, "is_current": False, "company_size": "10001+",
            },
            {
                "company": "BigCorp", "title": "Engineer",
                "start_date": "2020-01-01", "end_date": "2022-06-01",
                "duration_months": 30, "is_current": False, "company_size": "10001+",
            },
        ]
        fired, _ = rule_overlapping_full_time_roles(c)
        assert not fired


# ─── Rule 6: Massive Skill Count + Zero Evidence ──────────────────────────────

class TestMassiveSkillZeroEvidence:

    def test_clean_candidate_not_flagged(self):
        fired, _ = rule_massive_skill_count_zero_evidence(clean_candidate())
        assert not fired

    def test_many_skills_with_decent_duration_not_flagged(self):
        """25 skills but avg duration = 18 months → legitimate."""
        c = clean_candidate()
        c["skills"] = [
            {"name": f"Skill{i}", "proficiency": "advanced",
             "endorsements": 5, "duration_months": 18}
            for i in range(25)
        ]
        fired, _ = rule_massive_skill_count_zero_evidence(c)
        assert not fired

    def test_many_experts_near_zero_duration_flagged(self):
        """20+ skills, 5+ experts, avg duration < 3 months → keyword stuffer."""
        c = clean_candidate()
        c["skills"] = []
        # 5 "expert" skills with 0 duration
        for i in range(5):
            c["skills"].append({
                "name": f"ExpertSkill{i}", "proficiency": "expert",
                "endorsements": 2, "duration_months": 1
            })
        # 15 more skills with 1-2 months
        for i in range(15):
            c["skills"].append({
                "name": f"Skill{i}", "proficiency": "intermediate",
                "endorsements": 1, "duration_months": 2
            })
        fired, detail = rule_massive_skill_count_zero_evidence(c)
        assert fired


# ─── Rule 7: Future Career Dates ──────────────────────────────────────────────

class TestFutureCareerDates:

    def test_clean_candidate_not_flagged(self):
        fired, _ = rule_future_career_dates(clean_candidate())
        assert not fired

    def test_future_start_date_flagged(self):
        c = clean_candidate()
        c["career_history"][0]["start_date"] = "2030-01-01"  # future
        fired, detail = rule_future_career_dates(c)
        assert fired
        assert "future_start_date" in detail

    def test_future_end_date_non_current_flagged(self):
        c = clean_candidate()
        c["career_history"][0]["end_date"]   = "2028-12-31"
        c["career_history"][0]["is_current"] = False
        fired, detail = rule_future_career_dates(c)
        assert fired

    def test_current_role_no_end_date_not_flagged(self):
        c = clean_candidate()
        c["career_history"][0]["end_date"]   = None
        c["career_history"][0]["is_current"] = True
        fired, _ = rule_future_career_dates(c)
        assert not fired


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestFilterCandidates:

    def test_clean_candidate_passes_filter(self):
        candidates = [clean_candidate()]
        clean, flagged = filter_candidates(candidates)
        assert len(clean) == 1
        assert len(flagged) == 0

    def test_honeypot_is_excluded(self):
        c = clean_candidate()
        # Give it the JD's own stated honeypot example
        c["skills"].append({
            "name": "RAG", "proficiency": "expert",
            "endorsements": 10, "duration_months": 0
        })
        candidates = [clean_candidate(), c]
        # clean_candidate() has id CAND_0000001; our honeypot also has CAND_0000001
        # Fix: give honeypot a unique ID
        c["candidate_id"] = "CAND_0000002"
        clean, flagged = filter_candidates([clean_candidate(), c])
        assert len(flagged) == 1
        assert flagged[0]["candidate_id"] == "CAND_0000002"

    def test_mixed_batch(self):
        """3 clean + 2 honeypots → correct split."""
        candidates = []
        # 3 clean
        for i in range(1, 4):
            c = clean_candidate()
            c["candidate_id"] = f"CAND_{i:07d}"
            candidates.append(c)

        # 2 honeypots
        for i in range(4, 6):
            c = clean_candidate()
            c["candidate_id"] = f"CAND_{i:07d}"
            c["profile"]["years_of_experience"] = 15.0
            c["career_history"] = [{
                "company": "FakeCorp", "title": "CEO",
                "start_date": "2026-01-01", "end_date": None,
                "duration_months": 5, "is_current": True,
                "company_size": "1-10",
            }]
            candidates.append(c)

        clean, flagged = filter_candidates(candidates)
        assert len(clean)   == 3
        assert len(flagged) == 2

    def test_is_honeypot_function(self):
        c = clean_candidate()
        assert not is_honeypot(c)

        c["skills"].append({
            "name": "FAISS", "proficiency": "expert",
            "endorsements": 3, "duration_months": 0
        })
        assert is_honeypot(c)