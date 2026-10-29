"""Reconcile broad and skill-resolved wage accounts without reallocating residuals."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .positions import numeric, safe_divide


SKILLS: Tuple[str, str, str] = ("Low", "Medium", "High")


@dataclass
class WageAccount:
    """一个可审计的工资账户来源。"""

    source_module: str
    source_file: str
    wage_account_name: str
    unit: str
    total_wage: float
    skill_resolved: bool
    included_skills: str
    included_residual_categories: str


def _interpret_account(account: WageAccount, descriptive_total: float, policy_total: float) -> str:
    """根据账户覆盖范围解释工资总量差异。"""
    if descriptive_total > 0 and account.skill_resolved and account.total_wage < descriptive_total * 0.999:
        return "expected_skill-resolved wage account excludes residual wage component"
    if np.isclose(account.total_wage, descriptive_total, rtol=1e-6, atol=1e-6):
        return "matches descriptive broad wage total"
    if policy_total > 0 and np.isclose(account.total_wage, policy_total, rtol=1e-6, atol=1e-6):
        return "matches policy model skill-resolved wage total"
    return "requires review before relabelling results"


def compute_wage_reconciliation(accounts: Sequence[WageAccount]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成工资账户 reconciliation 表和口径审计表。"""
    if not accounts:
        raise ValueError("At least one wage account is required")
    descriptive_candidates = [a for a in accounts if "descriptive" in a.source_module.lower() and not a.skill_resolved]
    descriptive_total = float(descriptive_candidates[0].total_wage if descriptive_candidates else max(a.total_wage for a in accounts))
    policy_candidates = [a for a in accounts if "policy" in a.source_module.lower()]
    policy_total = float(policy_candidates[0].total_wage if policy_candidates else min(a.total_wage for a in accounts))

    rows: List[Dict[str, Any]] = []
    for account in accounts:
        rows.append(
            {
                "Source_Module": account.source_module,
                "Source_File": account.source_file,
                "Wage_Account_Name": account.wage_account_name,
                "Unit": account.unit,
                "Total_Wage": float(account.total_wage),
                "Skill_Resolved": bool(account.skill_resolved),
                "Included_Skills": account.included_skills,
                "Included_Residual_Categories": account.included_residual_categories,
                "Share_of_Descriptive_Total": account.total_wage / descriptive_total if descriptive_total > 0 else np.nan,
                "Share_of_Policy_Total": account.total_wage / policy_total if policy_total > 0 else np.nan,
                "Interpretation": _interpret_account(account, descriptive_total, policy_total),
            }
        )
    result = pd.DataFrame(rows).sort_values(["Source_Module", "Wage_Account_Name"]).reset_index(drop=True)

    residual_gap = descriptive_total - policy_total
    expected_gap = residual_gap >= -max(abs(descriptive_total), 1.0) * 1e-6
    audit = pd.DataFrame(
        [
            {
                "Audit_Name": "descriptive total greater than or equal to policy skill-resolved total",
                "Observed": descriptive_total,
                "Expected": policy_total,
                "Difference": residual_gap,
                "Passed": bool(expected_gap),
                "Interpretation": "expected if policy model uses Low/Medium/High skill-resolved wages and excludes residual categories",
            },
            {
                "Audit_Name": "wage reconciliation conclusion",
                "Observed": float(expected_gap),
                "Expected": 1.0,
                "Difference": float(expected_gap) - 1.0,
                "Passed": bool(expected_gap),
                "Interpretation": "Result 1 broad wage labels and policy skill-resolved wage labels should remain distinct",
            },
        ]
    )
    return result, audit


def write_outputs(reconciliation: pd.DataFrame, audit: pd.DataFrame, output_dir: Path) -> None:
    """写出工资口径 reconciliation CSV。"""
    csv_dir = output_dir if output_dir.name == "csv" else output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    reconciliation.to_csv(csv_dir / "wage_baseline_reconciliation.csv", index=False)
    audit.to_csv(csv_dir / "wage_account_mapping_audit.csv", index=False)
    print(f"保存: {csv_dir / 'wage_baseline_reconciliation.csv'}")
    print(f"保存: {csv_dir / 'wage_account_mapping_audit.csv'}")


def build_wage_account_summary(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize wage accounts using their broad and policy denominators."""
    reconciliation = reconciliation_df.copy()
    broad_row = reconciliation.loc[
        reconciliation["Wage_Account_Name"].astype(str).str.contains("broad wage", case=False, na=False)
        | reconciliation["Source_Module"].astype(str).str.contains("footprint engine", case=False, na=False)
    ]
    if broad_row.empty:
        broad_total = float(numeric(reconciliation["Total_Wage"]).max())
    else:
        broad_total = float(broad_row.iloc[0]["Total_Wage"])
    policy_row = reconciliation.loc[
        reconciliation["Source_Module"].astype(str).str.contains("policy", case=False, na=False)
        | reconciliation["Wage_Account_Name"].astype(str).str.contains("policy", case=False, na=False)
    ]
    policy_total = float(policy_row.iloc[0]["Total_Wage"]) if not policy_row.empty else float(numeric(reconciliation["Total_Wage"]).min())

    summary_rows = [
        {
            "Account_Name": "Broad descriptive wage account",
            "Use_In_Analysis": "aggregate descriptive accounting",
            "Total_M_EUR": broad_total,
            "Share_of_Broad_Descriptive_Wage_Total": 1.0,
            "Skill_Resolved": False,
            "Residual_Categories_Included": True,
            "Notes": "Used for broad descriptive wage totals; includes residual compensation categories.",
        },
        {
            "Account_Name": "Low/Medium/High skill-resolved wage account",
            "Use_In_Analysis": "skill comparisons; inequality diagnostics; policy wage outcomes",
            "Total_M_EUR": policy_total,
            "Share_of_Broad_Descriptive_Wage_Total": safe_divide(policy_total, broad_total),
            "Skill_Resolved": True,
            "Residual_Categories_Included": False,
            "Notes": "Used where Low/Medium/High skill attribution is required; residual categories are not force-allocated.",
        },
        {
            "Account_Name": "Residual wage component outside Low/Medium/High",
            "Use_In_Analysis": "documented residual; not allocated in the main skill diagnostics",
            "Total_M_EUR": broad_total - policy_total,
            "Share_of_Broad_Descriptive_Wage_Total": safe_divide(broad_total - policy_total, broad_total),
            "Skill_Resolved": False,
            "Residual_Categories_Included": True,
            "Notes": "Difference between broad descriptive wage and skill-resolved policy wage baseline.",
        },
    ]
    summary = pd.DataFrame(summary_rows)
    return summary



def wage_account_summary(reconciliation_df: pd.DataFrame) -> pd.DataFrame:
    """Return the wage reconciliation summary without reading or writing files."""
    return build_wage_account_summary(reconciliation_df)
