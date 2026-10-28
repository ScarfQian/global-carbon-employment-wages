"""Sensitivity of skill-carbon attribution to employment, hours and wage weights."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SKILLS: Tuple[str, str, str] = ("Low", "Medium", "High")
METHODS: Tuple[Tuple[str, str], ...] = (
    ("employment_share", "Employment"),
    ("labour_hour_share", "Hours"),
    ("wage_share", "Wage"),
)




def _safe_share(values: pd.Series, total: float) -> pd.Series:
    """安全份额计算。"""
    return values / total if total > 0 else pd.Series(np.nan, index=values.index, dtype=float)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """安全 ratio 计算，零分母返回 NaN。"""
    out = np.full(len(num), np.nan, dtype=float)
    np.divide(num.to_numpy(dtype=float), den.to_numpy(dtype=float), out=out, where=den.to_numpy(dtype=float) > 0)
    return pd.Series(out, index=num.index)


def _allocate(carbon: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按技能权重份额分配 sector carbon，并返回未归属碳。"""
    carbon_1d = np.asarray(carbon, dtype=float).reshape(-1)
    weight_total = weights.sum(axis=0)
    alpha = np.divide(weights, weight_total.reshape(1, -1), out=np.zeros_like(weights, dtype=float), where=weight_total.reshape(1, -1) > 0)
    attributed = alpha * carbon_1d.reshape(1, -1)
    unattributed = np.where((weight_total <= 0) & (carbon_1d > 0), carbon_1d, 0.0)
    return attributed, unattributed


def _region_skill_table(
    perspective: str,
    regions: Sequence[str],
    carbon: np.ndarray,
    employment: np.ndarray,
    hours: np.ndarray,
    wage: np.ndarray,
) -> pd.DataFrame:
    """生成一种 attribution method 下的 region-skill ratio 表。"""
    rows: List[Dict[str, Any]] = []
    region_index = pd.Index(regions)
    for method, weight_name in METHODS:
        weights = {"Employment": employment, "Hours": hours, "Wage": wage}[weight_name]
        carbon_attr, unattributed = _allocate(carbon, weights)
        for skill_idx, skill in enumerate(SKILLS):
            for region in region_index.unique():
                mask = region_index.to_numpy(dtype=object) == region
                rows.append(
                    {
                        "Attribution_Method": method,
                        "Accounting_Perspective": perspective,
                        "Region": str(region),
                        "Skill": skill,
                        "Carbon_Attributed": float(carbon_attr[skill_idx, mask].sum()),
                        "Employment": float(employment[skill_idx, mask].sum()),
                        "Hours": float(hours[skill_idx, mask].sum()),
                        "Wage": float(wage[skill_idx, mask].sum()),
                        "Total_Carbon": float(carbon[mask].sum()),
                        "Unattributed_Carbon": float(unattributed[mask].sum()),
                    }
                )
    result = pd.DataFrame(rows)
    result["Attribution_Coverage"] = 1.0 - pd.Series(
        np.divide(
            result["Unattributed_Carbon"].to_numpy(dtype=float),
            result["Total_Carbon"].to_numpy(dtype=float),
            out=np.full(len(result), np.nan, dtype=float),
            where=result["Total_Carbon"].to_numpy(dtype=float) > 0,
        ),
        index=result.index,
    )
    for method in result["Attribution_Method"].unique():
        mask = result["Attribution_Method"].eq(method)
        c_share = _safe_share(result.loc[mask, "Carbon_Attributed"], float(result.loc[mask, "Carbon_Attributed"].sum()))
        e_share = _safe_share(result.loc[mask, "Employment"], float(result.loc[mask, "Employment"].sum()))
        w_share = _safe_share(result.loc[mask, "Wage"], float(result.loc[mask, "Wage"].sum()))
        result.loc[mask, "Carbon_Share"] = c_share
        result.loc[mask, "R_CE"] = _ratio(c_share, e_share)
        result.loc[mask, "R_CW"] = _ratio(c_share, w_share)
    result["Rank_R_CE"] = result.groupby(["Attribution_Method", "Accounting_Perspective"])["R_CE"].rank(ascending=False, method="dense")
    result["Rank_R_CW"] = result.groupby(["Attribution_Method", "Accounting_Perspective"])["R_CW"].rank(ascending=False, method="dense")
    return result.sort_values(["Accounting_Perspective", "Attribution_Method", "Region", "Skill"]).reset_index(drop=True)


def compute_attribution_robustness(inputs: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算 attribution coverage、robustness 明细和 summary。"""
    qE = np.asarray(inputs.qE, dtype=float)
    qH = np.asarray(inputs.qH, dtype=float)
    qW = np.asarray(inputs.qW, dtype=float)
    production_carbon = np.asarray(inputs.qC, dtype=float) * np.asarray(inputs.x, dtype=float)
    production_emp = qE * np.asarray(inputs.x, dtype=float).reshape(1, -1)
    production_hours = qH * np.asarray(inputs.x, dtype=float).reshape(1, -1)
    production_wage = qW * np.asarray(inputs.x, dtype=float).reshape(1, -1)
    production = _region_skill_table(
        "Production",
        inputs.producer_regions,
        production_carbon,
        production_emp,
        production_hours,
        production_wage,
    )

    output_response = np.asarray(inputs.L, dtype=float) @ np.asarray(inputs.Y, dtype=float)
    fd_regions = pd.Index(inputs.final_demand_regions)
    frames = [production]
    for destination in fd_regions.unique():
        columns = np.where(fd_regions.to_numpy(dtype=object) == destination)[0]
        induced = output_response[:, columns].sum(axis=1)
        frames.append(
            _region_skill_table(
                "Consumption",
                [str(destination)] * len(induced),
                np.asarray(inputs.qC, dtype=float) * induced,
                qE * induced.reshape(1, -1),
                qH * induced.reshape(1, -1),
                qW * induced.reshape(1, -1),
            )
        )
    robustness = pd.concat(frames, ignore_index=True)
    parent = robustness[
        ["Attribution_Method", "Accounting_Perspective", "Region", "Total_Carbon", "Unattributed_Carbon"]
    ].drop_duplicates()
    coverage = parent.groupby(["Attribution_Method", "Accounting_Perspective"], as_index=False, observed=False)[
        ["Total_Carbon", "Unattributed_Carbon"]
    ].sum()
    attributed = robustness.groupby(["Attribution_Method", "Accounting_Perspective"], as_index=False, observed=False)[
        "Carbon_Attributed"
    ].sum()
    coverage = coverage.merge(attributed, on=["Attribution_Method", "Accounting_Perspective"], how="left")
    coverage["Attribution_Coverage"] = 1.0 - pd.Series(
        np.divide(
            coverage["Unattributed_Carbon"].to_numpy(dtype=float),
            coverage["Total_Carbon"].to_numpy(dtype=float),
            out=np.full(len(coverage), np.nan, dtype=float),
            where=coverage["Total_Carbon"].to_numpy(dtype=float) > 0,
        ),
        index=coverage.index,
    )

    base = robustness.loc[robustness["Attribution_Method"].eq("employment_share")]
    summary_rows: List[Dict[str, Any]] = []
    for method in ["labour_hour_share", "wage_share"]:
        alt = robustness.loc[robustness["Attribution_Method"].eq(method)]
        merged = base.merge(
            alt,
            on=["Accounting_Perspective", "Region", "Skill"],
            suffixes=("_employment_share", f"_{method}"),
        )
        summary_rows.append(
            {
                "Attribution_Method": method,
                "Rank_Correlation_R_CE": merged["R_CE_employment_share"].corr(merged[f"R_CE_{method}"], method="spearman"),
                "Rank_Correlation_R_CW": merged["R_CW_employment_share"].corr(merged[f"R_CW_{method}"], method="spearman"),
                "Top_Cell_Overlap_R_CE_10": int(
                    len(
                        set(merged.nlargest(10, "R_CE_employment_share").index).intersection(
                            set(merged.nlargest(10, f"R_CE_{method}").index)
                        )
                    )
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return coverage, robustness, summary


def write_outputs(coverage: pd.DataFrame, robustness: pd.DataFrame, summary: pd.DataFrame, project_root: Path) -> None:
    """写出 3.足迹计算 与 Supplementary 两套结果。"""
    csv3 = project_root / "01_Country_Skill" / "csv"
    supp = project_root / "03_Network_And_Lockin" / "csv"
    csv3.mkdir(parents=True, exist_ok=True)
    supp.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(csv3 / "skill_carbon_attribution_coverage.csv", index=False)
    robustness.to_csv(csv3 / "skill_carbon_attribution_robustness.csv", index=False)
    robustness.to_csv(supp / "skill_carbon_attribution_robustness.csv", index=False)
    summary.to_csv(supp / "skill_carbon_attribution_robustness_summary.csv", index=False)
    print(f"保存: {csv3 / 'skill_carbon_attribution_coverage.csv'}")
    print(f"保存: {csv3 / 'skill_carbon_attribution_robustness.csv'}")
    print(f"保存: {supp / 'skill_carbon_attribution_robustness.csv'}")
    print(f"保存: {supp / 'skill_carbon_attribution_robustness_summary.csv'}")
