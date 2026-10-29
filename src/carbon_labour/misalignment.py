"""Production and consumption skill-stratified misalignment diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SKILLS: Tuple[str, str, str] = ("Low", "Medium", "High")
RATIO_COLUMNS: Tuple[str, ...] = ("R_CE", "R_WE", "R_CW", "R_CH", "R_WH", "R_HE")
LOG_RATIO_COLUMNS: Tuple[str, ...] = tuple(f"log_{name}" for name in RATIO_COLUMNS)
COVERAGE_COLUMNS: List[str] = [
    "Total_Carbon",
    "Attributed_Carbon",
    "Unattributed_Carbon",
    "Unattributed_Carbon_Share",
    "Attribution_Coverage",
    "Attribution_Status",
]

COUNTRY_SKILL_COLUMNS: List[str] = [
    "Accounting_Perspective",
    "Region",
    "Skill",
    "Carbon_Attributed",
    "Employment",
    "Hours",
    "Wage",
    *COVERAGE_COLUMNS,
    "Carbon_Share",
    "Employment_Share",
    "Hours_Share",
    "Wage_Share",
    "R_CE",
    "R_WE",
    "R_CW",
    "R_CH",
    "R_WH",
    "R_HE",
    "log_R_CE",
    "log_R_WE",
    "log_R_CW",
    "log_R_CH",
    "log_R_WH",
    "log_R_HE",
    "Quadrant_CE_WE",
    "Valid_For_Ratio",
]

COUNTRY_SECTOR_SKILL_COLUMNS: List[str] = [
    "Accounting_Perspective",
    "Region",
    "Sector",
    "Sector_Code",
    "Skill",
    "Carbon_Attributed",
    "Employment",
    "Hours",
    "Wage",
    *COVERAGE_COLUMNS,
    "Carbon_Share",
    "Employment_Share",
    "Hours_Share",
    "Wage_Share",
    "R_CE",
    "R_WE",
    "R_CW",
    "R_CH",
    "R_WH",
    "R_HE",
    "log_R_CE",
    "log_R_WE",
    "log_R_CW",
    "log_R_CH",
    "log_R_WH",
    "log_R_HE",
    "Quadrant_CE_WE",
    "Valid_For_Ratio",
]


@dataclass
class MisalignmentInputs:
    """MRIO arrays needed to construct skill-attributed carbon diagnostics."""

    qC: np.ndarray
    qE: np.ndarray
    qH: np.ndarray
    qW: np.ndarray
    x: np.ndarray
    L: np.ndarray
    Y: np.ndarray
    producer_regions: Sequence[str]
    sector_labels: Sequence[str]
    final_demand_regions: Sequence[str]
    skills: Sequence[str] = SKILLS
    sector_codes: Optional[Sequence[str]] = None


def _as_1d(name: str, values: Any) -> np.ndarray:
    """转换为一维 float array，并显式检查非有限值。"""
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_2d(name: str, values: Any) -> np.ndarray:
    """转换为二维 float array，并显式检查非有限值。"""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_inputs(inputs: MisalignmentInputs) -> MisalignmentInputs:
    """检查所有 MRIO 输入维度，避免后续矩阵运算静默错位。"""
    qC = _as_1d("qC", inputs.qC)
    qE = _as_2d("qE", inputs.qE)
    qH = _as_2d("qH", inputs.qH)
    qW = _as_2d("qW", inputs.qW)
    x = _as_1d("x", inputs.x)
    L = _as_2d("L", inputs.L)
    Y = _as_2d("Y", inputs.Y)

    n = qC.size
    skills = list(inputs.skills)
    if qE.shape != (len(skills), n) or qH.shape != (len(skills), n) or qW.shape != (len(skills), n):
        raise ValueError("qE/qH/qW must have shape (n_skills, n_country_sector_rows)")
    if x.size != n or L.shape != (n, n) or Y.shape[0] != n:
        raise ValueError("x, L, and Y dimensions are inconsistent with qC")
    if len(inputs.producer_regions) != n or len(inputs.sector_labels) != n:
        raise ValueError("producer_regions and sector_labels must match qC length")
    if len(inputs.final_demand_regions) != Y.shape[1]:
        raise ValueError("final_demand_regions must match Y columns")
    if inputs.sector_codes is not None and len(inputs.sector_codes) != n:
        raise ValueError("sector_codes must match qC length")

    return MisalignmentInputs(
        qC=qC,
        qE=qE,
        qH=qH,
        qW=qW,
        x=x,
        L=L,
        Y=Y,
        producer_regions=list(inputs.producer_regions),
        sector_labels=list(inputs.sector_labels),
        final_demand_regions=list(inputs.final_demand_regions),
        skills=skills,
        sector_codes=list(inputs.sector_codes) if inputs.sector_codes is not None else list(inputs.sector_labels),
    )


def _safe_share(values: pd.Series, total: float) -> pd.Series:
    """计算全局份额；总量为 0 时返回 NaN，避免伪造 0 份额。"""
    if total > 0.0:
        return values.astype(float) / total
    return pd.Series(np.nan, index=values.index, dtype=float)


def _share_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """计算 share-normalized ratio；分母份额为 0 时返回 NaN。"""
    numerator_array = numerator.to_numpy(dtype=float)
    denominator_array = denominator.to_numpy(dtype=float)
    result = np.full_like(numerator_array, np.nan, dtype=float)
    np.divide(
        numerator_array,
        denominator_array,
        out=result,
        where=np.isfinite(denominator_array) & (denominator_array > 0.0),
    )
    return pd.Series(result, index=numerator.index, dtype=float)


def _log_ratio(values: pd.Series) -> pd.Series:
    """对正 ratio 取对数；0 或 NaN ratio 保持 NaN，避免导出 -inf。"""
    array = values.to_numpy(dtype=float)
    result = np.full_like(array, np.nan, dtype=float)
    mask = np.isfinite(array) & (array > 0.0)
    result[mask] = np.log(array[mask])
    return pd.Series(result, index=values.index, dtype=float)


def add_misalignment_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """为 analytical units 增加份额、ratio、log ratio 和 Figure 1 象限。"""
    result = df.copy()
    totals = {
        "Carbon": float(result["Carbon_Attributed"].sum()),
        "Employment": float(result["Employment"].sum()),
        "Hours": float(result["Hours"].sum()),
        "Wage": float(result["Wage"].sum()),
    }
    result["Carbon_Share"] = _safe_share(result["Carbon_Attributed"], totals["Carbon"])
    result["Employment_Share"] = _safe_share(result["Employment"], totals["Employment"])
    result["Hours_Share"] = _safe_share(result["Hours"], totals["Hours"])
    result["Wage_Share"] = _safe_share(result["Wage"], totals["Wage"])

    result["R_CE"] = _share_ratio(result["Carbon_Share"], result["Employment_Share"])
    result["R_WE"] = _share_ratio(result["Wage_Share"], result["Employment_Share"])
    result["R_CW"] = _share_ratio(result["Carbon_Share"], result["Wage_Share"])
    result["R_CH"] = _share_ratio(result["Carbon_Share"], result["Hours_Share"])
    result["R_WH"] = _share_ratio(result["Wage_Share"], result["Hours_Share"])
    result["R_HE"] = _share_ratio(result["Hours_Share"], result["Employment_Share"])

    for ratio_col in RATIO_COLUMNS:
        result[f"log_{ratio_col}"] = _log_ratio(result[ratio_col])

    denominator_valid = (
        (totals["Carbon"] > 0.0)
        & (totals["Employment"] > 0.0)
        & (totals["Hours"] > 0.0)
        & (totals["Wage"] > 0.0)
        & (result["Employment_Share"] > 0.0)
        & (result["Hours_Share"] > 0.0)
        & (result["Wage_Share"] > 0.0)
    )
    result["Valid_For_Ratio"] = denominator_valid.astype(bool)
    result["Quadrant_CE_WE"] = classify_quadrants(result["log_R_CE"], result["log_R_WE"], result["Valid_For_Ratio"])
    return result


def classify_quadrants(log_r_ce: pd.Series, log_r_we: pd.Series, valid: pd.Series) -> pd.Series:
    """按 CE 与 WE 的 log ratio 对 country-skill 单元分象限。"""
    labels = np.full(len(log_r_ce), "Undefined", dtype=object)
    ce = log_r_ce.to_numpy(dtype=float)
    we = log_r_we.to_numpy(dtype=float)
    valid_mask = valid.to_numpy(dtype=bool) & np.isfinite(ce) & np.isfinite(we)

    labels[valid_mask & (ce > 0.0) & (we < 0.0)] = "High-carbon low-return"
    labels[valid_mask & (ce > 0.0) & (we >= 0.0)] = "High-carbon high-return"
    labels[valid_mask & (ce <= 0.0) & (we < 0.0)] = "Low-carbon low-return"
    labels[valid_mask & (ce <= 0.0) & (we >= 0.0)] = "Low-carbon high-return"
    return pd.Series(labels, index=log_r_ce.index, dtype=object)


def _attribute_carbon_by_employment(
    carbon: np.ndarray,
    employment_by_skill: np.ndarray,
    perspective: str,
    region_label: str,
    sector_labels: Sequence[str],
    sector_codes: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """使用就业份额将 sector-level carbon 归因到技能，并显式保留未归属碳。

    当某个 sector/cell 的 skill-resolved employment 合计为 0 时，正碳排不分配给
    Low/Medium/High，而是作为 attribution-boundary 的 unattributed carbon 进入审计。
    """
    employment_total = employment_by_skill.sum(axis=0)
    alpha = np.zeros_like(employment_by_skill, dtype=float)
    positive = employment_total > 0.0
    alpha[:, positive] = employment_by_skill[:, positive] / employment_total[positive].reshape(1, -1)
    attributed = alpha * carbon.reshape(1, -1)

    carbon_1d = np.asarray(carbon, dtype=float).reshape(-1)
    unattributed = np.where((~positive) & (carbon_1d > 0.0), carbon_1d, 0.0)
    attributable = np.where(positive, carbon_1d, 0.0)
    coverage = np.divide(
        attributable,
        carbon_1d,
        out=np.full_like(carbon_1d, np.nan, dtype=float),
        where=carbon_1d > 0.0,
    )
    unattributed_share = np.divide(
        unattributed,
        carbon_1d,
        out=np.full_like(carbon_1d, np.nan, dtype=float),
        where=carbon_1d > 0.0,
    )
    status = np.full(carbon_1d.shape, "not_applicable", dtype=object)
    status[(carbon_1d > 0.0) & (unattributed == 0.0)] = "fully_attributed"
    status[unattributed > 0.0] = "partially_unattributed"

    zero_positions = np.where((~positive) & (carbon_1d > 0.0))[0]
    zero_rows: List[Dict[str, Any]] = []
    if zero_positions.size:
        examples = [
            f"{region_label}:{sector_codes[idx]}:{sector_labels[idx]}"
            for idx in zero_positions[:10]
        ]
        zero_rows.append(
            {
                "Audit_Category": "Attribution_Coverage",
                "Audit_Name": "zero skill employment with positive carbon",
                "Accounting_Perspective": perspective,
                "Unit_Level": "sector",
                "Observed": float(zero_positions.size),
                "Expected": float(zero_positions.size),
                "Difference": 0.0,
                "Relative_Difference": 0.0,
                "Passed": True,
                "Interpretation": "expected_unattributed_carbon_due_to_zero_skill_employment",
                "Details": "; ".join(examples),
            }
        )
    coverage_arrays = {
        "Total_Carbon": carbon_1d,
        "Attributed_Carbon": attributable,
        "Unattributed_Carbon": unattributed,
        "Unattributed_Carbon_Share": unattributed_share,
        "Attribution_Coverage": coverage,
        "Attribution_Status": status,
    }
    return attributed, coverage_arrays, zero_rows


def _build_skill_sector_frame(
    perspective: str,
    region: str,
    skills: Sequence[str],
    sector_labels: Sequence[str],
    sector_codes: Sequence[str],
    carbon_attr: np.ndarray,
    employment: np.ndarray,
    hours: np.ndarray,
    wage: np.ndarray,
    coverage_arrays: Mapping[str, np.ndarray],
    producer_regions: Optional[Sequence[str]] = None,
    consumer_region: Optional[str] = None,
) -> pd.DataFrame:
    """展开 region-sector-skill 的基础账户表。"""
    rows: List[Dict[str, Any]] = []
    for skill_idx, skill in enumerate(skills):
        for sector_idx, sector in enumerate(sector_labels):
            row = {
                "Accounting_Perspective": perspective,
                "Region": region,
                "Sector": sector,
                "Sector_Code": sector_codes[sector_idx],
                "Skill": skill,
                "Carbon_Attributed": float(carbon_attr[skill_idx, sector_idx]),
                "Employment": float(employment[skill_idx, sector_idx]),
                "Hours": float(hours[skill_idx, sector_idx]),
                "Wage": float(wage[skill_idx, sector_idx]),
            }
            for column in COVERAGE_COLUMNS:
                value = coverage_arrays[column][sector_idx]
                row[column] = str(value) if column == "Attribution_Status" else float(value)
            if producer_regions is not None:
                row["Producer_Region"] = str(producer_regions[sector_idx])
            if consumer_region is not None:
                row["Consumer_Region"] = str(consumer_region)
            rows.append(row)
    return pd.DataFrame(rows)


def _production_tables(inputs: MisalignmentInputs) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float], List[Dict[str, Any]]]:
    """构建生产侧 country-sector-skill 表和参考守恒总量。"""
    carbon = inputs.qC * inputs.x
    employment = inputs.qE * inputs.x.reshape(1, -1)
    hours = inputs.qH * inputs.x.reshape(1, -1)
    wage = inputs.qW * inputs.x.reshape(1, -1)
    carbon_attr, coverage_arrays, audit_rows = _attribute_carbon_by_employment(
        carbon=carbon,
        employment_by_skill=employment,
        perspective="Production",
        region_label="Production",
        sector_labels=inputs.sector_labels,
        sector_codes=inputs.sector_codes or inputs.sector_labels,
    )
    sector_frame = _build_skill_sector_frame(
        perspective="Production",
        region="Production",
        skills=inputs.skills,
        sector_labels=inputs.sector_labels,
        sector_codes=inputs.sector_codes or inputs.sector_labels,
        carbon_attr=carbon_attr,
        employment=employment,
        hours=hours,
        wage=wage,
        coverage_arrays=coverage_arrays,
        producer_regions=inputs.producer_regions,
    )
    sector_frame["Region"] = sector_frame["Producer_Region"]
    carbon_by_region = pd.Series(carbon, index=pd.Index(inputs.producer_regions, name="Region")).groupby(level=0).sum()
    metric_totals = {
        "Carbon": float(carbon.sum()),
        "Attributed_Carbon": float(coverage_arrays["Attributed_Carbon"].sum()),
        "Unattributed_Carbon": float(coverage_arrays["Unattributed_Carbon"].sum()),
        "Employment": float(employment.sum()),
        "Hours": float(hours.sum()),
        "Wage": float(wage.sum()),
    }
    return sector_frame, carbon_by_region, metric_totals, audit_rows


def _consumption_tables(inputs: MisalignmentInputs) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float], List[Dict[str, Any]]]:
    """构建消费侧 destination country-sector-skill 表和参考守恒总量。"""
    output_response = inputs.L @ inputs.Y
    fd_regions = pd.Index(inputs.final_demand_regions, name="Region")
    sector_frames: List[pd.DataFrame] = []
    audit_rows: List[Dict[str, Any]] = []
    carbon_by_region: MutableMapping[str, float] = {}
    metric_totals = {
        "Carbon": 0.0,
        "Attributed_Carbon": 0.0,
        "Unattributed_Carbon": 0.0,
        "Employment": 0.0,
        "Hours": 0.0,
        "Wage": 0.0,
    }

    for destination in fd_regions.unique():
        columns = np.where(fd_regions.to_numpy(dtype=object) == destination)[0]
        induced_output = output_response[:, columns].sum(axis=1)
        carbon = inputs.qC * induced_output
        employment = inputs.qE * induced_output.reshape(1, -1)
        hours = inputs.qH * induced_output.reshape(1, -1)
        wage = inputs.qW * induced_output.reshape(1, -1)
        carbon_attr, coverage_arrays, zero_rows = _attribute_carbon_by_employment(
            carbon=carbon,
            employment_by_skill=employment,
            perspective="Consumption",
            region_label=str(destination),
            sector_labels=inputs.sector_labels,
            sector_codes=inputs.sector_codes or inputs.sector_labels,
        )
        audit_rows.extend(zero_rows)
        sector_frames.append(
            _build_skill_sector_frame(
                perspective="Consumption",
                region=str(destination),
                skills=inputs.skills,
                sector_labels=inputs.sector_labels,
                sector_codes=inputs.sector_codes or inputs.sector_labels,
                carbon_attr=carbon_attr,
                employment=employment,
                hours=hours,
                wage=wage,
                coverage_arrays=coverage_arrays,
                producer_regions=inputs.producer_regions,
                consumer_region=str(destination),
            )
        )
        carbon_by_region[str(destination)] = float(carbon.sum())
        metric_totals["Carbon"] += float(carbon.sum())
        metric_totals["Attributed_Carbon"] += float(coverage_arrays["Attributed_Carbon"].sum())
        metric_totals["Unattributed_Carbon"] += float(coverage_arrays["Unattributed_Carbon"].sum())
        metric_totals["Employment"] += float(employment.sum())
        metric_totals["Hours"] += float(hours.sum())
        metric_totals["Wage"] += float(wage.sum())

    sector_frame = pd.concat(sector_frames, ignore_index=True) if sector_frames else pd.DataFrame()
    carbon_series = pd.Series(carbon_by_region, name="Carbon")
    return sector_frame, carbon_series, metric_totals, audit_rows


def _attribution_status(total: float, unattributed: float) -> str:
    """根据 parent account coverage 生成归因状态。"""
    if total <= 0.0:
        return "not_applicable"
    if unattributed > 0.0:
        return "partially_unattributed"
    return "fully_attributed"


def _country_skill_from_sector(sector_frame: pd.DataFrame, region_col: str) -> pd.DataFrame:
    """从 sector-skill 表聚合出 country/consumer-skill analytical unit。"""
    group_cols = ["Accounting_Perspective", region_col, "Skill"]
    numeric_cols = ["Carbon_Attributed", "Employment", "Hours", "Wage"]
    result = sector_frame.groupby(group_cols, as_index=False, observed=False)[numeric_cols].sum()

    parent_cols = ["Accounting_Perspective", region_col, "Sector", "Sector_Code", *COVERAGE_COLUMNS]
    if "Producer_Region" in sector_frame.columns:
        parent_cols.insert(2, "Producer_Region")
    if "Consumer_Region" in sector_frame.columns and "Consumer_Region" not in parent_cols:
        parent_cols.insert(2, "Consumer_Region")
    parent = sector_frame[parent_cols].drop_duplicates()
    coverage = parent.groupby(["Accounting_Perspective", region_col], as_index=False, observed=False)[
        ["Total_Carbon", "Attributed_Carbon", "Unattributed_Carbon"]
    ].sum()
    coverage["Unattributed_Carbon_Share"] = np.divide(
        coverage["Unattributed_Carbon"],
        coverage["Total_Carbon"],
        out=np.full(len(coverage), np.nan, dtype=float),
        where=coverage["Total_Carbon"].to_numpy(dtype=float) > 0.0,
    )
    coverage["Attribution_Coverage"] = np.divide(
        coverage["Attributed_Carbon"],
        coverage["Total_Carbon"],
        out=np.full(len(coverage), np.nan, dtype=float),
        where=coverage["Total_Carbon"].to_numpy(dtype=float) > 0.0,
    )
    coverage["Attribution_Status"] = [
        _attribution_status(total, unattr)
        for total, unattr in zip(coverage["Total_Carbon"], coverage["Unattributed_Carbon"])
    ]
    return result.merge(coverage, on=["Accounting_Perspective", region_col], how="left")


def _aggregate_consumer_sector_for_alias(consumption_sector: pd.DataFrame) -> pd.DataFrame:
    """将 producer-consumer-sector-skill 表聚合为旧版 consumer-sector-skill alias。"""
    group_cols = ["Accounting_Perspective", "Consumer_Region", "Sector", "Sector_Code", "Skill"]
    account_cols = ["Carbon_Attributed", "Employment", "Hours", "Wage"]
    result = consumption_sector.groupby(group_cols, as_index=False, observed=False)[account_cols].sum()
    parent = consumption_sector[
        [
            "Accounting_Perspective",
            "Consumer_Region",
            "Producer_Region",
            "Sector",
            "Sector_Code",
            *COVERAGE_COLUMNS,
        ]
    ].drop_duplicates()
    coverage = parent.groupby(
        ["Accounting_Perspective", "Consumer_Region", "Sector", "Sector_Code"],
        as_index=False,
        observed=False,
    )[["Total_Carbon", "Attributed_Carbon", "Unattributed_Carbon"]].sum()
    coverage["Unattributed_Carbon_Share"] = np.divide(
        coverage["Unattributed_Carbon"],
        coverage["Total_Carbon"],
        out=np.full(len(coverage), np.nan, dtype=float),
        where=coverage["Total_Carbon"].to_numpy(dtype=float) > 0.0,
    )
    coverage["Attribution_Coverage"] = np.divide(
        coverage["Attributed_Carbon"],
        coverage["Total_Carbon"],
        out=np.full(len(coverage), np.nan, dtype=float),
        where=coverage["Total_Carbon"].to_numpy(dtype=float) > 0.0,
    )
    coverage["Attribution_Status"] = [
        _attribution_status(total, unattr)
        for total, unattr in zip(coverage["Total_Carbon"], coverage["Unattributed_Carbon"])
    ]
    return result.merge(
        coverage,
        on=["Accounting_Perspective", "Consumer_Region", "Sector", "Sector_Code"],
        how="left",
    )


def _ordered(df: pd.DataFrame, columns: Sequence[str], sort_cols: Sequence[str]) -> pd.DataFrame:
    """稳定列序与行序，确保 CSV 可复现。"""
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df.loc[:, columns].sort_values(list(sort_cols)).reset_index(drop=True)


def _audit_row(
    category: str,
    name: str,
    perspective: str,
    unit_level: str,
    observed: float,
    expected: float,
    passed: bool,
    interpretation: str = "",
    details: str = "",
) -> Dict[str, Any]:
    """生成统一 audit 记录。"""
    diff = observed - expected if np.isfinite(observed) and np.isfinite(expected) else np.nan
    rel_diff = diff / expected if np.isfinite(diff) and expected != 0.0 else np.nan
    return {
        "Audit_Category": category,
        "Audit_Name": name,
        "Accounting_Perspective": perspective,
        "Unit_Level": unit_level,
        "Observed": float(observed) if np.isfinite(observed) else np.nan,
        "Expected": float(expected) if np.isfinite(expected) else np.nan,
        "Difference": float(diff) if np.isfinite(diff) else np.nan,
        "Relative_Difference": float(rel_diff) if np.isfinite(rel_diff) else np.nan,
        "Passed": bool(passed),
        "Interpretation": interpretation,
        "Details": details,
    }


def _append_conservation_audit(
    audit_rows: List[Dict[str, Any]],
    df: pd.DataFrame,
    reference_carbon_by_region: pd.Series,
    metric_totals: Mapping[str, float],
    perspective: str,
    unit_level: str,
    rtol: float = 1e-6,
) -> None:
    """检查碳归因覆盖、就业、工时和工资守恒。"""
    region_col = "Region"
    if region_col not in df.columns:
        region_col = "Consumer_Region" if "Consumer_Region" in df.columns else "Producer_Region"

    observed_carbon = df.groupby(region_col, observed=False)["Carbon_Attributed"].sum()
    aligned_reference = reference_carbon_by_region.reindex(observed_carbon.index).fillna(0.0)
    coverage_parent_cols = [region_col, "Total_Carbon", "Attributed_Carbon", "Unattributed_Carbon"]
    if "Producer_Region" in df.columns:
        coverage_parent_cols.insert(0, "Producer_Region")
    if "Consumer_Region" in df.columns and "Consumer_Region" not in coverage_parent_cols:
        coverage_parent_cols.insert(0, "Consumer_Region")
    if "Sector" in df.columns:
        coverage_parent_cols.extend(["Sector", "Sector_Code"])
    parent = df[coverage_parent_cols].drop_duplicates()
    reported_unattributed = parent.groupby(region_col, observed=False)["Unattributed_Carbon"].sum().reindex(
        observed_carbon.index
    ).fillna(0.0)
    coverage_diff = aligned_reference - observed_carbon
    region_error = float((coverage_diff - reported_unattributed).abs().max()) if not observed_carbon.empty else 0.0
    region_scale = float(max(aligned_reference.abs().max(), 1.0))
    audit_rows.append(
        _audit_row(
            "Carbon_Attribution_Coverage",
            "carbon full conservation by region after adding unattributed carbon",
            perspective,
            unit_level,
            region_error,
            0.0,
            bool(region_error <= rtol * region_scale),
            "expected_unattributed_carbon_due_to_zero_skill_employment"
            if float(reported_unattributed.sum()) > 0.0
            else "fully_attributed_or_not_applicable",
        )
    )
    attributed_total = float(df["Carbon_Attributed"].sum())
    parent_total_carbon = float(parent["Total_Carbon"].sum())
    parent_attributed = float(parent["Attributed_Carbon"].sum())
    parent_unattributed = float(parent["Unattributed_Carbon"].sum())
    full_expected = float(metric_totals["Carbon"])
    audit_rows.append(
        _audit_row(
            "Carbon_Attribution_Coverage",
            "attributed plus unattributed carbon equals full carbon",
            perspective,
            unit_level,
            parent_attributed + parent_unattributed,
            full_expected,
            bool(np.isclose(parent_attributed + parent_unattributed, full_expected, rtol=rtol, atol=rtol)),
            "expected_unattributed_carbon_due_to_zero_skill_employment"
            if parent_unattributed > 0.0
            else "fully_attributed_or_not_applicable",
        )
    )
    audit_rows.append(
        _audit_row(
            "Carbon_Attribution_Coverage",
            "reported parent carbon equals full carbon universe",
            perspective,
            unit_level,
            parent_total_carbon,
            full_expected,
            bool(np.isclose(parent_total_carbon, full_expected, rtol=rtol, atol=rtol)),
            "coverage_denominator_check",
        )
    )
    audit_rows.append(
        _audit_row(
            "Carbon_Attribution_Coverage",
            "skill attributed carbon equals attributable carbon universe",
            perspective,
            unit_level,
            attributed_total,
            parent_attributed,
            bool(np.isclose(attributed_total, parent_attributed, rtol=rtol, atol=rtol)),
            "ratio_denominators_use_attributed_carbon_universe",
        )
    )
    for metric, column in [
        ("Employment", "Employment"),
        ("Hours", "Hours"),
        ("Wage", "Wage"),
    ]:
        observed = float(df[column].sum())
        expected = float(metric_totals[metric])
        passed = np.isclose(observed, expected, rtol=rtol, atol=rtol)
        audit_rows.append(
            _audit_row(
                "Account_Conservation",
                f"{metric.lower()} global conservation",
                perspective,
                unit_level,
                observed,
                expected,
                bool(passed),
                "conservation_identity",
            )
        )


def _append_ratio_audit(
    audit_rows: List[Dict[str, Any]],
    df: pd.DataFrame,
    perspective: str,
    unit_level: str,
    rtol: float = 1e-6,
) -> None:
    """检查 ratio 加权均值、log ratio 一致性和无 infinity。"""
    identity_specs = {
        "R_CE": ("Carbon_Share", "Employment_Share"),
        "R_WE": ("Wage_Share", "Employment_Share"),
        "R_CW": ("Carbon_Share", "Wage_Share"),
        "R_CH": ("Carbon_Share", "Hours_Share"),
        "R_WH": ("Wage_Share", "Hours_Share"),
        "R_HE": ("Hours_Share", "Employment_Share"),
    }
    for ratio_col, (numerator_col, weight_col) in identity_specs.items():
        valid = df[ratio_col].notna() & df[weight_col].notna()
        raw_observed = float((df.loc[valid, weight_col] * df.loc[valid, ratio_col]).sum())
        eligible_expected = float(df.loc[valid, numerator_col].sum())
        observed = raw_observed / eligible_expected if eligible_expected > 0.0 else 1.0
        passed = np.isclose(observed, 1.0, rtol=rtol, atol=rtol)
        audit_rows.append(
            _audit_row(
                "Ratio_Identity",
                f"weighted mean identity {ratio_col}",
                perspective,
                unit_level,
                observed,
                1.0,
                bool(passed),
                "eligible_attributed_universe_identity",
            )
        )

    for ratio_col in RATIO_COLUMNS:
        log_col = f"log_{ratio_col}"
        valid = df[ratio_col].notna() & df[log_col].notna() & (df[ratio_col] > 0.0)
        if valid.any():
            abs_diff = (np.exp(df.loc[valid, log_col]) - df.loc[valid, ratio_col]).abs()
            max_diff = float((abs_diff / df.loc[valid, ratio_col].abs().clip(lower=1.0)).max())
        else:
            max_diff = 0.0
        audit_rows.append(
            _audit_row(
                "Ratio_Identity",
                f"log ratio consistency {ratio_col}",
                perspective,
                unit_level,
                max_diff,
                0.0,
                max_diff <= rtol,
                "log_ratio_matches_ratio",
            )
        )

    finite_values = df[list(RATIO_COLUMNS) + list(LOG_RATIO_COLUMNS)].to_numpy(dtype=float)
    inf_count = int(np.isinf(finite_values).sum())
    audit_rows.append(
        _audit_row(
            "Ratio_Validity",
            "no unintentional infinities",
            perspective,
            unit_level,
            float(inf_count),
            0.0,
            inf_count == 0,
            "nan_allowed_for_zero_denominator_but_infinity_forbidden",
        )
    )


def compute_misalignment_package(inputs: MisalignmentInputs) -> Dict[str, Any]:
    """计算生产侧和消费侧 skill-stratified misalignment diagnostics。"""
    checked = validate_inputs(inputs)
    audit_rows: List[Dict[str, Any]] = []

    production_sector, production_carbon_ref, production_totals, production_zero = _production_tables(checked)
    consumption_sector, consumption_carbon_ref, consumption_totals, consumption_zero = _consumption_tables(checked)
    audit_rows.extend(production_zero)
    audit_rows.extend(consumption_zero)

    production_country = _country_skill_from_sector(production_sector, "Region")
    consumption_country = _country_skill_from_sector(consumption_sector, "Consumer_Region")
    consumption_sector_alias = _aggregate_consumer_sector_for_alias(consumption_sector)

    production_country = add_misalignment_ratios(production_country)
    consumption_country = add_misalignment_ratios(consumption_country)
    production_sector = add_misalignment_ratios(production_sector)
    consumption_sector = add_misalignment_ratios(consumption_sector)
    consumption_sector_alias = add_misalignment_ratios(consumption_sector_alias)

    for frame, reference, totals, perspective, unit_level in [
        (production_country, production_carbon_ref, production_totals, "Production", "country-skill"),
        (consumption_country, consumption_carbon_ref, consumption_totals, "Consumption", "country-skill"),
        (production_sector, production_carbon_ref, production_totals, "Production", "country-sector-skill"),
        (consumption_sector, consumption_carbon_ref, consumption_totals, "Consumption", "country-sector-skill"),
    ]:
        _append_conservation_audit(audit_rows, frame, reference, totals, perspective, unit_level)
        _append_ratio_audit(audit_rows, frame, perspective, unit_level)

    package = {
        "producer_skill": _ordered(
            production_country.rename(columns={"Region": "Producer_Region"}),
            [column.replace("Region", "Producer_Region") for column in COUNTRY_SKILL_COLUMNS],
            ["Producer_Region", "Skill"],
        ),
        "consumer_skill": _ordered(
            consumption_country,
            [column.replace("Region", "Consumer_Region") for column in COUNTRY_SKILL_COLUMNS],
            ["Consumer_Region", "Skill"],
        ),
        "producer_sector_skill": _ordered(
            production_sector.drop(columns=["Region"], errors="ignore"),
            [column.replace("Region", "Producer_Region") for column in COUNTRY_SECTOR_SKILL_COLUMNS],
            ["Producer_Region", "Sector", "Sector_Code", "Skill"],
        ),
        "producer_consumer_sector_skill": _ordered(
            consumption_sector,
            [
                "Accounting_Perspective",
                "Producer_Region",
                "Consumer_Region",
                "Sector",
                "Sector_Code",
                "Skill",
                "Carbon_Attributed",
                "Employment",
                "Hours",
                "Wage",
                *COVERAGE_COLUMNS,
                "Carbon_Share",
                "Employment_Share",
                "Hours_Share",
                "Wage_Share",
                *RATIO_COLUMNS,
                *LOG_RATIO_COLUMNS,
                "Quadrant_CE_WE",
                "Valid_For_Ratio",
            ],
            ["Producer_Region", "Consumer_Region", "Sector", "Sector_Code", "Skill"],
        ),
        "country_skill_production": _ordered(production_country, COUNTRY_SKILL_COLUMNS, ["Region", "Skill"]),
        "country_skill_consumption": _ordered(
            consumption_country.rename(columns={"Consumer_Region": "Region"}),
            COUNTRY_SKILL_COLUMNS,
            ["Region", "Skill"],
        ),
        "country_sector_skill_production": _ordered(
            production_sector,
            COUNTRY_SECTOR_SKILL_COLUMNS,
            ["Region", "Sector", "Sector_Code", "Skill"],
        ),
        "country_sector_skill_consumption": _ordered(
            consumption_sector_alias.rename(columns={"Consumer_Region": "Region"}),
            COUNTRY_SECTOR_SKILL_COLUMNS,
            ["Region", "Sector", "Sector_Code", "Skill"],
        ),
        "audit": pd.DataFrame(audit_rows),
        "reference_totals": {
            "Production": {"Carbon_By_Region": production_carbon_ref, **production_totals},
            "Consumption": {"Carbon_By_Region": consumption_carbon_ref, **consumption_totals},
        },
    }
    return package




def build_inputs_from_footprint_engine(engine: Any) -> MisalignmentInputs:
    """从既有 FootprintEngine 提取 qC/qE/qH/qW/x/L/Y 和标签。"""
    qE, skills = engine._build_employment_skill_intensity_matrix()
    qH, _ = engine._build_hours_skill_intensity_matrix()
    qW, _ = engine._build_wage_skill_intensity_matrix()

    y_index = engine.mrio.Y.index
    if isinstance(y_index, pd.MultiIndex):
        producer_regions = y_index.get_level_values(0).astype(str).tolist()
        sector_labels = y_index.get_level_values(1).astype(str).tolist()
    else:
        producer_regions = engine._get_producer_region_labels().astype(str).tolist()
        sector_labels = [str(value) for value in getattr(engine, "products", range(len(producer_regions)))]

    return MisalignmentInputs(
        qC=np.asarray(engine.co2_emissions, dtype=float).reshape(-1),
        qE=np.asarray(qE, dtype=float),
        qH=np.asarray(qH, dtype=float),
        qW=np.asarray(qW, dtype=float),
        x=np.asarray(engine.x, dtype=float).reshape(-1),
        L=np.asarray(engine.L, dtype=float),
        Y=np.asarray(engine.Y, dtype=float),
        producer_regions=producer_regions,
        sector_labels=sector_labels,
        final_demand_regions=engine._get_final_demand_region_labels().astype(str).tolist(),
        skills=list(skills),
        sector_codes=sector_labels,
    )


def load_inputs_from_exiobase(data_path: str) -> MisalignmentInputs:
    """调用既有 FootprintEngine 加载 EXIOBASE，并转换为本脚本输入。"""
    from .footprint import FootprintEngine

    engine = FootprintEngine(data_path)
    if not engine.load_data():
        raise RuntimeError("FootprintEngine failed to load EXIOBASE data")
    return build_inputs_from_footprint_engine(engine)


def _csv_dir_from_output(output_dir: Path) -> Path:
    """允许传入 `.../3.足迹计算` 或其下 `csv` 子目录。"""
    return output_dir if output_dir.name == "csv" else output_dir / "csv"


def write_outputs(package: Mapping[str, Any], output_dir: Path) -> None:
    """写出新增 CSV，不覆盖既有主模型输出。"""
    csv_dir = _csv_dir_from_output(output_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "misalignment_producer_skill.csv": package["producer_skill"],
        "misalignment_consumer_skill.csv": package["consumer_skill"],
        "misalignment_producer_sector_skill.csv": package["producer_sector_skill"],
        "misalignment_producer_consumer_sector_skill.csv": package["producer_consumer_sector_skill"],
        "misalignment_country_skill_production.csv": package["country_skill_production"],
        "misalignment_country_skill_consumption.csv": package["country_skill_consumption"],
        "misalignment_country_sector_skill_production.csv": package["country_sector_skill_production"],
        "misalignment_country_sector_skill_consumption.csv": package["country_sector_skill_consumption"],
        "misalignment_diagnostics_audit.csv": package["audit"],
    }
    for file_name, df in outputs.items():
        df.to_csv(csv_dir / file_name, index=False)
        print(f"保存: {csv_dir / file_name}")
