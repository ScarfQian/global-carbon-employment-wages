"""Country-skill quadrants and mutually exclusive corridor-skill summaries."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


SKILLS = ("Low", "Medium", "High")


TOL = 1e-6


QUADRANT_LABELS = (
    "High-carbon high-return",
    "High-carbon low-return",
    "Low-carbon high-return",
    "Low-carbon low-return",
)


EXCLUSIVE_TYPES = (
    "carbon-heavy burden-over-return",
    "carbon-heavy wage-compensated",
    "low-carbon burden-over-return",
    "low-carbon wage-compensated",
)


def numeric(series: pd.Series) -> pd.Series:
    """将列安全转换为数值列。"""
    return pd.to_numeric(series, errors="coerce")


def safe_divide(numerator: float, denominator: float) -> float:
    """安全除法；分母为零时返回 NaN。"""
    if abs(float(denominator)) <= 1e-30:
        return float("nan")
    return float(numerator) / float(denominator)


def finite_mask(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """返回所有给定列都是有限数的布尔掩码。"""
    mask = pd.Series(True, index=df.index)
    for column in columns:
        mask &= np.isfinite(numeric(df[column]).to_numpy(dtype=float))
    return mask


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    """计算权重均值，用于 share-normalized ratio 的审计和摘要。"""
    values = numeric(df[value_col])
    weights = numeric(df[weight_col])
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def weighted_sum_identity(df: pd.DataFrame, ratio_col: str, share_col: str) -> float:
    """计算 sum(share * ratio)，理论值应为 1。"""
    values = numeric(df[ratio_col])
    shares = numeric(df[share_col])
    mask = np.isfinite(values) & np.isfinite(shares)
    return float((values[mask] * shares[mask]).sum())


def audit_row(
    category: str,
    name: str,
    observed: float | int | str | bool,
    expected: float | int | str | bool,
    passed: bool,
    interpretation: str,
    difference: Optional[float] = None,
) -> Dict[str, object]:
    """构造统一审计行。"""
    if difference is None:
        try:
            difference = float(observed) - float(expected)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            difference = float("nan")
    rel = safe_divide(abs(difference), abs(float(expected))) if isinstance(expected, (int, float)) else float("nan")
    return {
        "Audit_Category": category,
        "Audit_Name": name,
        "Observed": observed,
        "Expected": expected,
        "Difference": difference,
        "Relative_Difference": rel,
        "Passed": bool(passed),
        "Interpretation": interpretation,
    }


def classify_quadrant(r_ce: float, r_we: float) -> str:
    """按 R_CE 与 R_WE 将 country-skill cell 归入四象限。"""
    if not np.isfinite(r_ce) or not np.isfinite(r_we):
        return "Undefined"
    if r_ce > 1.0 and r_we > 1.0:
        return "High-carbon high-return"
    if r_ce > 1.0 and r_we <= 1.0:
        return "High-carbon low-return"
    if r_ce <= 1.0 and r_we > 1.0:
        return "Low-carbon high-return"
    return "Low-carbon low-return"


def prepare_country_skill_table(df: pd.DataFrame, perspective: str) -> pd.DataFrame:
    """统一 producer/consumer country-skill 表的区域列与诊断份额。"""
    work = df.copy()
    region_col = "Producer_Region" if "Producer_Region" in work.columns else "Consumer_Region"
    if region_col not in work.columns:
        if "Region" not in work.columns:
            raise ValueError("Country-skill table must include Producer_Region, Consumer_Region, or Region.")
        region_col = "Region"

    required = ["Skill", "Carbon_Attributed", "Employment", "Hours", "Wage", "R_CE", "R_WE", "R_CW"]
    missing = [column for column in required if column not in work.columns]
    if missing:
        raise ValueError(f"Missing country-skill columns: {missing}")

    valid = work.get("Valid_For_Ratio", True)
    if isinstance(valid, pd.Series):
        valid_mask = valid.astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        valid_mask = pd.Series(bool(valid), index=work.index)
    work = work.loc[valid_mask & finite_mask(work, ["R_CE", "R_WE", "R_CW"])].copy()

    work["Accounting_Perspective"] = perspective
    work["Region"] = work[region_col].astype(str)
    if perspective == "Production":
        work["Producer_Region"] = work["Region"]
        work["Consumer_Region"] = ""
    else:
        work["Consumer_Region"] = work["Region"]
        work["Producer_Region"] = ""
    work["Quadrant"] = [
        classify_quadrant(r_ce, r_we)
        for r_ce, r_we in zip(numeric(work["R_CE"]), numeric(work["R_WE"]))
    ]
    work = work.loc[work["Quadrant"].isin(QUADRANT_LABELS)].copy()

    totals = {
        "Carbon_Attributed": numeric(work["Carbon_Attributed"]).sum(),
        "Employment": numeric(work["Employment"]).sum(),
        "Hours": numeric(work["Hours"]).sum(),
        "Wage": numeric(work["Wage"]).sum(),
    }
    for value_col, share_col in [
        ("Carbon_Attributed", "Carbon_Attributed_Share"),
        ("Employment", "Employment_Share"),
        ("Hours", "Hours_Share"),
        ("Wage", "Wage_Share"),
    ]:
        values = numeric(work[value_col]).to_numpy(dtype=float)
        work[share_col] = np.divide(
            values,
            totals[value_col],
            out=np.full(len(work), np.nan, dtype=float),
            where=abs(totals[value_col]) > 1e-30,
        )
    return work.sort_values(["Accounting_Perspective", "Quadrant", "Region", "Skill"]).reset_index(drop=True)


def top_cell_label(row: pd.Series) -> str:
    """生成紧凑 country-skill cell 标签。"""
    return f"{row['Region']}|{row['Skill']}"


def build_quadrant_summary(work: pd.DataFrame) -> pd.DataFrame:
    """生成四象限规模份额摘要。"""
    rows: List[Dict[str, object]] = []
    total = {
        "Carbon_Attributed": numeric(work["Carbon_Attributed"]).sum(),
        "Employment": numeric(work["Employment"]).sum(),
        "Hours": numeric(work["Hours"]).sum(),
        "Wage": numeric(work["Wage"]).sum(),
    }
    for quadrant in QUADRANT_LABELS:
        group = work.loc[work["Quadrant"].eq(quadrant)].copy()
        if group.empty:
            continue
        top_carbon = group.loc[numeric(group["Carbon_Attributed"]).idxmax()]
        top_employment = group.loc[numeric(group["Employment"]).idxmax()]
        top_rcw = group.loc[numeric(group["R_CW"]).idxmax()]
        rows.append(
            {
                "Accounting_Perspective": str(group["Accounting_Perspective"].iloc[0]),
                "Quadrant": quadrant,
                "N_Cells": int(len(group)),
                "Carbon_Attributed_Total": float(numeric(group["Carbon_Attributed"]).sum()),
                "Employment_Total": float(numeric(group["Employment"]).sum()),
                "Hours_Total": float(numeric(group["Hours"]).sum()),
                "Wage_Total": float(numeric(group["Wage"]).sum()),
                "Carbon_Attributed_Share": safe_divide(numeric(group["Carbon_Attributed"]).sum(), total["Carbon_Attributed"]),
                "Employment_Share": safe_divide(numeric(group["Employment"]).sum(), total["Employment"]),
                "Hours_Share": safe_divide(numeric(group["Hours"]).sum(), total["Hours"]),
                "Wage_Share": safe_divide(numeric(group["Wage"]).sum(), total["Wage"]),
                "Mean_R_CE_EmploymentWeighted": weighted_mean(group, "R_CE", "Employment"),
                "Mean_R_WE_EmploymentWeighted": weighted_mean(group, "R_WE", "Employment"),
                "Mean_R_CW_WageWeighted": weighted_mean(group, "R_CW", "Wage"),
                "Top_Cell_By_Carbon": top_cell_label(top_carbon),
                "Top_Cell_By_Employment": top_cell_label(top_employment),
                "Top_Cell_By_R_CW": top_cell_label(top_rcw),
            }
        )
    columns = [
        "Accounting_Perspective",
        "Quadrant",
        "N_Cells",
        "Carbon_Attributed_Total",
        "Employment_Total",
        "Hours_Total",
        "Wage_Total",
        "Carbon_Attributed_Share",
        "Employment_Share",
        "Hours_Share",
        "Wage_Share",
        "Mean_R_CE_EmploymentWeighted",
        "Mean_R_WE_EmploymentWeighted",
        "Mean_R_CW_WageWeighted",
        "Top_Cell_By_Carbon",
        "Top_Cell_By_Employment",
        "Top_Cell_By_R_CW",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_top_country_skill_cells(work: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """生成每个象限按就业、碳和 R_CW 排名的 top cells。"""
    rows: List[Dict[str, object]] = []
    rank_specs = {
        "by_employment": "Employment_Share",
        "by_carbon": "Carbon_Attributed_Share",
        "by_R_CW": "R_CW",
    }
    for (perspective, quadrant), group in work.groupby(["Accounting_Perspective", "Quadrant"], observed=False):
        for rank_type, sort_col in rank_specs.items():
            ranked = group.sort_values(sort_col, ascending=False).head(top_n).copy()
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "Accounting_Perspective": perspective,
                        "Quadrant": quadrant,
                        "Rank_Type": rank_type,
                        "Rank": rank,
                        "Region": row["Region"],
                        "Producer_Region": row.get("Producer_Region", ""),
                        "Consumer_Region": row.get("Consumer_Region", ""),
                        "Skill": row["Skill"],
                        "Carbon_Attributed": row["Carbon_Attributed"],
                        "Employment": row["Employment"],
                        "Hours": row["Hours"],
                        "Wage": row["Wage"],
                        "Carbon_Attributed_Share": row["Carbon_Attributed_Share"],
                        "Employment_Share": row["Employment_Share"],
                        "Hours_Share": row["Hours_Share"],
                        "Wage_Share": row["Wage_Share"],
                        "R_CE": row["R_CE"],
                        "R_WE": row["R_WE"],
                        "R_CW": row["R_CW"],
                        "R_CH": row.get("R_CH", np.nan),
                        "R_WH": row.get("R_WH", np.nan),
                        "R_HE": row.get("R_HE", np.nan),
                        "log_R_CE": row.get("log_R_CE", np.nan),
                        "log_R_WE": row.get("log_R_WE", np.nan),
                        "log_R_CW": row.get("log_R_CW", np.nan),
                    }
                )
    columns = [
        "Accounting_Perspective",
        "Quadrant",
        "Rank_Type",
        "Rank",
        "Region",
        "Producer_Region",
        "Consumer_Region",
        "Skill",
        "Carbon_Attributed",
        "Employment",
        "Hours",
        "Wage",
        "Carbon_Attributed_Share",
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
    ]
    return pd.DataFrame(rows, columns=columns)


def build_quadrant_audit(production: pd.DataFrame, consumption: pd.DataFrame) -> pd.DataFrame:
    """生成 country-skill 四象限摘要的审计表。"""
    rows: List[Dict[str, object]] = []
    for perspective, work in [("Production", production), ("Consumption", consumption)]:
        for share_col in ["Carbon_Attributed_Share", "Employment_Share", "Hours_Share", "Wage_Share"]:
            observed = float(numeric(work[share_col]).sum())
            rows.append(
                audit_row(
                    "quadrant_summary",
                    f"{perspective}_{share_col}_sums_to_one",
                    observed,
                    1.0,
                    abs(observed - 1.0) <= TOL,
                    "quadrant shares partition the valid country-skill diagnostic universe",
                )
            )
        for ratio_col, share_col in [("R_CE", "Employment_Share"), ("R_WE", "Employment_Share"), ("R_CW", "Wage_Share")]:
            observed = weighted_sum_identity(work, ratio_col, share_col)
            rows.append(
                audit_row(
                    "quadrant_ratio_identity",
                    f"{perspective}_{share_col}_weighted_{ratio_col}",
                    observed,
                    1.0,
                    abs(observed - 1.0) <= TOL,
                    "share-normalized ratio identity over valid country-skill cells",
                )
            )
        observed_labels = int(work["Quadrant"].nunique())
        rows.append(
            audit_row(
                "quadrant_labels",
                f"{perspective}_exclusive_quadrant_labels",
                observed_labels,
                min(4, observed_labels),
                set(work["Quadrant"].unique()).issubset(set(QUADRANT_LABELS)) and observed_labels <= 4,
                "exactly four possible quadrant labels; absent labels are allowed only for zero-cell quadrants",
                0.0 if observed_labels <= 4 else observed_labels - 4,
            )
        )
    return pd.DataFrame(rows)


def prepare_corridor_table(df: pd.DataFrame) -> pd.DataFrame:
    """统一 corridor-skill 表，创建互斥类型与 scale-weighted scores。"""
    required = ["From_Region", "To_Region", "Skill", "Carbon_Transfer", "Employment_Transfer", "Hours_Transfer", "Wage_Transfer", "R_CE", "R_CW", "R_CH", "R_WH"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing corridor columns: {missing}")

    work = df.copy()
    valid = work.get("Valid_For_Ratio", True)
    if isinstance(valid, pd.Series):
        valid_mask = valid.astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        valid_mask = pd.Series(bool(valid), index=work.index)
    work = work.loc[
        valid_mask
        & work["From_Region"].astype(str).ne(work["To_Region"].astype(str))
        & finite_mask(work, ["R_CE", "R_CW", "R_CH", "R_WH"])
    ].copy()

    totals = {
        "Carbon_Transfer": numeric(work["Carbon_Transfer"]).sum(),
        "Employment_Transfer": numeric(work["Employment_Transfer"]).sum(),
        "Hours_Transfer": numeric(work["Hours_Transfer"]).sum(),
        "Wage_Transfer": numeric(work["Wage_Transfer"]).sum(),
    }
    for value_col, share_col in [
        ("Carbon_Transfer", "Carbon_Share"),
        ("Employment_Transfer", "Employment_Share"),
        ("Hours_Transfer", "Hours_Share"),
        ("Wage_Transfer", "Wage_Share"),
    ]:
        values = numeric(work[value_col]).to_numpy(dtype=float)
        work[share_col] = np.divide(
            values,
            totals[value_col],
            out=np.full(len(work), np.nan, dtype=float),
            where=abs(totals[value_col]) > 1e-30,
        )

    work["log_R_CW"] = np.log(numeric(work["R_CW"]))
    work["log_R_CE"] = np.log(numeric(work["R_CE"]))
    work["ScaleWeighted_R_CW"] = work["log_R_CW"] * work["Carbon_Share"]
    work["ScaleWeighted_R_CE"] = work["log_R_CE"] * work["Carbon_Share"]
    work["ScaleWeighted_Burden_Return_Score"] = np.maximum(work["log_R_CW"], 0.0) * work["Carbon_Share"]
    work["ScaleWeighted_Carbon_Employment_Score"] = np.maximum(work["log_R_CE"], 0.0) * work["Carbon_Share"]

    work["Is_Carbon_Heavy"] = numeric(work["R_CE"]) > 1.0
    work["Is_Burden_Over_Return"] = numeric(work["R_CW"]) > 1.0
    work["Is_Hour_Low_Return"] = numeric(work["R_WH"]) < 1.0
    work["Is_Wage_Capturing"] = numeric(work["R_CW"]) < 1.0

    conditions = [
        work["Is_Carbon_Heavy"] & work["Is_Burden_Over_Return"],
        work["Is_Carbon_Heavy"] & ~work["Is_Burden_Over_Return"],
        ~work["Is_Carbon_Heavy"] & work["Is_Burden_Over_Return"],
        ~work["Is_Carbon_Heavy"] & ~work["Is_Burden_Over_Return"],
    ]
    work["Exclusive_Type"] = np.select(conditions, EXCLUSIVE_TYPES, default="undefined")
    return work.sort_values(["Exclusive_Type", "From_Region", "To_Region", "Skill"]).reset_index(drop=True)


def scale_weighted_corridor_tables(work: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank carbon-wage mismatch using full-network carbon shares."""
    selected = work.loc[work["R_CW"] > 1.0].copy()
    selected["Scale_Weighted_Burden_Return_Score"] = selected["ScaleWeighted_Burden_Return_Score"]
    selected = selected.sort_values(
        ["Scale_Weighted_Burden_Return_Score", "From_Region", "To_Region", "Skill"],
        ascending=[False, True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    transfer_columns = ["Carbon_Transfer", "Employment_Transfer", "Hours_Transfer", "Wage_Transfer"]
    share_columns = ["Carbon_Share", "Employment_Share", "Hours_Share", "Wage_Share"]
    pairs = selected.groupby(["From_Region", "To_Region"], sort=True)[
        transfer_columns + share_columns + ["Scale_Weighted_Burden_Return_Score"]
    ].sum().reset_index()
    pairs = pairs.rename(columns={column: column + "_Total" for column in transfer_columns})
    group = selected.groupby(["From_Region", "To_Region"], sort=True)
    counts = group.size().rename("N_Skill_Cells")
    skills = group["Skill"].agg(";".join).rename("Top_Skills")
    pairs = pairs.merge(counts, on=["From_Region", "To_Region"]).merge(skills, on=["From_Region", "To_Region"])
    pairs = pairs.sort_values(
        ["Scale_Weighted_Burden_Return_Score", "From_Region", "To_Region"],
        ascending=[False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    return selected, pairs


def corridor_label(row: pd.Series) -> str:
    """生成紧凑 corridor-skill 标签。"""
    return f"{row['From_Region']}->{row['To_Region']}|{row['Skill']}"


def build_corridor_exclusive_summary(work: pd.DataFrame) -> pd.DataFrame:
    """生成互斥 corridor type 的规模摘要。"""
    rows: List[Dict[str, object]] = []
    total = {
        "Carbon_Transfer": numeric(work["Carbon_Transfer"]).sum(),
        "Employment_Transfer": numeric(work["Employment_Transfer"]).sum(),
        "Hours_Transfer": numeric(work["Hours_Transfer"]).sum(),
        "Wage_Transfer": numeric(work["Wage_Transfer"]).sum(),
    }
    for exclusive_type in EXCLUSIVE_TYPES:
        group = work.loc[work["Exclusive_Type"].eq(exclusive_type)].copy()
        if group.empty:
            continue
        top_carbon = group.loc[numeric(group["Carbon_Transfer"]).idxmax()]
        top_score = group.loc[numeric(group["ScaleWeighted_Burden_Return_Score"]).idxmax()]
        rows.append(
            {
                "Exclusive_Type": exclusive_type,
                "N_Corridor_Skill_Cells": int(len(group)),
                "Carbon_Transfer_Total": float(numeric(group["Carbon_Transfer"]).sum()),
                "Employment_Transfer_Total": float(numeric(group["Employment_Transfer"]).sum()),
                "Hours_Transfer_Total": float(numeric(group["Hours_Transfer"]).sum()),
                "Wage_Transfer_Total": float(numeric(group["Wage_Transfer"]).sum()),
                "Carbon_Share": safe_divide(numeric(group["Carbon_Transfer"]).sum(), total["Carbon_Transfer"]),
                "Employment_Share": safe_divide(numeric(group["Employment_Transfer"]).sum(), total["Employment_Transfer"]),
                "Hours_Share": safe_divide(numeric(group["Hours_Transfer"]).sum(), total["Hours_Transfer"]),
                "Wage_Share": safe_divide(numeric(group["Wage_Transfer"]).sum(), total["Wage_Transfer"]),
                "Mean_R_CE_EmploymentWeighted": weighted_mean(group, "R_CE", "Employment_Transfer"),
                "Mean_R_CW_WageWeighted": weighted_mean(group, "R_CW", "Wage_Transfer"),
                "Mean_R_CH_HoursWeighted": weighted_mean(group, "R_CH", "Hours_Transfer"),
                "Mean_R_WH_HoursWeighted": weighted_mean(group, "R_WH", "Hours_Transfer"),
                "Top_Corridor_By_Carbon": corridor_label(top_carbon),
                "Top_Corridor_By_ScaleWeighted_R_CW": corridor_label(top_score),
            }
        )
    return pd.DataFrame(rows)


def build_top_corridors_by_exclusive_type(work: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """生成每类互斥 corridor type 的 top corridor-skill 单元。"""
    rows: List[Dict[str, object]] = []
    for exclusive_type, group in work.groupby("Exclusive_Type", observed=False):
        for rank_type, sort_col in [
            ("by_carbon", "Carbon_Transfer"),
            ("by_scaleweighted_R_CW", "ScaleWeighted_Burden_Return_Score"),
        ]:
            ranked = group.sort_values(sort_col, ascending=False).head(top_n)
            for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
                rows.append(
                    {
                        "Exclusive_Type": exclusive_type,
                        "Rank_Type": rank_type,
                        "Rank": rank,
                        "From_Region": row["From_Region"],
                        "To_Region": row["To_Region"],
                        "Skill": row["Skill"],
                        "Carbon_Transfer": row["Carbon_Transfer"],
                        "Employment_Transfer": row["Employment_Transfer"],
                        "Hours_Transfer": row["Hours_Transfer"],
                        "Wage_Transfer": row["Wage_Transfer"],
                        "Carbon_Share": row["Carbon_Share"],
                        "Employment_Share": row["Employment_Share"],
                        "Hours_Share": row["Hours_Share"],
                        "Wage_Share": row["Wage_Share"],
                        "R_CE": row["R_CE"],
                        "R_CW": row["R_CW"],
                        "R_CH": row["R_CH"],
                        "R_WH": row["R_WH"],
                        "ScaleWeighted_R_CW": row["ScaleWeighted_R_CW"],
                        "ScaleWeighted_R_CE": row["ScaleWeighted_R_CE"],
                        "ScaleWeighted_Burden_Return_Score": row["ScaleWeighted_Burden_Return_Score"],
                    }
                )
    columns = [
        "Exclusive_Type",
        "Rank_Type",
        "Rank",
        "From_Region",
        "To_Region",
        "Skill",
        "Carbon_Transfer",
        "Employment_Transfer",
        "Hours_Transfer",
        "Wage_Transfer",
        "Carbon_Share",
        "Employment_Share",
        "Hours_Share",
        "Wage_Share",
        "R_CE",
        "R_CW",
        "R_CH",
        "R_WH",
        "ScaleWeighted_R_CW",
        "ScaleWeighted_R_CE",
        "ScaleWeighted_Burden_Return_Score",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_corridor_group_summary(work: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """按技能或 region-pair 汇总互斥 corridor types。"""
    total = {
        "Carbon_Transfer": numeric(work["Carbon_Transfer"]).sum(),
        "Employment_Transfer": numeric(work["Employment_Transfer"]).sum(),
        "Hours_Transfer": numeric(work["Hours_Transfer"]).sum(),
        "Wage_Transfer": numeric(work["Wage_Transfer"]).sum(),
    }
    rows: List[Dict[str, object]] = []
    for keys, group in work.groupby(group_cols, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "N_Corridor_Skill_Cells": int(len(group)),
                "Carbon_Transfer_Total": float(numeric(group["Carbon_Transfer"]).sum()),
                "Employment_Transfer_Total": float(numeric(group["Employment_Transfer"]).sum()),
                "Hours_Transfer_Total": float(numeric(group["Hours_Transfer"]).sum()),
                "Wage_Transfer_Total": float(numeric(group["Wage_Transfer"]).sum()),
                "Carbon_Share": safe_divide(numeric(group["Carbon_Transfer"]).sum(), total["Carbon_Transfer"]),
                "Employment_Share": safe_divide(numeric(group["Employment_Transfer"]).sum(), total["Employment_Transfer"]),
                "Hours_Share": safe_divide(numeric(group["Hours_Transfer"]).sum(), total["Hours_Transfer"]),
                "Wage_Share": safe_divide(numeric(group["Wage_Transfer"]).sum(), total["Wage_Transfer"]),
                "Mean_R_CE_EmploymentWeighted": weighted_mean(group, "R_CE", "Employment_Transfer"),
                "Mean_R_CW_WageWeighted": weighted_mean(group, "R_CW", "Wage_Transfer"),
                "Mean_R_CH_HoursWeighted": weighted_mean(group, "R_CH", "Hours_Transfer"),
                "Mean_R_WH_HoursWeighted": weighted_mean(group, "R_WH", "Hours_Transfer"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_corridor_exclusive_audit(work: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """生成 corridor exclusive type 审计。"""
    rows: List[Dict[str, object]] = []
    valid_raw = raw.loc[raw["From_Region"].astype(str).ne(raw["To_Region"].astype(str))].copy()
    if "Valid_For_Ratio" in valid_raw.columns:
        valid_raw = valid_raw.loc[valid_raw["Valid_For_Ratio"].astype(str).str.lower().isin(["true", "1", "yes"])]
    valid_raw = valid_raw.loc[finite_mask(valid_raw, ["R_CE", "R_CW", "R_CH", "R_WH"])]

    rows.append(
        audit_row(
            "exclusive_corridor_types",
            "valid_corridor_skill_cells_partitioned_once",
            int(len(work)),
            int(len(valid_raw)),
            len(work) == len(valid_raw) and work["Exclusive_Type"].isin(EXCLUSIVE_TYPES).all(),
            "exclusive types partition all valid cross-border corridor-skill cells exactly once",
        )
    )
    for share_col in ["Carbon_Share", "Employment_Share", "Hours_Share", "Wage_Share"]:
        observed = float(numeric(work[share_col]).sum())
        rows.append(
            audit_row(
                "exclusive_corridor_shares",
                f"{share_col}_sums_to_one",
                observed,
                1.0,
                abs(observed - 1.0) <= TOL,
                "exclusive corridor types use the same valid cross-border diagnostic universe",
            )
        )
    domestic = int(work["From_Region"].astype(str).eq(work["To_Region"].astype(str)).sum())
    rows.append(
        audit_row(
            "exclusive_corridor_scope",
            "domestic_corridors_excluded",
            domestic,
            0,
            domestic == 0,
            "default corridor-skill diagnostics exclude domestic rows",
        )
    )
    for ratio_col, share_col in [("R_CE", "Employment_Share"), ("R_CW", "Wage_Share"), ("R_CH", "Hours_Share"), ("R_WH", "Hours_Share")]:
        observed = weighted_sum_identity(work, ratio_col, share_col)
        rows.append(
            audit_row(
                "exclusive_corridor_ratio_identity",
                f"{share_col}_weighted_{ratio_col}",
                observed,
                1.0,
                abs(observed - 1.0) <= TOL,
                "share-normalized corridor ratio identity over valid cross-border cells",
            )
        )
    return pd.DataFrame(rows)
