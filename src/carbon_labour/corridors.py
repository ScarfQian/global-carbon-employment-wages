"""Share-based corridor-skill misalignment diagnostics."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


METRIC_VALUE_COLUMNS: Dict[str, str] = {
    "carbon": "Carbon_Transfer",
    "employment": "Employment_Transfer",
    "wage": "Wage_Transfer",
    "hours": "Hours_Transfer",
}
HIGH_SCALE_CARBON_SHARE_THRESHOLD = 0.001
HIGH_SCALE_QUANTILE = 0.90

OUTPUT_COLUMNS: List[str] = [
    "From_Region",
    "To_Region",
    "Skill",
    "Carbon_Transfer",
    "Employment_Transfer",
    "Wage_Transfer",
    "Hours_Transfer",
    "Carbon_Share",
    "Employment_Share",
    "Wage_Share",
    "Hours_Share",
    "R_CE",
    "R_CW",
    "R_CH",
    "R_WH",
    "log_R_CE",
    "log_R_CW",
    "log_R_CH",
    "log_R_WH",
    "Weighted_Log_R_CW",
    "Weighted_Log_R_CE",
    "Weighted_Burden_Return_Score",
    "Weighted_Carbon_Employment_Score",
    "Carbon_Heavy_Corridor",
    "Burden_Over_Return_Corridor",
    "Wage_Capturing_Corridor",
    "Hour_Carbon_Exposed_Corridor",
    "Hour_Low_Return_Corridor",
    "High_Scale_Corridor",
    "Corridor_Type",
    "Combined_Corridor_Type",
    "High_Scale_Carbon_Share_Threshold",
    "High_Scale_Carbon_Transfer_Quantile",
    "Corridor_Flags",
    "Valid_For_Ratio",
]


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """按大小写无关方式寻找列名。"""
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_bilateral_transfer(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """把不同命名的 bilateral transfer 表标准化为 corridor-skill 表。"""
    if metric not in METRIC_VALUE_COLUMNS:
        raise ValueError(f"Unsupported metric: {metric}")
    value_col = METRIC_VALUE_COLUMNS[metric]
    from_col = _resolve_column(df, ["From_Region", "from_region", "Producer_Country", "producer_country"])
    to_col = _resolve_column(df, ["To_Region", "to_region", "Demand_Country", "demand_country"])
    skill_col = _resolve_column(df, ["Skill", "skill"])
    raw_value_col = _resolve_column(df, [value_col, "Transfer_Value", "transfer_value", metric])
    missing = [
        name
        for name, column in [("from", from_col), ("to", to_col), ("skill", skill_col), ("value", raw_value_col)]
        if column is None
    ]
    if missing:
        raise ValueError(f"Missing columns for {metric}: {missing}")

    result = pd.DataFrame(
        {
            "From_Region": df[from_col].astype(str),
            "To_Region": df[to_col].astype(str),
            "Skill": df[skill_col].astype(str),
            value_col: pd.to_numeric(df[raw_value_col], errors="coerce"),
        }
    )
    if result[value_col].isna().any():
        raise ValueError(f"{metric} transfer contains non-numeric values")
    result = result.groupby(["From_Region", "To_Region", "Skill"], as_index=False, observed=False)[value_col].sum()
    return result


def _safe_share(series: pd.Series, total: float) -> pd.Series:
    """计算 transfer share，总量为 0 时返回 NaN。"""
    if total > 0.0:
        return series.astype(float) / total
    return pd.Series(np.nan, index=series.index, dtype=float)


def _share_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """计算 share-normalized ratio；分母份额为 0 时返回 NaN。"""
    num = numerator.to_numpy(dtype=float)
    den = denominator.to_numpy(dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=out, where=np.isfinite(den) & (den > 0.0))
    return pd.Series(out, index=numerator.index, dtype=float)


def _log_ratio(values: pd.Series) -> pd.Series:
    """对正 ratio 取 log；0 或 NaN 保持 NaN，避免 infinity。"""
    array = values.to_numpy(dtype=float)
    out = np.full_like(array, np.nan, dtype=float)
    valid = np.isfinite(array) & (array > 0.0)
    out[valid] = np.log(array[valid])
    return pd.Series(out, index=values.index, dtype=float)


def _audit_row(check: str, observed: float, expected: float, passed: bool, details: str = "") -> Dict[str, Any]:
    """生成统一 audit 行。"""
    abs_diff = abs(observed - expected) if np.isfinite(observed) and np.isfinite(expected) else np.nan
    rel_diff = abs_diff / abs(expected) if np.isfinite(abs_diff) and expected != 0.0 else np.nan
    return {
        "Check": check,
        "Observed": float(observed) if np.isfinite(observed) else np.nan,
        "Expected": float(expected) if np.isfinite(expected) else np.nan,
        "Abs_Diff": float(abs_diff) if np.isfinite(abs_diff) else np.nan,
        "Rel_Diff": float(rel_diff) if np.isfinite(rel_diff) else np.nan,
        "Passed": bool(passed),
        "Details": details,
    }


def compute_corridor_skill_misalignment(frames: Mapping[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """合并四类 by-skill corridor transfer，并计算 corridor-skill ratios。"""
    audit_rows: List[Dict[str, Any]] = []
    normalized: Dict[str, pd.DataFrame] = {}
    input_totals: Dict[str, float] = {}
    for metric in ("carbon", "employment", "wage", "hours"):
        if metric not in frames:
            raise ValueError(f"Missing input frame: {metric}")
        table = normalize_bilateral_transfer(frames[metric], metric)
        value_col = METRIC_VALUE_COLUMNS[metric]
        before_total = float(pd.to_numeric(frames[metric][value_col], errors="coerce").sum()) if value_col in frames[metric] else float(table[value_col].sum())
        normalized[metric] = table
        input_totals[metric] = before_total

    merged = normalized["carbon"]
    for metric in ("employment", "wage", "hours"):
        merged = merged.merge(normalized[metric], on=["From_Region", "To_Region", "Skill"], how="outer")
    for value_col in METRIC_VALUE_COLUMNS.values():
        merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce").fillna(0.0)
    merged = merged.loc[merged["From_Region"] != merged["To_Region"]].copy()

    for metric, value_col in METRIC_VALUE_COLUMNS.items():
        observed = float(merged[value_col].sum())
        expected = float(normalized[metric].loc[normalized[metric]["From_Region"] != normalized[metric]["To_Region"], value_col].sum())
        audit_rows.append(
            _audit_row(
                f"{metric} total transfer preserved after merge",
                observed,
                expected,
                bool(np.isclose(observed, expected, rtol=1e-6, atol=1e-6)),
                details=f"raw_input_total={input_totals[metric]}",
            )
        )

    duplicate_count = int(merged.duplicated(["From_Region", "To_Region", "Skill"]).sum())
    audit_rows.append(
        _audit_row(
            "no duplicate corridor-skill rows after aggregation",
            float(duplicate_count),
            0.0,
            duplicate_count == 0,
        )
    )

    totals = {column: float(merged[column].sum()) for column in METRIC_VALUE_COLUMNS.values()}
    merged["Carbon_Share"] = _safe_share(merged["Carbon_Transfer"], totals["Carbon_Transfer"])
    merged["Employment_Share"] = _safe_share(merged["Employment_Transfer"], totals["Employment_Transfer"])
    merged["Wage_Share"] = _safe_share(merged["Wage_Transfer"], totals["Wage_Transfer"])
    merged["Hours_Share"] = _safe_share(merged["Hours_Transfer"], totals["Hours_Transfer"])

    merged["R_CE"] = _share_ratio(merged["Carbon_Share"], merged["Employment_Share"])
    merged["R_CW"] = _share_ratio(merged["Carbon_Share"], merged["Wage_Share"])
    merged["R_CH"] = _share_ratio(merged["Carbon_Share"], merged["Hours_Share"])
    merged["R_WH"] = _share_ratio(merged["Wage_Share"], merged["Hours_Share"])
    for ratio_col in ("R_CE", "R_CW", "R_CH", "R_WH"):
        merged[f"log_{ratio_col}"] = _log_ratio(merged[ratio_col])

    merged["Weighted_Log_R_CW"] = merged["log_R_CW"] * merged["Carbon_Share"]
    merged["Weighted_Log_R_CE"] = merged["log_R_CE"] * merged["Carbon_Share"]
    merged["Weighted_Burden_Return_Score"] = np.maximum(merged["log_R_CW"], 0.0) * merged["Carbon_Share"]
    merged["Weighted_Carbon_Employment_Score"] = np.maximum(merged["log_R_CE"], 0.0) * merged["Carbon_Share"]
    merged["Carbon_Heavy_Corridor"] = merged["R_CE"] > 1.0
    merged["Burden_Over_Return_Corridor"] = merged["R_CW"] > 1.0
    merged["Wage_Capturing_Corridor"] = merged["R_CW"] < 1.0
    merged["Hour_Carbon_Exposed_Corridor"] = merged["R_CH"] > 1.0
    merged["Hour_Low_Return_Corridor"] = merged["R_WH"] < 1.0
    carbon_transfer_threshold = float(merged["Carbon_Transfer"].quantile(HIGH_SCALE_QUANTILE)) if not merged.empty else np.nan
    merged["High_Scale_Corridor"] = (merged["Carbon_Transfer"] >= carbon_transfer_threshold) | (
        merged["Carbon_Share"] >= HIGH_SCALE_CARBON_SHARE_THRESHOLD
    )
    merged["High_Scale_Carbon_Share_Threshold"] = HIGH_SCALE_CARBON_SHARE_THRESHOLD
    merged["High_Scale_Carbon_Transfer_Quantile"] = HIGH_SCALE_QUANTILE
    merged["Corridor_Type"] = [
        "; ".join(
            label
            for condition, label in [
                (row["R_CW"] > 1.0, "Burden_over_return"),
                (row["R_CW"] < 1.0, "Wage_capturing"),
                (row["R_CE"] > 1.0, "Carbon_heavy"),
                (row["R_CE"] < 1.0, "Employment_heavy"),
                (row["R_WH"] < 1.0, "Hour_low_return"),
            ]
            if bool(condition)
        )
        for _, row in merged.iterrows()
    ]
    merged["Combined_Corridor_Type"] = [
        "; ".join(
            label
            for condition, label in [
                (
                    row["R_CE"] > 1.0 and row["R_CW"] > 1.0,
                    "Carbon_heavy_and_burden_over_return",
                ),
                (
                    row["R_CW"] > 1.0 and row["High_Scale_Corridor"],
                    "High_scale_burden_over_return",
                ),
                (
                    row["R_CW"] > 1.0 and not row["High_Scale_Corridor"],
                    "High_ratio_burden_over_return",
                ),
            ]
            if bool(condition)
        )
        for _, row in merged.iterrows()
    ]
    flag_specs = [
        ("Carbon_Heavy_Corridor", "Carbon-heavy corridor"),
        ("Burden_Over_Return_Corridor", "Burden-over-return corridor"),
        ("Wage_Capturing_Corridor", "Wage-capturing corridor"),
        ("Hour_Carbon_Exposed_Corridor", "Hour-carbon exposed corridor"),
        ("Hour_Low_Return_Corridor", "Hour-low-return corridor"),
    ]
    merged["Corridor_Flags"] = [
        "; ".join(label for column, label in flag_specs if bool(row[column]))
        for _, row in merged.iterrows()
    ]
    merged["Valid_For_Ratio"] = (
        (merged["Employment_Share"] > 0.0)
        & (merged["Wage_Share"] > 0.0)
        & (merged["Hours_Share"] > 0.0)
        & (totals["Carbon_Transfer"] > 0.0)
    )

    identity_specs = {
        "R_CE": "Employment_Share",
        "R_CW": "Wage_Share",
        "R_CH": "Hours_Share",
        "R_WH": "Hours_Share",
    }
    for ratio_col, weight_col in identity_specs.items():
        valid = merged[ratio_col].notna() & merged[weight_col].notna()
        observed = float((merged.loc[valid, weight_col] * merged.loc[valid, ratio_col]).sum())
        audit_rows.append(
            _audit_row(
                f"weighted mean identity {ratio_col}",
                observed,
                1.0,
                bool(np.isclose(observed, 1.0, rtol=1e-6, atol=1e-6)),
            )
        )

    finite_values = merged[["R_CE", "R_CW", "R_CH", "R_WH", "log_R_CE", "log_R_CW", "log_R_CH", "log_R_WH"]].to_numpy(dtype=float)
    inf_count = int(np.isinf(finite_values).sum())
    audit_rows.append(_audit_row("ratio columns are finite for valid denominator rows", float(inf_count), 0.0, inf_count == 0))

    merged = merged.loc[:, OUTPUT_COLUMNS].sort_values(["R_CW", "R_CE"], ascending=[False, False]).reset_index(drop=True)
    return merged, pd.DataFrame(audit_rows)


def _carbon_weighted_mean(group: pd.DataFrame, column: str) -> float:
    """计算按 corridor carbon transfer 加权的 ratio 均值。"""
    weights = group["Carbon_Transfer"].to_numpy(dtype=float)
    values = group[column].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def _top_large_corridors(group: pd.DataFrame, n: int = 3) -> str:
    """输出每组中规模较大的代表 corridor，供报告使用。"""
    top = group.sort_values("Carbon_Transfer", ascending=False).head(n)
    return "; ".join(
        f"{row.From_Region}->{row.To_Region}:{row.Skill}"
        for row in top.itertuples(index=False)
    )


def _summary_by(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """生成 scale-aware corridor summary。"""
    totals = {
        "Carbon_Transfer": float(df["Carbon_Transfer"].sum()),
        "Employment_Transfer": float(df["Employment_Transfer"].sum()),
        "Hours_Transfer": float(df["Hours_Transfer"].sum()),
        "Wage_Transfer": float(df["Wage_Transfer"].sum()),
    }
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(list(group_cols), dropna=False, observed=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = {column: value for column, value in zip(group_cols, key_values)}
        row.update(
            {
                "N_cells": int(len(group)),
                "Carbon_share": float(group["Carbon_Transfer"].sum() / totals["Carbon_Transfer"]) if totals["Carbon_Transfer"] > 0 else np.nan,
                "Employment_share": float(group["Employment_Transfer"].sum() / totals["Employment_Transfer"]) if totals["Employment_Transfer"] > 0 else np.nan,
                "Hours_share": float(group["Hours_Transfer"].sum() / totals["Hours_Transfer"]) if totals["Hours_Transfer"] > 0 else np.nan,
                "Wage_share": float(group["Wage_Transfer"].sum() / totals["Wage_Transfer"]) if totals["Wage_Transfer"] > 0 else np.nan,
                "Mean_R_CE_weighted_by_carbon": _carbon_weighted_mean(group, "R_CE"),
                "Mean_R_CW_weighted_by_carbon": _carbon_weighted_mean(group, "R_CW"),
                "Top_large_corridors": _top_large_corridors(group),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Carbon_share", ascending=False).reset_index(drop=True)


def _explode_type_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """将多标签 corridor 类型拆成一行一个类型，便于汇总。"""
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        labels = [label.strip() for label in str(row[column]).split(";") if label.strip()]
        for label in labels or ["Unclassified"]:
            item = row.to_dict()
            item[column] = label
            rows.append(item)
    return pd.DataFrame(rows)


def build_corridor_summaries(corridor: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """生成 skill、type、region-pair 三类 scale-aware 汇总表。"""
    if corridor.empty:
        empty = pd.DataFrame()
        return {"summary_by_skill": empty, "summary_by_type": empty, "summary_by_region_pair": empty}
    by_skill = _summary_by(corridor, ["Skill"])
    by_type = _summary_by(_explode_type_rows(corridor, "Corridor_Type"), ["Corridor_Type", "Skill"])
    by_region_pair = _summary_by(corridor, ["From_Region", "To_Region"])
    return {
        "summary_by_skill": by_skill,
        "summary_by_type": by_type,
        "summary_by_region_pair": by_region_pair,
    }


def _read_metric_tables(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """读取现有 network_bilateral_*_transfer_by_skill.csv。"""
    frames: Dict[str, pd.DataFrame] = {}
    for metric in ("carbon", "employment", "wage", "hours"):
        path = input_dir / f"network_bilateral_{metric}_transfer_by_skill.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        frames[metric] = pd.read_csv(path)
    return frames


def _normalize_sector_driver_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """标准化 country-sector transfer 表，映射为 From=Producer, To=Demand。"""
    value_col = METRIC_VALUE_COLUMNS[metric]
    demand_col = _resolve_column(df, ["Demand_Country", "To_Region"])
    producer_col = _resolve_column(df, ["Producer_Country", "From_Region"])
    sector_col = _resolve_column(df, ["Sector", "sector"])
    raw_value_col = _resolve_column(df, [value_col, "Transfer_Value"])
    if demand_col is None or producer_col is None or sector_col is None or raw_value_col is None:
        raise ValueError(f"Cannot normalize country-sector table for {metric}")
    result = pd.DataFrame(
        {
            "From_Region": df[producer_col].astype(str),
            "To_Region": df[demand_col].astype(str),
            "Sector": df[sector_col].astype(str),
            value_col: pd.to_numeric(df[raw_value_col], errors="coerce"),
        }
    )
    if result[value_col].isna().any():
        raise ValueError(f"{metric} country-sector transfer contains non-numeric values")
    return result.loc[result["From_Region"] != result["To_Region"]].copy()


def build_sector_drivers(input_dir: Path, top_corridors: pd.DataFrame, top_sectors: int = 5) -> pd.DataFrame:
    """为 top corridor-skill by R_CW 连接 corridor-level sector carbon drivers。"""
    carbon_path = input_dir / "network_bilateral_carbon_transfer_by_country_sector.csv"
    if not carbon_path.exists() or top_corridors.empty:
        return pd.DataFrame(
            columns=[
                "From_Region",
                "To_Region",
                "Skill",
                "Sector",
                "Carbon_Transfer",
                "Sector_Carbon_Share_In_Corridor",
                "Sector_Rank_By_Carbon",
                "R_CW",
            ]
        )
    carbon_sector = _normalize_sector_driver_table(pd.read_csv(carbon_path), "carbon")
    grouped = carbon_sector.groupby(["From_Region", "To_Region", "Sector"], as_index=False, observed=False)["Carbon_Transfer"].sum()
    rows: List[pd.DataFrame] = []
    for _, corridor in top_corridors.iterrows():
        subset = grouped.loc[
            (grouped["From_Region"] == corridor["From_Region"])
            & (grouped["To_Region"] == corridor["To_Region"])
        ].copy()
        if subset.empty:
            continue
        total = float(subset["Carbon_Transfer"].sum())
        subset["Sector_Carbon_Share_In_Corridor"] = subset["Carbon_Transfer"] / total if total > 0.0 else np.nan
        subset = subset.sort_values("Carbon_Transfer", ascending=False).head(top_sectors).copy()
        subset["Sector_Rank_By_Carbon"] = np.arange(1, len(subset) + 1)
        subset["Skill"] = corridor["Skill"]
        subset["R_CW"] = corridor["R_CW"]
        rows.append(subset)
    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    columns = [
        "From_Region",
        "To_Region",
        "Skill",
        "Sector",
        "Carbon_Transfer",
        "Sector_Carbon_Share_In_Corridor",
        "Sector_Rank_By_Carbon",
        "R_CW",
    ]
    return result.loc[:, columns]


def write_outputs(
    corridor: pd.DataFrame,
    top_corridors: pd.DataFrame,
    top_by_scale: pd.DataFrame,
    sector_drivers: pd.DataFrame,
    audit: pd.DataFrame,
    summaries: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """写出 corridor-skill diagnostics CSV。"""
    csv_dir = output_dir if output_dir.name == "csv" else output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "corridor_skill_misalignment.csv": corridor,
        "corridor_skill_misalignment_scale_weighted.csv": corridor,
        "top_corridor_skill_misalignment.csv": top_corridors,
        "top_corridor_skill_misalignment_by_ratio.csv": top_corridors,
        "top_corridor_skill_misalignment_by_scale_weighted_score.csv": top_by_scale,
        "corridor_skill_misalignment_sector_drivers.csv": sector_drivers,
        "corridor_skill_misalignment_summary_by_skill.csv": summaries.get("summary_by_skill", pd.DataFrame()),
        "corridor_skill_misalignment_summary_by_type.csv": summaries.get("summary_by_type", pd.DataFrame()),
        "corridor_skill_misalignment_summary_by_region_pair.csv": summaries.get("summary_by_region_pair", pd.DataFrame()),
        "corridor_skill_misalignment_audit.csv": audit,
    }
    for file_name, df in outputs.items():
        df.to_csv(csv_dir / file_name, index=False)
        print(f"保存: {csv_dir / file_name}")








