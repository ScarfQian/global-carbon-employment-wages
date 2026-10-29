"""Overlap, concentration and sector divergence of embodied bilateral networks."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


# 路径与常量配置

NETWORKS: List[str] = ["carbon", "employment", "wage", "hours"]
NETWORK_PAIRS: List[Tuple[str, str]] = list(combinations(NETWORKS, 2))
TOP_KS_OVERLAP: List[int] = [5, 10, 20, 50, 100]
TOP_KS_CONCENTRATION: List[int] = [1, 5, 10, 20, 50, 100]
TOP_KS_SECTOR: List[int] = [5, 10, 20]

DEFAULT_OUTPUT_DIR = Path("network")
DEFAULT_SKILL_INPUTS: Dict[str, str] = {
    metric: str(DEFAULT_OUTPUT_DIR / "csv" / f"network_bilateral_{metric}_transfer_by_skill.csv")
    for metric in NETWORKS
}
DEFAULT_SECTOR_INPUTS: Dict[str, str] = {
    metric: str(DEFAULT_OUTPUT_DIR / "csv" / f"network_bilateral_{metric}_transfer_by_country_sector.csv")
    for metric in NETWORKS
}

SKILL_VALUE_COLUMNS: Dict[str, str] = {
    "carbon": "Carbon_Transfer",
    "employment": "Employment_Transfer",
    "wage": "Wage_Transfer",
    "hours": "Hours_Transfer",
}
SECTOR_VALUE_COLUMNS: Dict[str, str] = dict(SKILL_VALUE_COLUMNS)

OUTPUT_FILE_NAMES: Dict[str, str] = {
    "input_checks_csv": "csv/embodied_network_input_checks.csv",
    "overlap_summary_csv": "csv/embodied_network_overlap_summary.csv",
    "concentration_summary_csv": "csv/embodied_network_concentration_summary.csv",
    "edge_rank_full_csv": "csv/embodied_network_edge_rank_full.csv",
    "corridor_divergence_summary_csv": "csv/embodied_corridor_sector_divergence_summary.csv",
    "corridor_sector_share_long_csv": "csv/embodied_corridor_sector_share_long.csv",
    "overlap_summary_json": "json/embodied_network_overlap_summary.json",
    "concentration_summary_json": "json/embodied_network_concentration_summary.json",
    "note_md": "network_methods.md",
}


# 通用数值工具


def _safe_float(value: float) -> float:
    """将非有限值客观替换为 0，避免摘要表出现 NaN。"""
    if value is None:
        return 0.0
    numeric = float(value)
    if not np.isfinite(numeric):
        return 0.0
    return numeric


def _safe_series_corr(left: np.ndarray, right: np.ndarray, method: str) -> float:
    """安全计算相关系数，常数向量或空向量时返回 0。"""
    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    if left_array.size == 0 or right_array.size == 0 or left_array.size != right_array.size:
        return 0.0
    if np.allclose(left_array, left_array[0]) or np.allclose(right_array, right_array[0]):
        return 0.0
    value = pd.Series(left_array).corr(pd.Series(right_array), method=method)
    return _safe_float(0.0 if pd.isna(value) else value)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """计算余弦相似度；零向量情形客观返回 0。"""
    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return _safe_float(np.dot(left_array, right_array) / (left_norm * right_norm))


def _to_share_vector(values: np.ndarray) -> np.ndarray:
    """把非负边权转换为 share 向量；总量为 0 时返回零向量。"""
    array = np.asarray(values, dtype=float).reshape(-1)
    total = float(array.sum())
    if total <= 0.0:
        return np.zeros_like(array)
    return array / total


def _l1_distance(left: np.ndarray, right: np.ndarray) -> float:
    """计算 L1 距离。"""
    return _safe_float(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)).sum())


def _l2_distance(left: np.ndarray, right: np.ndarray) -> float:
    """计算 L2 距离。"""
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return _safe_float(np.sqrt(np.square(delta).sum()))


def _jaccard_index(left_keys: Sequence[Tuple], right_keys: Sequence[Tuple]) -> float:
    """计算集合 Jaccard；双空集情形定义为 1。"""
    left_set = set(left_keys)
    right_set = set(right_keys)
    union = left_set | right_set
    if not union:
        return 1.0
    return _safe_float(len(left_set & right_set) / len(union))


def _js_divergence(left_share: np.ndarray, right_share: np.ndarray) -> float:
    """计算 Jensen-Shannon divergence；单边零分布时客观记为 1。"""
    left = np.asarray(left_share, dtype=float).reshape(-1)
    right = np.asarray(right_share, dtype=float).reshape(-1)
    left_total = float(left.sum())
    right_total = float(right.sum())
    if left_total <= 0.0 and right_total <= 0.0:
        return 0.0
    if left_total <= 0.0 or right_total <= 0.0:
        return 1.0
    left_prob = left / left_total
    right_prob = right / right_total
    mean_prob = 0.5 * (left_prob + right_prob)

    def _kl_divergence(prob: np.ndarray, ref: np.ndarray) -> float:
        mask = prob > 0.0
        if not mask.any():
            return 0.0
        return float(np.sum(prob[mask] * np.log(prob[mask] / ref[mask])))

    return _safe_float(0.5 * _kl_divergence(left_prob, mean_prob) + 0.5 * _kl_divergence(right_prob, mean_prob))


def _gini(values: np.ndarray) -> float:
    """按标准非负序列公式计算 Gini 系数。"""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    if np.any(array < 0.0):
        array = np.clip(array, 0.0, None)
    total = float(array.sum())
    if total <= 0.0:
        return 0.0
    sorted_array = np.sort(array)
    n = sorted_array.size
    cum = np.cumsum(sorted_array)
    gini = (n + 1 - 2.0 * cum.sum() / total) / n
    return _safe_float(gini)


def _top_share(shares: np.ndarray, top_k: int) -> float:
    """计算 share 序列前 K 项的累计份额。"""
    array = np.sort(np.asarray(shares, dtype=float).reshape(-1))[::-1]
    if array.size == 0:
        return 0.0
    return _safe_float(array[: min(top_k, array.size)].sum())


# 输入读取与标准化


def _load_csv(path: str, value_column: str) -> pd.DataFrame:
    """读取输入文件并标准化数值列。"""
    frame = pd.read_csv(path)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    return frame


def _summarize_input_table(
    dataset: str,
    level: str,
    metric: str,
    frame: pd.DataFrame,
    origin_col: str,
    destination_col: str,
    value_col: str,
    note_text: str,
) -> Dict[str, object]:
    """生成单个输入文件的完整性与口径检查摘要。"""
    domestic_mask = frame[origin_col].eq(frame[destination_col])
    total_transfer = float(frame[value_col].sum(skipna=True))
    domestic_transfer = float(frame.loc[domestic_mask, value_col].sum(skipna=True))
    cross_border_transfer = float(frame.loc[~domestic_mask, value_col].sum(skipna=True))
    if total_transfer > 0.0:
        domestic_share = domestic_transfer / total_transfer
        cross_border_share = cross_border_transfer / total_transfer
    else:
        domestic_share = 0.0
        cross_border_share = 0.0
    return {
        "dataset": dataset,
        "level": level,
        "metric": metric,
        "total_rows": int(len(frame)),
        "domestic_rows": int(domestic_mask.sum()),
        "cross_border_rows": int((~domestic_mask).sum()),
        "total_transfer": _safe_float(total_transfer),
        "domestic_share": _safe_float(domestic_share),
        "cross_border_share": _safe_float(cross_border_share),
        "min_transfer": _safe_float(frame[value_col].min(skipna=True)),
        "max_transfer": _safe_float(frame[value_col].max(skipna=True)),
        "has_nan": bool(frame.isna().any().any()),
        "has_negative": bool(frame[value_col].lt(0.0).any()),
        "notes": note_text,
    }


def _build_input_checks(
    skill_inputs: Mapping[str, str],
    sector_inputs: Mapping[str, str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """读取全部输入文件，并返回检查表与原始数据。"""
    check_rows: List[Dict[str, object]] = []
    skill_frames: Dict[str, pd.DataFrame] = {}
    sector_frames: Dict[str, pd.DataFrame] = {}

    for metric in NETWORKS:
        skill_frame = _load_csv(skill_inputs[metric], SKILL_VALUE_COLUMNS[metric])
        sector_frame = _load_csv(sector_inputs[metric], SECTOR_VALUE_COLUMNS[metric])
        skill_frames[metric] = skill_frame
        sector_frames[metric] = sector_frame

        check_rows.append(
            _summarize_input_table(
                dataset=Path(skill_inputs[metric]).name,
                level="country_country_skill_bilateral",
                metric=metric,
                frame=skill_frame,
                origin_col="From_Region",
                destination_col="To_Region",
                value_col=SKILL_VALUE_COLUMNS[metric],
                note_text="Cross-border check uses From_Region and To_Region on by-skill bilateral rows.",
            )
        )
        check_rows.append(
            _summarize_input_table(
                dataset=Path(sector_inputs[metric]).name,
                level="country_country_sector_bilateral",
                metric=metric,
                frame=sector_frame,
                origin_col="Demand_Country",
                destination_col="Producer_Country",
                value_col=SECTOR_VALUE_COLUMNS[metric],
                note_text="Domestic and cross-border rows are both retained in the raw country-sector export.",
            )
        )

    check_frame = pd.DataFrame(check_rows)
    return check_frame.sort_values(["level", "metric", "dataset"]).reset_index(drop=True), skill_frames, sector_frames


# 边表标准化与 overlap / concentration / edge rank


def _prepare_skill_network_levels(skill_frames: Mapping[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """把 by-skill bilateral 结果整理为两个层次的标准边表。"""
    levels: Dict[str, Dict[str, pd.DataFrame]] = {
        "country_country": {},
        "country_country_skill": {},
    }

    for metric in NETWORKS:
        value_col = SKILL_VALUE_COLUMNS[metric]
        raw = skill_frames[metric].copy()
        raw = raw.rename(columns={value_col: "transfer_value"})
        raw["transfer_value"] = pd.to_numeric(raw["transfer_value"], errors="coerce").fillna(0.0)
        raw = raw[raw["transfer_value"] > 0.0].copy()

        country_country_skill = (
            raw.groupby(["From_Region", "To_Region", "Skill"], as_index=False, dropna=False)["transfer_value"]
            .sum()
            .sort_values(["From_Region", "To_Region", "Skill"])
            .reset_index(drop=True)
        )
        country_country = (
            country_country_skill.groupby(["From_Region", "To_Region"], as_index=False)["transfer_value"]
            .sum()
            .sort_values(["From_Region", "To_Region"])
            .reset_index(drop=True)
        )

        levels["country_country_skill"][metric] = country_country_skill
        levels["country_country"][metric] = country_country

    return levels


def _make_edge_key_frame(frame: pd.DataFrame, key_cols: Sequence[str]) -> pd.Series:
    """把边表转换为以 key 为索引的 transfer 序列。"""
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.set_index(list(key_cols))["transfer_value"].sort_index()


def _top_keys(frame: pd.DataFrame, key_cols: Sequence[str], top_k: int) -> List[Tuple]:
    """提取边权前 K 名的 key 集合。"""
    if frame.empty:
        return []
    subset = frame.sort_values("transfer_value", ascending=False).head(min(top_k, len(frame)))
    return [tuple(row[col] for col in key_cols) for _, row in subset.iterrows()]


def _compute_overlap_summary(level_tables: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    """计算 country_country 与 country_country_skill 两个层次的 overlap / decoupling 指标。"""
    rows: List[Dict[str, object]] = []
    key_columns_map = {
        "country_country": ["From_Region", "To_Region"],
        "country_country_skill": ["From_Region", "To_Region", "Skill"],
    }

    for level, network_tables in level_tables.items():
        key_cols = key_columns_map[level]
        for network_a, network_b in NETWORK_PAIRS:
            left = network_tables[network_a]
            right = network_tables[network_b]
            left_series = _make_edge_key_frame(left, key_cols)
            right_series = _make_edge_key_frame(right, key_cols)
            union_index = left_series.index.union(right_series.index)
            left_values = left_series.reindex(union_index, fill_value=0.0).to_numpy(dtype=float)
            right_values = right_series.reindex(union_index, fill_value=0.0).to_numpy(dtype=float)
            left_share = _to_share_vector(left_values)
            right_share = _to_share_vector(right_values)

            row = {
                "level": level,
                "network_a": network_a,
                "network_b": network_b,
                "edge_count_union": int(len(union_index)),
                "edge_count_intersection_nonzero": int(np.logical_and(left_values > 0.0, right_values > 0.0).sum()),
                "pearson_corr": _safe_series_corr(left_values, right_values, method="pearson"),
                "spearman_corr": _safe_series_corr(left_values, right_values, method="spearman"),
                "kendall_corr": _safe_series_corr(left_values, right_values, method="kendall"),
                "cosine_similarity": _cosine_similarity(left_values, right_values),
                "l1_distance_share": _l1_distance(left_share, right_share),
                "l2_distance_share": _l2_distance(left_share, right_share),
                "notes": "Union-aligned bilateral edges with zero fill; correlations use raw transfer values and distances use share vectors.",
            }
            for top_k in TOP_KS_OVERLAP:
                row[f"top{top_k}_jaccard"] = _jaccard_index(
                    _top_keys(left, key_cols, top_k),
                    _top_keys(right, key_cols, top_k),
                )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["level", "network_a", "network_b"]).reset_index(drop=True)


def _compute_concentration_summary(level_tables: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    """计算各网络在两个层次上的集中度指标。"""
    rows: List[Dict[str, object]] = []

    for level, network_tables in level_tables.items():
        for network, frame in network_tables.items():
            values = frame["transfer_value"].to_numpy(dtype=float) if not frame.empty else np.array([], dtype=float)
            total_transfer = float(values.sum()) if values.size else 0.0
            shares = _to_share_vector(values)
            mean_value = float(values.mean()) if values.size else 0.0
            median_value = float(np.median(values)) if values.size else 0.0
            std_value = float(values.std(ddof=0)) if values.size else 0.0
            row = {
                "level": level,
                "network": network,
                "edge_count": int(len(frame)),
                "total_transfer": _safe_float(total_transfer),
                "mean_edge_weight": _safe_float(mean_value),
                "median_edge_weight": _safe_float(median_value),
                "std_edge_weight": _safe_float(std_value),
                "cv_edge_weight": _safe_float(std_value / mean_value) if mean_value > 0.0 else 0.0,
                "hhi_edge_share": _safe_float(np.square(shares).sum()),
                "entropy_edge_share": _safe_float(-np.sum(shares[shares > 0.0] * np.log(shares[shares > 0.0]))) if shares.size else 0.0,
                "gini_edge_weight": _gini(values),
                "notes": "Metrics are computed on bilateral transfer edges with positive transfer values.",
            }
            for top_k in TOP_KS_CONCENTRATION:
                row[f"top{top_k}_share"] = _top_share(shares, top_k)
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["level", "network"]).reset_index(drop=True)


def _build_edge_rank_full(level_tables: Mapping[str, Mapping[str, pd.DataFrame]]) -> pd.DataFrame:
    """生成两个层次下四类网络的全量边排序表。"""
    rows: List[pd.DataFrame] = []

    for level, network_tables in level_tables.items():
        for network, frame in network_tables.items():
            ordered = frame.sort_values("transfer_value", ascending=False).reset_index(drop=True).copy()
            total_transfer = float(ordered["transfer_value"].sum()) if not ordered.empty else 0.0
            ordered["level"] = level
            ordered["network"] = network
            ordered["rank_desc"] = np.arange(1, len(ordered) + 1)
            ordered["from_region"] = ordered["From_Region"]
            ordered["to_region"] = ordered["To_Region"]
            if "Skill" in ordered.columns:
                ordered["skill"] = ordered["Skill"].fillna("")
            else:
                ordered["skill"] = ""
            ordered["edge_key"] = ordered.apply(
                lambda row: f"{row['from_region']}->{row['to_region']}|{row['skill']}" if row["skill"] else f"{row['from_region']}->{row['to_region']}",
                axis=1,
            )
            if total_transfer > 0.0:
                ordered["transfer_share"] = ordered["transfer_value"] / total_transfer
            else:
                ordered["transfer_share"] = 0.0
            ordered["is_cross_border"] = ordered["from_region"].ne(ordered["to_region"])
            rows.append(
                ordered[[
                    "level",
                    "network",
                    "rank_desc",
                    "from_region",
                    "to_region",
                    "skill",
                    "edge_key",
                    "transfer_value",
                    "transfer_share",
                    "is_cross_border",
                ]]
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "level",
                "network",
                "rank_desc",
                "from_region",
                "to_region",
                "skill",
                "edge_key",
                "transfer_value",
                "transfer_share",
                "is_cross_border",
            ]
        )
    return pd.concat(rows, ignore_index=True)


# corridor-sector divergence 与 sector share 长表


def _prepare_cross_border_sector_tables(sector_frames: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """将原始 country-sector 表筛成 cross-border only，并聚合到 corridor × sector。"""
    prepared: Dict[str, pd.DataFrame] = {}

    for metric in NETWORKS:
        value_col = SECTOR_VALUE_COLUMNS[metric]
        raw = sector_frames[metric].copy()
        raw = raw.rename(columns={value_col: "transfer_value"})
        raw["transfer_value"] = pd.to_numeric(raw["transfer_value"], errors="coerce").fillna(0.0)
        raw = raw[raw["Demand_Country"].ne(raw["Producer_Country"])].copy()
        raw = raw[raw["transfer_value"] > 0.0].copy()
        grouped = (
            raw.groupby(["Demand_Country", "Producer_Country", "Sector"], as_index=False)["transfer_value"]
            .sum()
            .sort_values(["Demand_Country", "Producer_Country", "Sector"])
            .reset_index(drop=True)
        )
        prepared[metric] = grouped

    return prepared


def _build_corridor_sector_share_long(sector_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """输出 cross-border only 的 corridor-sector 长表。"""
    rows: List[pd.DataFrame] = []

    for network, frame in sector_tables.items():
        ordered = frame.copy()
        if ordered.empty:
            continue
        totals = ordered.groupby(["Demand_Country", "Producer_Country"]) ["transfer_value"].transform("sum")
        ordered["sector_share"] = np.divide(
            ordered["transfer_value"],
            totals,
            out=np.zeros(len(ordered), dtype=float),
            where=totals.to_numpy(dtype=float) > 0.0,
        )
        ordered = ordered.sort_values(
            ["Demand_Country", "Producer_Country", "transfer_value", "Sector"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        ordered["rank_desc"] = ordered.groupby(["Demand_Country", "Producer_Country"]).cumcount() + 1
        ordered["is_top10"] = ordered["rank_desc"].le(10)
        ordered["is_top20"] = ordered["rank_desc"].le(20)
        ordered["network"] = network
        rows.append(
            ordered.rename(
                columns={
                    "Demand_Country": "demand_country",
                    "Producer_Country": "producer_country",
                    "Sector": "sector",
                }
            )[[
                "demand_country",
                "producer_country",
                "network",
                "sector",
                "transfer_value",
                "sector_share",
                "rank_desc",
                "is_top10",
                "is_top20",
            ]]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "demand_country",
                "producer_country",
                "network",
                "sector",
                "transfer_value",
                "sector_share",
                "rank_desc",
                "is_top10",
                "is_top20",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _build_sector_series_map(sector_tables: Mapping[str, pd.DataFrame]) -> Tuple[Dict[str, Dict[Tuple[str, str], pd.Series]], Dict[str, Dict[Tuple[str, str], float]]]:
    """为 divergence 计算构建 corridor -> sector series 与 corridor totals 映射。"""
    series_map: Dict[str, Dict[Tuple[str, str], pd.Series]] = {metric: {} for metric in NETWORKS}
    totals_map: Dict[str, Dict[Tuple[str, str], float]] = {metric: {} for metric in NETWORKS}

    for metric, frame in sector_tables.items():
        if frame.empty:
            continue
        for (demand_country, producer_country), subset in frame.groupby(["Demand_Country", "Producer_Country"]):
            series = subset.set_index("Sector")["transfer_value"].sort_index()
            corridor = (str(demand_country), str(producer_country))
            series_map[metric][corridor] = series
            totals_map[metric][corridor] = float(series.sum())

    return series_map, totals_map


def _top_sector_keys(series: pd.Series, top_k: int) -> List[Tuple[str]]:
    """提取 corridor 内 sector 前 K 名集合。"""
    if series is None or series.empty:
        return []
    ordered = series.sort_values(ascending=False).head(min(top_k, len(series)))
    return [(str(index),) for index in ordered.index.tolist()]


def _compute_corridor_sector_divergence(sector_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """对全部可识别的 cross-border corridor 计算 sector-share divergence。"""
    series_map, totals_map = _build_sector_series_map(sector_tables)
    corridor_union = sorted(
        set().union(*(set(metric_map.keys()) for metric_map in series_map.values()))
    )
    rows: List[Dict[str, object]] = []

    for demand_country, producer_country in corridor_union:
        corridor = (demand_country, producer_country)
        for network_a, network_b in NETWORK_PAIRS:
            left_series = series_map[network_a].get(corridor, pd.Series(dtype=float))
            right_series = series_map[network_b].get(corridor, pd.Series(dtype=float))
            union_index = left_series.index.union(right_series.index)
            left_values = left_series.reindex(union_index, fill_value=0.0).to_numpy(dtype=float)
            right_values = right_series.reindex(union_index, fill_value=0.0).to_numpy(dtype=float)
            left_share = _to_share_vector(left_values)
            right_share = _to_share_vector(right_values)

            row = {
                "demand_country": demand_country,
                "producer_country": producer_country,
                "pair_name": f"{network_a}_vs_{network_b}",
                "network_a": network_a,
                "network_b": network_b,
                "sector_count_union": int(len(union_index)),
                "sector_count_intersection_nonzero": int(np.logical_and(left_values > 0.0, right_values > 0.0).sum()),
                "cosine_similarity": _cosine_similarity(left_share, right_share),
                "l1_distance_share": _l1_distance(left_share, right_share),
                "l2_distance_share": _l2_distance(left_share, right_share),
                "js_divergence": _js_divergence(left_share, right_share),
                "total_transfer_a": _safe_float(totals_map[network_a].get(corridor, 0.0)),
                "total_transfer_b": _safe_float(totals_map[network_b].get(corridor, 0.0)),
                "notes": "Cross-border sector-share vectors are aligned on corridor-specific sector unions with zero fill for missing sectors.",
            }
            for top_k in TOP_KS_SECTOR:
                row[f"top{top_k}_sector_overlap_jaccard"] = _jaccard_index(
                    _top_sector_keys(left_series, top_k),
                    _top_sector_keys(right_series, top_k),
                )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["demand_country", "producer_country", "network_a", "network_b"]
    ).reset_index(drop=True)


# 说明文件与结果写出


def _write_note(output_dir: Path, skill_input_paths: Mapping[str, str], sector_input_paths: Mapping[str, str]) -> Path:
    """Record the accounting scope of the generated network metrics."""
    note_path = output_dir / OUTPUT_FILE_NAMES["note_md"]
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# Embodied network metrics\n\n"
        "By-skill networks use cross-border intermediate-input transfers. "
        "Country-sector networks use final-demand-induced production; domestic rows "
        "are excluded from corridor divergence. Carbon by skill is attributed exposure. "
        "Overlap and concentration use bilateral flows rather than net transfers.\n",
        encoding="utf-8",
    )
    return note_path


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    """统一写出 CSV，保持索引关闭。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json_records(frame: pd.DataFrame, path: Path) -> None:
    """把摘要表按 records 形式写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


# 主流程


def run_all(
    skill_input_paths: Mapping[str, str] | None = None,
    sector_input_paths: Mapping[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, str]:
    """执行全部系统指标计算并写出标准化结果。"""
    skill_paths = dict(DEFAULT_SKILL_INPUTS if skill_input_paths is None else skill_input_paths)
    sector_paths = dict(DEFAULT_SECTOR_INPUTS if sector_input_paths is None else sector_input_paths)
    output_path = Path(DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)

    input_checks, skill_frames, sector_frames = _build_input_checks(skill_paths, sector_paths)
    level_tables = _prepare_skill_network_levels(skill_frames)
    overlap_summary = _compute_overlap_summary(level_tables)
    concentration_summary = _compute_concentration_summary(level_tables)
    edge_rank_full = _build_edge_rank_full(level_tables)
    cross_border_sector_tables = _prepare_cross_border_sector_tables(sector_frames)
    corridor_sector_share_long = _build_corridor_sector_share_long(cross_border_sector_tables)
    corridor_divergence_summary = _compute_corridor_sector_divergence(cross_border_sector_tables)
    note_path = _write_note(output_path, skill_paths, sector_paths)

    output_paths = {
        key: str(output_path / relative_path) for key, relative_path in OUTPUT_FILE_NAMES.items() if key != "note_md"
    }
    output_paths["note_md"] = str(note_path)

    _write_dataframe(input_checks, Path(output_paths["input_checks_csv"]))
    _write_dataframe(overlap_summary, Path(output_paths["overlap_summary_csv"]))
    _write_dataframe(concentration_summary, Path(output_paths["concentration_summary_csv"]))
    _write_dataframe(edge_rank_full, Path(output_paths["edge_rank_full_csv"]))
    _write_dataframe(corridor_divergence_summary, Path(output_paths["corridor_divergence_summary_csv"]))
    _write_dataframe(corridor_sector_share_long, Path(output_paths["corridor_sector_share_long_csv"]))
    _write_json_records(overlap_summary, Path(output_paths["overlap_summary_json"]))
    _write_json_records(concentration_summary, Path(output_paths["concentration_summary_json"]))

    return output_paths
