"""Summarize policy frontiers, labour-hour lock and network distributions."""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .pipeline import SIGNED_OBJECTIVES, SPECIFICATION_ID, EXACT_STATUS, SKILLS

CREATED_AT = "2026-10-28"
EXECUTOR = "ScarfQian"
RUN_VERSION = "1.0.0"
DECISION_COLUMNS = ("tax_rate", "renewable_share", "shift_share", "training_support_level")
KEY_YEARS = (2030, 2050, 2060)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_ROOT = Path.cwd() / "results"
CURRENT_READ_DIR = FINAL_ROOT / "01_Current"
FUTURE_READ_DIR = FINAL_ROOT / "02_Future"
SOURCE_READ_DIR = FINAL_ROOT / "03_SourceData"
FRONTIER_DIR = FINAL_ROOT / "04_Summaries"
HOUR_DIR = FINAL_ROOT / "04_Summaries"
FUTURE_DIR = FINAL_ROOT / "04_Summaries"
EVIDENCE_DIR = FINAL_ROOT / "04_Summaries"


def configure_paths(output_root: Path) -> None:
    """Read computed tables and write summaries under one chosen result root."""
    global FINAL_ROOT, CURRENT_READ_DIR, FUTURE_READ_DIR, SOURCE_READ_DIR
    global FRONTIER_DIR, HOUR_DIR, FUTURE_DIR, EVIDENCE_DIR
    FINAL_ROOT = output_root.expanduser().resolve()
    CURRENT_READ_DIR = FINAL_ROOT / "01_Current"
    FUTURE_READ_DIR = FINAL_ROOT / "02_Future"
    SOURCE_READ_DIR = FINAL_ROOT / "03_SourceData"
    FRONTIER_DIR = HOUR_DIR = FUTURE_DIR = EVIDENCE_DIR = FINAL_ROOT / "04_Summaries"
    FRONTIER_DIR.mkdir(parents=True, exist_ok=True)


def relative_project_path(path: Path) -> str:
    return os.path.relpath(path, PROJECT_ROOT)


def write_csv(frame: pd.DataFrame, path: Path, sort_by: Optional[Sequence[str]] = None) -> None:
    """稳定排序并写出 UTF-8 CSV。"""
    output = frame.copy()
    if sort_by:
        valid = [column for column in sort_by if column in output.columns]
        if valid:
            output = output.sort_values(valid, kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)



def stamp(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """为 patch 表补充统一审计元数据。"""
    result = frame.copy()
    result["Run_Version"] = RUN_VERSION
    result["Created_At"] = CREATED_AT
    result["Executor"] = EXECUTOR
    result["Specification_ID"] = SPECIFICATION_ID
    result["Exact_or_Surrogate"] = EXACT_STATUS
    result["Upstream_Sources"] = source
    return result



def nondominated_mask(values: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    """按四目标最大化和给定数值容差独立判定非支配集合。"""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 4 or not np.isfinite(matrix).all():
        raise ValueError("Pareto 输入必须是有限的 n x 4 矩阵。")
    flags = np.ones(len(matrix), dtype=bool)
    for idx, row in enumerate(matrix):
        no_worse = np.all(matrix >= row - tolerance, axis=1)
        meaningfully_better = np.any(matrix > row + tolerance, axis=1)
        no_worse[idx] = False
        flags[idx] = not bool(np.any(no_worse & meaningfully_better))
    return flags



def normalized_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """对指定列做稳定 min-max normalization。"""
    values = frame[list(columns)].to_numpy(dtype=float)
    lower = np.min(values, axis=0)
    span = np.ptp(values, axis=0)
    return np.divide(values - lower, span, out=np.zeros_like(values), where=span > 0.0)



def build_frontier_tolerance_sensitivity(candidates: pd.DataFrame, formal_frontier: pd.DataFrame) -> pd.DataFrame:
    """对同一去重候选宇宙重算 strict 与三个容差前沿。"""
    unique = candidates.drop_duplicates(subset=list(DECISION_COLUMNS), keep="first").reset_index(drop=True)
    values = unique[list(SIGNED_OBJECTIVES)].to_numpy(dtype=float)
    formal_ids = set(formal_frontier["Policy_ID"].astype(str))
    strict_mask = nondominated_mask(values, 0.0)
    strict_ids = set(unique.loc[strict_mask, "Policy_ID"].astype(str))
    if strict_ids != formal_ids:
        raise AssertionError(f"STRICT_FRONTIER_DEFECT: reconstructed={len(strict_ids)} formal={len(formal_ids)} overlap={len(strict_ids & formal_ids)}")
    rows = []
    for tolerance in (0.0, 1e-10, 1e-8, 1e-6):
        mask = nondominated_mask(values, tolerance)
        ids = set(unique.loc[mask, "Policy_ID"].astype(str))
        overlap, union = len(ids & strict_ids), len(ids | strict_ids)
        rows.append({
            "Tolerance": tolerance,
            "Frontier_Size": len(ids),
            "Overlap_With_Strict": overlap,
            "Jaccard_With_Strict": overlap / union if union else 1.0,
            "Notes": "formal strict reconstruction" if tolerance == 0.0 else "sensitivity only; formal strict frontier unchanged",
        })
    output = stamp(pd.DataFrame(rows), relative_project_path(FINAL_ROOT / "01_Current" / "current_nsga3_union_all_candidates.csv"))
    write_csv(output, FRONTIER_DIR / "frontier_tolerance_sensitivity.csv", ["Tolerance"])
    return output



def connected_clusters(matrix: np.ndarray, tolerance: float) -> list[list[int]]:
    """用半径邻接关系构建 near-duplicate connected components。"""
    if len(matrix) == 0:
        return []
    pairs = cKDTree(np.asarray(matrix, dtype=float)).query_pairs(r=float(tolerance), output_type="set")
    parent = np.arange(len(matrix), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in pairs:
        union(int(left), int(right))
    groups: Dict[int, list[int]] = {}
    for index in range(len(matrix)):
        groups.setdefault(find(index), []).append(index)
    return [members for members in groups.values() if len(members) > 1]



def exact_duplicate_groups(frame: pd.DataFrame, columns: Sequence[str]) -> list[list[int]]:
    """按完全一致列值返回多成员分组。"""
    groups = frame.groupby(list(columns), dropna=False, sort=False).indices
    return [list(map(int, members)) for members in groups.values() if len(members) > 1]



def cluster_rows(frame: pd.DataFrame, matrix: np.ndarray, groups: Sequence[Sequence[int]], universe: str, vector_type: str, tolerance: float) -> list[Dict[str, object]]:
    """把重复分组转为可审计 cluster 表。"""
    rows = []
    for cluster_index, members in enumerate(groups, start=1):
        member_list = list(map(int, members))
        points = matrix[member_list, :]
        max_distance = 0.0
        if len(points) > 1:
            max_distance = float(max(np.linalg.norm(points[i] - points[j]) for i, j in itertools.combinations(range(len(points)), 2)))
        rows.append({
            "Universe": universe,
            "Vector_Type": vector_type,
            "Tolerance": tolerance,
            "Cluster_ID": f"{universe}_{vector_type}_{tolerance:g}_{cluster_index:04d}",
            "N_Members": len(member_list),
            "Policy_IDs": ";".join(frame.iloc[member_list]["Policy_ID"].astype(str)),
            "Max_Normalized_Euclidean_Distance": max_distance,
        })
    return rows



def build_duplicate_audit(candidates: pd.DataFrame, frontier: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """审计 all candidates 与 final frontier 的 exact/near duplicates。"""
    audit_rows, all_cluster_rows = [], []
    for universe, frame in (("all_6300_candidates", candidates.reset_index(drop=True)), ("strict_841_frontier", frontier.reset_index(drop=True))):
        for vector_type, columns in (("decision", DECISION_COLUMNS), ("objective", SIGNED_OBJECTIVES)):
            norm = normalized_matrix(frame, columns)
            exact_groups = exact_duplicate_groups(frame, columns)
            exact_rows = sum(len(group) for group in exact_groups)
            audit_rows.append({
                "Universe": universe, "Vector_Type": vector_type, "Duplicate_Definition": "exact",
                "Tolerance": 0.0, "N_Rows": len(frame), "N_Groups_With_Multiple": len(exact_groups),
                "N_Rows_In_Multiple": exact_rows, "Excess_Duplicate_Rows": sum(len(group) - 1 for group in exact_groups),
            })
            all_cluster_rows.extend(cluster_rows(frame, norm, exact_groups, universe, vector_type, 0.0))
            for tolerance in (1e-6, 1e-4, 1e-3):
                groups = connected_clusters(norm, tolerance)
                audit_rows.append({
                    "Universe": universe, "Vector_Type": vector_type,
                    "Duplicate_Definition": "normalized_euclidean_near_duplicate", "Tolerance": tolerance,
                    "N_Rows": len(frame), "N_Groups_With_Multiple": len(groups),
                    "N_Rows_In_Multiple": sum(len(group) for group in groups),
                    "Excess_Duplicate_Rows": sum(len(group) - 1 for group in groups),
                })
                all_cluster_rows.extend(cluster_rows(frame, norm, groups, universe, vector_type, tolerance))
    audit = stamp(pd.DataFrame(audit_rows), "current exact candidates and strict frontier")
    clusters = stamp(pd.DataFrame(all_cluster_rows), "current exact candidates and strict frontier")
    write_csv(audit, FRONTIER_DIR / "frontier_duplicate_audit.csv", ["Universe", "Vector_Type", "Tolerance"])
    write_csv(clusters, FRONTIER_DIR / "frontier_near_duplicate_clusters.csv", ["Universe", "Vector_Type", "Tolerance", "Cluster_ID"])
    return audit, clusters



def build_seed_convergence(candidates: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    """展开每个 seed 的实际评价、贡献、范围与收敛量。"""
    existing = pd.read_csv(CURRENT_READ_DIR / "current_nsga3_convergence_audit.csv")
    rows = []
    for _, audit in existing.iterrows():
        seed = int(audit["Seed"])
        all_seed = candidates.loc[candidates["Seed"].astype(int).eq(seed)]
        front_seed = pd.read_csv(CURRENT_READ_DIR / f"current_nsga3_seed{seed}_frontier.csv")
        union_contribution = frontier.loc[frontier["Seed"].astype(int).eq(seed)]
        row: Dict[str, object] = {
            "Seed": seed, "Exact_Evaluated_Candidates": len(all_seed), "Seed_Frontier_Size": len(front_seed),
            "Contribution_to_Final_Union_Frontier": len(union_contribution),
            "Contribution_Share_of_Union": len(union_contribution) / len(frontier),
            "Normalized_Hypervolume": float(audit["Normalized_Hypervolume"]),
            "Normalized_Union_to_Seed_Distance": float(audit["Normalized_Union_to_Seed_Frontier_Distance"]),
            "Frontier_Coverage": float(audit["Minimum_Union_Objective_Range_Coverage"]),
            "Convergence_Passed": bool(audit["Passed"]),
        }
        for column in SIGNED_OBJECTIVES:
            row[f"{column}_Min"], row[f"{column}_Max"] = float(all_seed[column].min()), float(all_seed[column].max())
        for column in DECISION_COLUMNS:
            row[f"{column}_Min"], row[f"{column}_Max"] = float(all_seed[column].min()), float(all_seed[column].max())
        rows.append(row)
    output = stamp(pd.DataFrame(rows), relative_project_path(FINAL_ROOT / "01_Current" / "current_nsga3_convergence_audit.csv"))
    write_csv(output, FRONTIER_DIR / "frontier_seed_convergence.csv", ["Seed"])
    return output



def stable_pick(frame: pd.DataFrame, score: pd.Series, maximize: bool) -> pd.Series:
    """按 score 与固定 lexicographic tie-break 选择一行。"""
    work = frame.copy()
    work["__score"] = score.to_numpy(dtype=float)
    columns = ["__score", *DECISION_COLUMNS, "Policy_ID"]
    ascending = [not maximize, True, True, True, True, True]
    return work.sort_values(columns, ascending=ascending, kind="mergesort").iloc[0]



def select_representatives(frontier: pd.DataFrame) -> pd.DataFrame:
    """按附件规则确定性选择代表政策。"""
    normalized = normalized_matrix(frontier, SIGNED_OBJECTIVES)
    ideal_distance = np.linalg.norm(1.0 - normalized, axis=1)
    joint = frontier.loc[(frontier[list(SIGNED_OBJECTIVES)] > 0.0).all(axis=1)].copy()
    selections: list[tuple[str, str, pd.Series]] = []
    for label, rule, column in (
        ("carbon_leading", "maximum Carbon_Reduction_Signed", "Carbon_Reduction_Signed"),
        ("employment_leading", "maximum Employment_Gain_Signed", "Employment_Gain_Signed"),
        ("wage_leading", "maximum Wage_Gain_Signed", "Wage_Gain_Signed"),
        ("inequality_leading", "maximum Inequality_Reduction_Signed", "Inequality_Reduction_Signed"),
    ):
        selections.append((label, rule, stable_pick(frontier, frontier[column], maximize=True)))
    selections.append(("balanced_ideal_distance", "minimum normalized Euclidean distance to the four-objective ideal point", stable_pick(frontier, pd.Series(ideal_distance, index=frontier.index), maximize=False)))
    selections.append(("max_net_benefit", "maximum Net_Benefit_Signed", stable_pick(frontier, frontier["Net_Benefit_Signed"], maximize=True)))
    if not joint.empty:
        selections.append(("lowest_policy_intensity", "minimum Policy_Cost_Index among joint-improvement frontier points", stable_pick(joint, joint["Policy_Cost_Index"], maximize=False)))
    rows = []
    for label, rule, row in selections:
        rows.append({
            "Representative_Label": label,
            "Selection_Rule": rule + "; ties: tax_rate, renewable_share, shift_share, training_support_level, Policy_ID",
            "Candidate_ID": str(row["Policy_ID"]), "Seed": int(row["Seed"]),
            "Carbon_Price": float(row["tax_rate"]), "Renewable_Transition": float(row["renewable_share"]),
            "Demand_Shift": float(row["shift_share"]), "Transition_Support": float(row["training_support_level"]),
            "Carbon_Reduction": float(row["Carbon_Reduction_Signed"]), "Employment_Gain": float(row["Employment_Gain_Signed"]),
            "Skill_Resolved_Wage_Gain": float(row["Wage_Gain_Signed"]), "Inequality_Reduction": float(row["Inequality_Reduction_Signed"]),
            "D_H": float(row["Hours_Change"]), "D_H_Per_E": float(row["Hours_Per_Worker_Change"]),
            "D_W_Per_H": float(row["Wage_Per_Hour_Change"]), "Net_Benefit_Signed": float(row["Net_Benefit_Signed"]),
            "Policy_Cost_Index": float(row["Policy_Cost_Index"]),
        })
    return stamp(pd.DataFrame(rows), relative_project_path(FINAL_ROOT / "01_Current" / "current_nsga3_final_frontier.csv"))



def build_representatives(frontier: pd.DataFrame) -> pd.DataFrame:
    """写出可复现代表政策点。"""
    output = select_representatives(frontier)
    write_csv(output, FRONTIER_DIR / "current_frontier_representative_points.csv")
    return output



def build_frontier_structure(frontier: pd.DataFrame, duplicate_audit: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """量化前沿相关性、PCA、近最优比例与取值范围。"""
    rows: list[Dict[str, object]] = []
    for block, columns in (("objective_correlation", SIGNED_OBJECTIVES), ("decision_correlation", DECISION_COLUMNS)):
        correlation = frontier[list(columns)].corr(method="pearson")
        for left, right in itertools.combinations(columns, 2):
            rows.append({"Block": block, "Metric": "Pearson_r", "Variable_X": left, "Variable_Y": right, "Value": float(correlation.loc[left, right]), "Unit": "correlation", "Notes": "frontier-only"})
    normalized = normalized_matrix(frontier, SIGNED_OBJECTIVES)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    explained = variance / variance.sum() if variance.sum() > 0.0 else np.zeros(4)
    cumulative = np.cumsum(explained)
    for index, value in enumerate(explained, start=1):
        rows.append({"Block": "objective_pca", "Metric": f"PC{index}_Explained_Variance", "Variable_X": f"PC{index}", "Variable_Y": "", "Value": float(value), "Unit": "share", "Notes": f"cumulative={cumulative[index-1]:.12g}"})
    positive_net = float((frontier["Net_Benefit_Signed"] > 0.0).mean())
    rows.append({"Block": "net_benefit", "Metric": "Proportion_Positive_Net_Benefit", "Variable_X": "Net_Benefit_Signed", "Variable_Y": "", "Value": positive_net, "Unit": "share", "Notes": "frontier-only"})
    for column in SIGNED_OBJECTIVES:
        maximum, minimum = float(frontier[column].max()), float(frontier[column].min())
        rows.extend([
            {"Block": "frontier_range", "Metric": "Minimum", "Variable_X": column, "Variable_Y": "", "Value": minimum, "Unit": "signed share or index points", "Notes": "frontier-only"},
            {"Block": "frontier_range", "Metric": "Maximum", "Variable_X": column, "Variable_Y": "", "Value": maximum, "Unit": "signed share or index points", "Notes": "frontier-only"},
        ])
        for percent in (1, 2, 5):
            threshold = maximum - abs(maximum) * percent / 100.0
            rows.append({"Block": "near_objective_maximum", "Metric": f"Within_{percent}pct_of_Max", "Variable_X": column, "Variable_Y": "", "Value": float((frontier[column] >= threshold).mean()), "Unit": "share", "Notes": f"threshold={threshold:.12g}"})
    near = duplicate_audit.loc[
        duplicate_audit["Universe"].eq("strict_841_frontier")
        & duplicate_audit["Vector_Type"].eq("objective")
        & np.isclose(duplicate_audit["Tolerance"], 1e-3)
    ]
    near_share = float(near["N_Rows_In_Multiple"].iloc[0] / len(frontier)) if len(near) else 0.0
    low_dimensional, dense = bool(cumulative[1] >= 0.95), bool(len(frontier) >= 500 and near_share >= 0.10)
    assessment = "yes" if low_dimensional and dense else "partly" if low_dimensional or dense else "no"
    rows.append({"Block": "manifold_assessment", "Metric": "Effectively_Dense_Low_Dimensional_Tradeoff_Manifold", "Variable_X": "", "Variable_Y": "", "Value": assessment, "Unit": "categorical", "Notes": f"PC1+PC2={cumulative[1]:.6f}; objective-near-duplicate-share@1e-3={near_share:.6f}; N={len(frontier)}"})
    output = stamp(pd.DataFrame(rows), relative_project_path(FINAL_ROOT / "01_Current" / "current_nsga3_final_frontier.csv"))
    write_csv(output, FRONTIER_DIR / "current_frontier_structure_summary.csv", ["Block", "Metric", "Variable_X", "Variable_Y"])
    report = f"""# D06 current frontier structure report

日期：{CREATED_AT}
执行者：{EXECUTOR}
规范：`{SPECIFICATION_ID}`

==========

## Assessment

The {len(frontier)}-point strict frontier is assessed as **{assessment}** with respect to being an effectively dense low-dimensional trade-off manifold.

- PC1 + PC2 explained variance: `{cumulative[1]:.6f}`.
- Objective near-duplicate membership at normalized tolerance 1e-3: `{near_share:.6f}`.
- Frontier size: `{len(frontier)}` exact non-dominated policies.
- Positive net-benefit share: `{positive_net:.6f}`.

Policy outcomes are evaluated across the four objectives.
"""
    (FRONTIER_DIR / "current_frontier_structure_report.md").write_text(report, encoding="utf-8")
    return output, assessment



def build_core_scenario_outcomes(engine) -> Dict[str, Mapping[str, object]]:
    """只运行 S0-S5 exact state algebra，返回固定 mask 所需部门数组。"""
    if not engine.baseline:
        engine.compute_baseline()
    base_employment = np.asarray(engine.employment["data"], dtype=float)
    base_wage = np.asarray(engine.wage["data"], dtype=float)
    base_co2 = np.asarray(engine.co2_emissions, dtype=float)
    base_y = np.asarray(engine.Y, dtype=float)
    outcomes: Dict[str, Mapping[str, object]] = {"S0_baseline": engine.baseline}

    s1_y = engine._apply_carbon_tax_y_shock(
        base_Y=base_y, tax_rate=100.0, theta_tax=0.8,
        epsilon_highcarbon=-0.5, substitution_share=0.7,
    )
    outcomes["S1_carbon_tax"] = engine.compute_outcomes_exact_fast(s1_y, base_co2, base_employment, base_wage)

    s2_co2, s2_y, s2_employment = engine._apply_energy_transition_shocks(
        base_co2=base_co2, base_Y=base_y, base_E=base_employment,
        renewable_share=0.60, eta_renewable_co2=0.75,
        lambda_green_jobs=1.25, investment_boost=0.20,
    )
    outcomes["S2_energy_transition"] = engine.compute_outcomes_exact_fast(s2_y, s2_co2, s2_employment, base_wage)

    s3_y = engine._apply_consumption_shift_y_shock(
        base_Y=base_y, shift_share=0.20, substitution_share=1.0,
    )
    outcomes["S3_consumption_shift"] = engine.compute_outcomes_exact_fast(s3_y, base_co2, base_employment, base_wage)

    s4_employment, s4_wage = engine._apply_just_transition_shocks(
        base_E=base_employment, base_W=base_wage,
        training_support_level=0.30, wage_support_strength=0.15,
    )
    outcomes["S4_just_transition"] = engine.compute_outcomes_exact_fast(base_y, base_co2, s4_employment, s4_wage)

    s5_y = engine._apply_carbon_tax_y_shock(
        base_Y=base_y, tax_rate=100.0, theta_tax=0.8,
        epsilon_highcarbon=-0.5, substitution_share=0.7,
    )
    s5_co2, s5_y, s5_employment = engine._apply_energy_transition_shocks(
        base_co2=base_co2, base_Y=s5_y, base_E=base_employment,
        renewable_share=0.60, eta_renewable_co2=0.75,
        lambda_green_jobs=1.25, investment_boost=0.20,
    )
    s5_y = engine._apply_consumption_shift_y_shock(
        base_Y=s5_y, shift_share=0.20, substitution_share=1.0,
    )
    s5_employment, s5_wage = engine._apply_just_transition_shocks(
        base_E=s5_employment, base_W=base_wage,
        training_support_level=0.30, wage_support_strength=0.15,
    )
    outcomes["S5_policy_package"] = engine.compute_outcomes_exact_fast(
        s5_y, s5_co2, s5_employment, s5_wage,
    )
    return outcomes



def sector_identifiers(engine, n_sectors: int) -> tuple[list[str], list[str]]:
    """构造 region::product 稳定 ID 和可读 product label。"""
    labels = list(getattr(engine, "row_labels", []))
    if len(labels) != n_sectors:
        labels = [f"sector_{index}" for index in range(n_sectors)]
    regions = list(getattr(engine, "regions", []))
    n_products = int(getattr(engine, "n_products", n_sectors))
    ids = []
    for index, label in enumerate(labels):
        region_index = index // n_products
        region = str(regions[region_index]) if region_index < len(regions) else "GLOBAL"
        ids.append(f"{region}::{label}")
    return ids, labels



def build_fixed_hour_lock(engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """用 S0 固定 mask 重算 S0-S5 skill-hour lock。"""
    outcomes = build_core_scenario_outcomes(engine)
    baseline = outcomes["S0_baseline"]
    baseline_carbon = np.clip(np.asarray(baseline["Carbon_By_Producer"], dtype=float), 0.0, None)
    positive = baseline_carbon[baseline_carbon > 0.0]
    if positive.size == 0:
        raise RuntimeError("BASELINE_HIGH_CARBON_MASK_EMPTY")
    threshold = float(np.quantile(positive, 0.80))
    mask = (baseline_carbon > 0.0) & (baseline_carbon >= threshold)
    sector_ids, sector_names = sector_identifiers(engine, len(mask))
    sector_set = stamp(pd.DataFrame({
        "Sector_ID": sector_ids,
        "Sector_Name": sector_names,
        "Baseline_Direct_Carbon": baseline_carbon,
        "Threshold": threshold,
        "Included": mask,
        "Baseline_High_Carbon_Sector_Set": mask,
    }), "exact S0 Carbon_By_Producer")
    write_csv(sector_set, HOUR_DIR / "baseline_high_carbon_sector_set.csv", ["Sector_ID"])

    dynamic = pd.read_csv(CURRENT_READ_DIR / "current_labour_hour_diagnostics_final.csv")
    dynamic = dynamic.set_index("Scenario", drop=False)
    baseline_fixed: Dict[str, float] = {}
    base_hours_matrix = np.asarray(baseline["Hours_By_Skill_And_Sector"], dtype=float)
    for skill_index, skill in enumerate(SKILLS):
        denominator = float(base_hours_matrix[skill_index, :].sum())
        baseline_fixed[skill] = float(base_hours_matrix[skill_index, mask].sum()) / denominator

    rows, audit_rows = [], []
    core_reference = pd.read_csv(CURRENT_READ_DIR / "current_core_scenarios_final.csv").set_index("Scenario")
    for scenario, scenario_outcomes in outcomes.items():
        hours_matrix = np.asarray(scenario_outcomes["Hours_By_Skill_And_Sector"], dtype=float)
        scenario_total = float(scenario_outcomes["Hours_Total"])
        audit_rows.append({
            "Audit_Check": f"{scenario}_skill_hours_sum",
            "Observed": float(hours_matrix.sum()), "Expected": scenario_total,
            "Passed": bool(np.isclose(hours_matrix.sum(), scenario_total, rtol=1e-12, atol=1e-6)),
            "Interpretation": "skill-sector hours sum to scenario total",
        })
        reference = core_reference.loc[scenario]
        for field in ("Carbon_Footprint", "Employment_Total", "Wage_Total", "Hours_Total"):
            observed = float(scenario_outcomes[field])
            expected = float(reference[field])
            audit_rows.append({
                "Audit_Check": f"{scenario}_{field}_physical_regression",
                "Observed": observed, "Expected": expected,
                "Passed": bool(np.isclose(observed, expected, rtol=1e-10, atol=1e-5)),
                "Interpretation": "six-state post-processing preserves authoritative physical result",
            })
        for skill_index, skill in enumerate(SKILLS):
            skill_hours = hours_matrix[skill_index, :]
            denominator = float(skill_hours.sum())
            numerator = float(skill_hours[mask].sum())
            policy_lock = numerator / denominator if denominator > 0.0 else np.nan
            reduction = baseline_fixed[skill] - policy_lock
            dynamic_lock = float(dynamic.loc[scenario, f"Hours_Lock_{skill}"])
            dynamic_reduction = float(dynamic.loc[scenario, f"Hours_Lock_Reduction_{skill}"])
            interpretation = (
                "fixed-set high-carbon hour lock declines" if reduction > 0.0
                else "fixed-set high-carbon hour lock increases" if reduction < 0.0
                else "no fixed-set high-carbon hour lock change"
            )
            rows.append({
                "Scenario": scenario, "Skill": skill,
                "Baseline_High_Carbon_Sector_Set": "S0 positive total direct carbon >= 80th percentile",
                "High_Carbon_Sector_Count": int(mask.sum()),
                "Threshold": threshold, "Hours_In_Fixed_Set": numerator,
                "Hours_Total_By_Skill": denominator, "Hours_Unit": "M.hours",
                "Baseline_Fixed_Lock": baseline_fixed[skill], "Policy_Fixed_Lock": policy_lock,
                "Fixed_Lock_Reduction": reduction, "Dynamic_Set_Lock": dynamic_lock,
                "Dynamic_Set_Baseline_Minus_Policy": dynamic_reduction,
                "Interpretation": interpretation,
            })
    audit_rows.extend([
        {"Audit_Check": "all_skill_denominators_positive", "Observed": int(sum(float(row["Hours_Total_By_Skill"]) > 0.0 for row in rows)), "Expected": len(rows), "Passed": all(float(row["Hours_Total_By_Skill"]) > 0.0 for row in rows), "Interpretation": "all lock denominators are positive"},
        {"Audit_Check": "S0_fixed_reduction_zero", "Observed": max(abs(float(row["Fixed_Lock_Reduction"])) for row in rows if row["Scenario"] == "S0_baseline"), "Expected": 0.0, "Passed": all(np.isclose(float(row["Fixed_Lock_Reduction"]), 0.0, atol=1e-14) for row in rows if row["Scenario"] == "S0_baseline"), "Interpretation": "baseline minus baseline equals zero"},
        {"Audit_Check": "hours_units_explicit", "Observed": sorted(set(str(row["Hours_Unit"]) for row in rows)), "Expected": "M.hours", "Passed": set(str(row["Hours_Unit"]) for row in rows) == {"M.hours"}, "Interpretation": "hours unit is not mixed"},
    ])
    detail = stamp(pd.DataFrame(rows), "S0-S5 exact scenario arrays; current dynamic hour diagnostics")
    summary_columns = [
        "Scenario", "Skill", "Baseline_Fixed_Lock", "Policy_Fixed_Lock", "Fixed_Lock_Reduction",
        "Dynamic_Set_Lock", "Dynamic_Set_Baseline_Minus_Policy", "Interpretation",
        "High_Carbon_Sector_Count", "Threshold", "Hours_Unit",
    ]
    summary = stamp(detail[summary_columns], "04_Summaries/policy_hour_lock_fixed_baseline.csv")
    audit = stamp(pd.DataFrame(audit_rows), "fixed S0 mask and computed core scenario tables")
    write_csv(detail, HOUR_DIR / "policy_hour_lock_fixed_baseline.csv", ["Scenario", "Skill"])
    write_csv(summary, HOUR_DIR / "policy_hour_lock_fixed_baseline_summary.csv", ["Scenario", "Skill"])
    write_csv(audit, HOUR_DIR / "policy_hour_lock_dynamic_vs_fixed_audit.csv", ["Audit_Check"])
    if not audit["Passed"].all():
        failed = audit.loc[~audit["Passed"], "Audit_Check"].tolist()
        raise AssertionError(f"FIXED_HOUR_LOCK_AUDIT_FAILED: {failed}")
    return detail, summary, audit



def build_future_scope_inventory() -> pd.DataFrame:
    """列出每个 world、scope、seed 的评价量与前沿量。"""
    all_candidates = pd.read_csv(FUTURE_READ_DIR / "future_nsga3_all_candidates_final.csv")
    frontiers = pd.read_csv(FUTURE_READ_DIR / "future_nsga3_frontiers_final.csv")
    group_columns = ["Pathway", "Year", "Future_NSGA_Scope", "Seed", "NSGA3_Population_Size", "NSGA3_Generations"]
    candidate_counts = all_candidates.groupby(group_columns, dropna=False, sort=False).size().rename("Number_of_Evaluations").reset_index()
    frontier_counts = frontiers.groupby(group_columns, dropna=False, sort=False).size().rename("Frontier_Size").reset_index()
    inventory = candidate_counts.merge(frontier_counts, on=group_columns, how="outer").fillna({"Number_of_Evaluations": 0, "Frontier_Size": 0})
    inventory = inventory.rename(columns={
        "Future_NSGA_Scope": "Scope", "NSGA3_Population_Size": "Population", "NSGA3_Generations": "Generations",
    })
    inventory["Union_or_Seed"] = np.where(inventory["Scope"].astype(str).str.contains("union", case=False), "union", "seed")
    inventory["Source_File"] = relative_project_path(FINAL_ROOT / "02_Future" / "future_nsga3_frontiers_final.csv")
    inventory["Number_of_Evaluations"] = inventory["Number_of_Evaluations"].astype(int)
    inventory["Frontier_Size"] = inventory["Frontier_Size"].astype(int)
    output = stamp(inventory[[
        "Pathway", "Year", "Seed", "Population", "Generations", "Union_or_Seed",
        "Number_of_Evaluations", "Frontier_Size", "Scope", "Source_File",
    ]], "future exact all-candidate and frontier files")
    write_csv(output, FUTURE_DIR / "future_frontier_scope_inventory.csv", ["Pathway", "Year", "Generations", "Scope", "Seed"])
    return output



def select_future_authoritative_scopes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个 key world 选择最高代数的 authoritative frontier。"""
    frontiers = pd.read_csv(FUTURE_READ_DIR / "future_nsga3_frontiers_final.csv")
    convergence = pd.read_csv(FUTURE_READ_DIR / "future_nsga3_selected_world_convergence.csv")
    scope_summary = frontiers.groupby(["Pathway", "Year", "Future_NSGA_Scope"], sort=False).agg(
        Generations=("NSGA3_Generations", "max"),
        Frontier_Size=("Policy_ID", "size"),
        Seed_Count=("Seed", "nunique"),
    ).reset_index()
    scope_summary["Is_Union"] = scope_summary["Future_NSGA_Scope"].astype(str).str.contains("union", case=False)
    rows, selected_parts = [], []
    for (pathway, year), group in scope_summary.groupby(["Pathway", "Year"], sort=True):
        ordered = group.sort_values(
            ["Generations", "Is_Union", "Frontier_Size", "Future_NSGA_Scope"],
            ascending=[False, False, False, True], kind="mergesort",
        )
        chosen = ordered.iloc[0]
        scope = str(chosen["Future_NSGA_Scope"])
        selected = frontiers.loc[
            frontiers["Pathway"].eq(pathway) & frontiers["Year"].astype(int).eq(int(year))
            & frontiers["Future_NSGA_Scope"].eq(scope)
        ].copy()
        selected_parts.append(selected)
        world = f"{pathway}-{int(year)}"
        match = convergence.loc[
            convergence["World"].astype(str).eq(world)
            & convergence["Generations"].astype(int).eq(int(chosen["Generations"]))
        ]
        if not match.empty:
            convergence_status = "passed" if bool(match["Passed"].all()) else "failed"
        else:
            convergence_status = "single_seed_g15_authoritative_available"
        if bool(chosen["Is_Union"]):
            rule = "highest-generation multi-seed union frontier"
        else:
            rule = "highest-generation authoritative frontier available; no union scope available"
        rows.append({
            "Pathway": pathway, "Year": int(year), "Selected_Scope": scope,
            "Selection_Rule": rule,
            "Source_File": relative_project_path(FINAL_ROOT / "02_Future" / "future_nsga3_frontiers_final.csv"),
            "Frontier_Size": len(selected), "Convergence_Status": convergence_status,
            "Generations": int(chosen["Generations"]),
        })
    scopes = stamp(pd.DataFrame(rows), "future exact frontier scopes and selected-world convergence")
    selected_frontiers = pd.concat(selected_parts, ignore_index=True)
    write_csv(scopes, FUTURE_DIR / "future_authoritative_frontier_scope.csv", ["Pathway", "Year"])
    return scopes, selected_frontiers



def build_future_policy_bands(selected_frontiers: pd.DataFrame) -> pd.DataFrame:
    """按 key world 重算 non-dominated solution policy quantiles。"""
    parameter_map = {
        "tax_rate": ("Carbon_Price", 50.0, 150.0),
        "renewable_share": ("Renewable_Transition", 0.30, 0.90),
        "shift_share": ("Demand_Shift", 0.10, 0.30),
        "training_support_level": ("Transition_Support", 0.00, 0.60),
    }
    rows = []
    for (pathway, year), group in selected_frontiers.groupby(["Pathway", "Year"], sort=True):
        scope = str(group["Future_NSGA_Scope"].iloc[0])
        for parameter, (label, lower, upper) in parameter_map.items():
            quantiles = group[parameter].quantile([0.05, 0.25, 0.50, 0.75, 0.95])
            rows.append({
                "Pathway": pathway, "Year": int(year), "Selected_Scope": scope,
                "Parameter": parameter, "Parameter_Label": label,
                "Feasible_Lower": lower, "Feasible_Upper": upper,
                "P05": float(quantiles.loc[0.05]), "P25": float(quantiles.loc[0.25]),
                "P50": float(quantiles.loc[0.50]), "P75": float(quantiles.loc[0.75]),
                "P95": float(quantiles.loc[0.95]), "N_Frontier_Points": len(group),
                "Terminology": "Frontier-consistent policy bands",
            })
    output = stamp(pd.DataFrame(rows), "04_Summaries/future_authoritative_frontier_scope.csv; 02_Future/future_nsga3_frontiers_final.csv")
    write_csv(output, FUTURE_DIR / "future_frontier_policy_bands_final.csv", ["Pathway", "Year", "Parameter"])
    return output



def build_future_probability_summary() -> pd.DataFrame:
    """整合 12-world signed probabilities 与四目标中位数。"""
    future_root = FUTURE_READ_DIR
    probability = pd.read_csv(future_root / "future_monte_carlo_signed_probability_12worlds_final.csv")
    summary = pd.read_csv(future_root / "future_monte_carlo_summary_12worlds_final.csv")
    medians = summary.pivot_table(index=["Pathway", "Year"], columns="Metric", values="P50", aggfunc="first").reset_index()
    output = probability.merge(medians, on=["Pathway", "Year"], how="left")
    output = output.rename(columns={
        "P_All_Four_Signed_Objectives_Positive": "P_All_Four_Positive",
        "P_Inequality_Reduction_Signed_Positive": "P_Inequality_Positive",
        "P_Net_Benefit_Signed_Positive": "P_Net_Benefit_Positive",
        "Carbon_Reduction_Signed": "Carbon_Median",
        "Employment_Gain_Signed": "Employment_Median",
        "Wage_Gain_Signed": "Wage_Median",
        "Inequality_Reduction_Signed": "Inequality_Median",
    })
    required = [
        "Pathway", "Year", "P_All_Four_Positive", "P_Inequality_Positive", "P_Net_Benefit_Positive",
        "Carbon_Median", "Employment_Median", "Wage_Median", "Inequality_Median",
    ]
    result = stamp(output[required], "future 12-world Monte Carlo probability and summary tables")
    write_csv(result, FUTURE_DIR / "future_probability_summary_final.csv", ["Pathway", "Year"])
    return result



def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    """计算两个非负 share vectors 的 Jensen-Shannon divergence。"""
    p = np.clip(np.asarray(left, dtype=float), 0.0, None)
    q = np.clip(np.asarray(right, dtype=float), 0.0, None)
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)
    positive_p, positive_q = p > 0.0, q > 0.0
    kl_p = np.sum(p[positive_p] * np.log(p[positive_p] / midpoint[positive_p]))
    kl_q = np.sum(q[positive_q] * np.log(q[positive_q] / midpoint[positive_q]))
    return float(0.5 * (kl_p + kl_q))



def build_network_overlap_jsd() -> pd.DataFrame:
    """由现有 bilateral transfer tables 补充 network overlap JSD 证据。"""
    network_dir = FINAL_ROOT / "02_Network" / "csv"
    metric_columns = {
        "carbon": "Carbon_Transfer",
        "employment": "Employment_Transfer",
        "hours": "Hours_Transfer",
        "wage": "Wage_Transfer",
    }
    frames = {}
    keys = ["From_Region", "To_Region", "Skill"]
    for metric, value_column in metric_columns.items():
        path = network_dir / f"network_bilateral_{metric}_transfer_by_skill.csv"
        frame = pd.read_csv(path)[keys + [value_column]].rename(columns={value_column: metric})
        frames[metric] = frame
    merged = None
    for metric, frame in frames.items():
        merged = frame if merged is None else merged.merge(frame, on=keys, how="outer")
    assert merged is not None
    merged = merged.fillna(0.0)
    rows = []
    for level, level_frame in (
        ("country_country_skill", merged),
        ("country_country", merged.groupby(["From_Region", "To_Region"], as_index=False)[list(metric_columns)].sum()),
    ):
        for left, right in itertools.combinations(metric_columns, 2):
            rows.append({
                "Level": level, "Network_A": left, "Network_B": right,
                "Jensen_Shannon_Divergence": jensen_shannon_divergence(level_frame[left], level_frame[right]),
                "Edge_Union_Count": len(level_frame),
                "Interpretation": "network-share divergence diagnostic; not a policy inequality objective",
            })
    output = stamp(pd.DataFrame(rows), "02_Network/network_bilateral_*_transfer_by_skill.csv")
    write_csv(output, EVIDENCE_DIR / "network_overlap_jsd_publication_patch.csv", ["Level", "Network_A", "Network_B"])
    return output


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Summarize previously generated current or future policy tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=FINAL_ROOT)
    parser.add_argument("--stage", choices=("current-frontier", "future-frontiers", "future-probabilities", "network"), required=True)
    args = parser.parse_args(argv)
    configure_paths(args.output_root)
    if args.stage == "current-frontier":
        candidates = pd.read_csv(CURRENT_READ_DIR / "current_nsga3_union_all_candidates.csv")
        frontier = pd.read_csv(CURRENT_READ_DIR / "current_nsga3_final_frontier.csv")
        build_frontier_tolerance_sensitivity(candidates, frontier)
        duplicate_audit, _ = build_duplicate_audit(candidates, frontier)
        build_seed_convergence(candidates, frontier)
        build_representatives(frontier)
        build_frontier_structure(frontier, duplicate_audit)
    elif args.stage == "future-frontiers":
        build_future_scope_inventory()
        _, selected = select_future_authoritative_scopes()
        build_future_policy_bands(selected)
    elif args.stage == "future-probabilities":
        build_future_probability_summary()
    else:
        build_network_overlap_jsd()
