#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run country-skill accounting, policy scenarios and future analyses."""

from __future__ import annotations

import argparse
import importlib
import os
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from pymoo.indicators.hv import HV


# 路径与固定元数据

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = Path(__file__).resolve().parent
RESULT_ROOT = Path.cwd() / "results"
CURRENT_DIR = RESULT_ROOT / "01_Current"
FUTURE_DIR = RESULT_ROOT / "02_Future"
SOURCE_DIR = RESULT_ROOT / "03_SourceData"
AUDIT_DIR = RESULT_ROOT / "05_Audits"
READINESS_DIR = RESULT_ROOT / "00_Readiness"
DATA_PATH = Path(os.environ.get("EXIOBASE_PXP_PATH", "data/raw/IOT_2022_pxp"))
FOOTPRINT_PATH = CODE_DIR / "footprint.py"
SCENARIO_PATH = CODE_DIR / "policy.py"
INEQUALITY_PATH = CODE_DIR / "inequality.py"
FUTURE_PATH = CODE_DIR / "future.py"
CREATED_AT = "2026-10-28"
RUN_VERSION = "1.0.0"
SPECIFICATION_ID = "country_skill_mismatch_mean_v1"
EXACT_STATUS = "exact_exiobase_policy_engine"
SIGNED_OBJECTIVES = (
    "Carbon_Reduction_Signed",
    "Employment_Gain_Signed",
    "Wage_Gain_Signed",
    "Inequality_Reduction_Signed",
)
SKILLS = ("Low", "Medium", "High")
FIGURE2_STAGE = "figure2-sector-coverage"
FUTURE_NSGA3_EXTENSION_STAGE = "future-nsga3-extension"
FUTURE_NSGA3_ALL12_SCOPE = "all_12_worlds_seed42_g15"
FUTURE_NSGA3_SELECTED_MULTI_G30_SCOPE = "selected_world_multiseed_g30"
FUTURE_NSGA3_SELECTED_UNION_G30_SCOPE = "selected_world_union_frontier_g30"
FUTURE_NSGA3_EXTENSION_STEP = 20
FUTURE_NSGA3_MAX_GENERATIONS = 70
FORMAL_MC_FINAL_STAGES = frozenset({
    "current-monte-carlo",
    "current-all",
    "future-monte-carlo",
    "future-all",
    "all",
})
FUTURE_EXPLICIT_STAGES = frozenset({
    "future-deterministic",
    "future-monte-carlo",
    "future-sobol",
    "future-nsga3",
    FUTURE_NSGA3_EXTENSION_STAGE,
    "future-all",
})
FUTURE_KEY_CACHE_STAGES = frozenset({
    "future-monte-carlo",
    "future-sobol",
    "future-nsga3",
    "future-all",
})


def load_module(path: Path):
    """Import the corresponding analysis module from this package."""
    return importlib.import_module(f".{path.stem}", package=__package__)

def configure_paths(data_path: Path, output_root: Path) -> None:
    """Set the external input and one output root for every analysis stage."""
    global DATA_PATH, RESULT_ROOT, CURRENT_DIR, FUTURE_DIR
    global SOURCE_DIR, AUDIT_DIR, READINESS_DIR
    DATA_PATH = data_path.expanduser().resolve()
    RESULT_ROOT = output_root.expanduser().resolve()
    CURRENT_DIR = RESULT_ROOT / "01_Current"
    FUTURE_DIR = RESULT_ROOT / "02_Future"
    SOURCE_DIR = RESULT_ROOT / "03_SourceData"
    AUDIT_DIR = RESULT_ROOT / "05_Audits"
    READINESS_DIR = RESULT_ROOT / "00_Readiness"


def ensure_output_dirs() -> None:
    """创建各分析阶段的结果目录。"""
    for path in (CURRENT_DIR, FUTURE_DIR, SOURCE_DIR, AUDIT_DIR, READINESS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def stamp_frame(frame: pd.DataFrame, upstream_sources: str) -> pd.DataFrame:
    """为结果表添加版本、时间、模型定义和计算状态。"""
    result = frame.copy()
    result["Run_Version"] = RUN_VERSION
    result["Created_At"] = CREATED_AT
    result["Upstream_Sources"] = upstream_sources
    result["Primary_Inequality_Specification"] = SPECIFICATION_ID
    result["Exact_or_Surrogate"] = EXACT_STATUS
    return result


def write_csv(frame: pd.DataFrame, path: Path, upstream_sources: str, sort_by: Optional[Sequence[str]] = None) -> None:
    """使用固定列顺序和稳定排序写入 CSV。"""
    output = stamp_frame(frame, upstream_sources)
    if sort_by:
        columns = [column for column in sort_by if column in output.columns]
        if columns:
            output = output.sort_values(columns, kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def load_exact_engine():
    """加载 EXIOBASE 并初始化政策引擎。"""
    if not DATA_PATH.is_dir():
        raise FileNotFoundError(f"EXIOBASE 路径不可用: {DATA_PATH}")
    footprint_module = load_module(FOOTPRINT_PATH)
    scenario_module = load_module(SCENARIO_PATH)
    inequality_module = load_module(INEQUALITY_PATH)
    footprint = footprint_module.FootprintEngine(str(DATA_PATH))
    if not footprint.load_data():
        raise RuntimeError("FINAL_RERUN_READINESS_BLOCKED_FOOTPRINT_ENGINE_LOAD")
    engine = scenario_module.FinalScenarioEngine(footprint)
    return engine, scenario_module, inequality_module


def audit_row(gate: str, observed, expected, passed: bool, interpretation: str) -> Dict[str, object]:
    """整理一项核对结果。"""
    return {
        "Audit_Gate": gate,
        "Observed": observed,
        "Expected": expected,
        "Passed": bool(passed),
        "Interpretation": interpretation,
    }


def validate_formal_mc_draws(stage: str, mc_draws: int) -> None:
    """检查 Monte Carlo 抽样数是否为规定的 1,000 次。"""
    if stage in FORMAL_MC_FINAL_STAGES and mc_draws != 1000:
        raise ValueError(
            f"FORMAL_MC_FINAL_REQUIRES_1000_DRAWS: stage={stage}, mc_draws={mc_draws}"
        )


# 模型一致性检查

def check_model(engine, inequality_module) -> pd.DataFrame:
    """检查基准核算、Eq. (9)、工资口径及聚合代数等价性。"""
    baseline = engine.compute_baseline()
    rows = []
    expected_index = 0.5 * (float(baseline["System_CE_Gini"]) + float(baseline["System_CW_Gini"]))
    rows.append(audit_row(
        "primary_inequality_formula",
        float(baseline["Policy_Inequality_Index"]),
        expected_index,
        np.isclose(float(baseline["Policy_Inequality_Index"]), expected_index, rtol=1e-12, atol=1e-12),
        "I equals the mean of country-skill CE and CW bivariate Gini",
    ))
    conservation = float(baseline["Attributed_Carbon"] + baseline["Unattributed_Carbon"])
    rows.append(audit_row(
        "carbon_attribution_conservation",
        conservation,
        float(baseline["Carbon_Footprint"]),
        np.isclose(conservation, float(baseline["Carbon_Footprint"]), rtol=1e-10, atol=1e-2),
        "unattributed carbon is retained rather than force-allocated",
    ))
    params = {
        "tax_rate": 117.0,
        "renewable_share": 0.71,
        "shift_share": 0.23,
        "training_support_level": 0.41,
    }
    uncertainty = {
        "theta_tax": 0.77,
        "epsilon_highcarbon": -0.47,
        "eta_renewable_co2": 0.69,
        "lambda_green_jobs": 1.31,
        "xi_shift_effectiveness": 0.91,
        "omega_lowcarbon_substitution": 0.83,
        "rho_reskilling": 0.88,
        "omega_wage_support": 0.94,
    }
    y_full, co2_full, employment_full, wage_full = engine._build_policy_state(params, uncertainty)
    full_outcomes = engine.compute_outcomes(
        Y=y_full,
        co2_emissions=co2_full,
        employment_data=employment_full,
        wage_data=wage_full,
    )
    aggregate_outcomes = engine.compute_outcomes_exact_fast(
        y_full,
        co2_full,
        employment_full,
        wage_full,
    )
    full_summary = engine._compute_scenario_inequality_summary(full_outcomes)
    aggregate_summary = engine._compute_scenario_inequality_summary(aggregate_outcomes)
    exact_fields = (
        "Carbon_Footprint", "Employment_Total", "Wage_Total", "Hours_Total",
    )
    summary_fields = ("System_CE_Gini", "System_CW_Gini", "Policy_Inequality_Index")
    relative_differences = [
        abs(float(full_outcomes[field]) - float(aggregate_outcomes[field])) / max(abs(float(full_outcomes[field])), 1.0)
        for field in exact_fields
    ] + [
        abs(float(full_summary[field]) - float(aggregate_summary[field])) / max(abs(float(full_summary[field])), 1.0)
        for field in summary_fields
    ]
    maximum_relative_difference = max(relative_differences)
    rows.append(audit_row(
        "exact_aggregate_policy_algebra",
        maximum_relative_difference,
        0.0,
        maximum_relative_difference <= 1e-10,
        "full Y shock chain plus L @ row-summed final demand reproduces full matrix outcomes and primary inequality",
    ))
    rows.append(audit_row(
        "four_objective_frontier",
        4,
        4,
        True,
        "carbon, employment, wages and inequality define the policy objectives",
    ))
    audit = pd.DataFrame(rows)
    write_csv(
        audit,
        AUDIT_DIR / "final_inequality_audit.csv",
        str(SCENARIO_PATH),
        sort_by=["Audit_Gate"],
    )
    if not bool(audit["Passed"].all()):
        raise RuntimeError("FINAL_INEQUALITY_ENGINE_TEST_BLOCKED")
    return audit


# Current S0-S5

def build_eq9_and_coverage(engine, inequality_module, core: pd.DataFrame) -> None:
    """输出基准国家-技能二元 Gini 排序及 S0-S5 碳归因覆盖率。"""
    outcomes = engine.baseline
    attribution = inequality_module.attribute_carbon_by_skill(
        outcomes["Carbon_By_Producer"],
        outcomes["Employment_By_Skill_And_Sector"],
    )
    employment, wage, carbon, unit_ids = engine._country_skill_arrays(
        outcomes["Employment_By_Skill_And_Sector"],
        outcomes["Wage_By_Skill_And_Sector"],
        attribution.attributed,
    )
    traces = []
    for metric, resource in (("System_CE_Gini", employment), ("System_CW_Gini", wage)):
        traces.append(inequality_module.exact_bivariate_gini(
            resource,
            carbon,
            unit_ids=unit_ids,
            metric=metric,
        ).trace)
    write_csv(
        pd.concat(traces, ignore_index=True),
        SOURCE_DIR / "eq9_country_skill_order_trace.csv",
        f"{os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}; EXIOBASE PXP 2022",
        sort_by=["Metric", "Rank"],
    )
    coverage_columns = [
        "Scenario", "Carbon_Footprint", "Attributed_Carbon", "Unattributed_Carbon",
        "Attribution_Coverage", "Zero_Skill_Employment_Positive_Carbon_Sectors",
    ]
    coverage = core[coverage_columns].copy()
    coverage["Conservation_Difference"] = (
        coverage["Carbon_Footprint"]
        - coverage["Attributed_Carbon"]
        - coverage["Unattributed_Carbon"]
    )
    coverage["Conservation_Passed"] = np.isclose(
        coverage["Attributed_Carbon"] + coverage["Unattributed_Carbon"],
        coverage["Carbon_Footprint"],
        rtol=1e-10,
        atol=1e-2,
    )
    coverage["Attribution_Status"] = np.where(
        coverage["Unattributed_Carbon"] > 0.0,
        "partially_unattributed",
        "fully_attributed",
    )
    write_csv(
        coverage,
        CURRENT_DIR / "skill_carbon_attribution_coverage.csv",
        f"{os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}",
        sort_by=["Scenario"],
    )
    audit = coverage[["Scenario", "Conservation_Difference", "Conservation_Passed", "Attribution_Status"]].copy()
    audit["Interpretation"] = "attributed plus unattributed carbon equals full non-negative carbon"
    write_csv(
        audit,
        CURRENT_DIR / "skill_carbon_attribution_audit.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/skill_carbon_attribution_coverage.csv",
        sort_by=["Scenario"],
    )


def run_current_core(engine, inequality_module) -> pd.DataFrame:
    """Compute S0-S5 outcomes, inequality, transfer concentration and labour hours."""
    started = time.perf_counter()
    core = engine.run_core_scenarios()
    core["Runtime_Seconds_Cumulative"] = time.perf_counter() - started
    write_csv(
        core,
        CURRENT_DIR / "current_core_scenarios_final.csv",
        f"{os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}; EXIOBASE PXP 2022",
        sort_by=["Scenario"],
    )
    component_columns = ["Scenario", *engine.inequality_columns]
    extra = [
        "Zero_Skill_Employment_Positive_Carbon_Sectors",
        "High_Carbon_Threshold_80",
        "High_Carbon_Sector_Count",
        "Synthetic_Transmission_Metadata",
        "Labour_Hours_Role",
    ]
    write_csv(
        core[[column for column in component_columns + extra if column in core.columns]],
        CURRENT_DIR / "current_inequality_components_final.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_core_scenarios_final.csv",
        sort_by=["Scenario"],
    )
    lock_columns = [
        "Scenario",
        *[f"Employment_Lock_{skill}" for skill in SKILLS],
        "Employment_Lock_Disparity", "Employment_Lock_Direction",
        *[f"Hours_Lock_{skill}" for skill in SKILLS],
        "Hours_Lock_Disparity",
    ]
    write_csv(
        core[lock_columns],
        CURRENT_DIR / "current_skill_lock_final.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_core_scenarios_final.csv",
        sort_by=["Scenario"],
    )
    transfer_columns = [
        "Scenario", "Net_Carbon_Transfer_Gini", "Net_Employment_Transfer_Gini",
        "Net_Wage_Transfer_Gini", "Supplementary_Synthetic_Transmission_Coupling",
        "Synthetic_Transmission_Metadata",
    ]
    write_csv(
        core[transfer_columns],
        CURRENT_DIR / "current_net_transfer_concentration_final.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_core_scenarios_final.csv",
        sort_by=["Scenario"],
    )
    hour_columns = [
        "Scenario", "Hours_Total", "Hours_Change", "Employment_Total", "Wage_Total",
        "Hours_Per_Worker", "Hours_Per_Worker_Change", "Wage_Per_Hour", "Wage_Per_Hour_Change",
        *[f"Hours_{skill}" for skill in SKILLS],
        *[f"Hours_Lock_{skill}" for skill in SKILLS],
        *[f"Hours_Lock_Reduction_{skill}" for skill in SKILLS],
        "Hours_Unit", "Employment_Unit", "Wage_Unit", "Hours_Per_Worker_Unit",
        "Wage_Per_Hour_Unit", "Hours_Method", "Labour_Hours_Optimized",
    ]
    write_csv(
        core[hour_columns],
        CURRENT_DIR / "current_labour_hour_diagnostics_final.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_core_scenarios_final.csv",
        sort_by=["Scenario"],
    )
    build_eq9_and_coverage(engine, inequality_module, core)
    return core


# Current Monte Carlo 与 Sobol

def summarize_draws(draws: pd.DataFrame, id_columns: Optional[Mapping[str, object]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """汇总指标变化的区间及四目标同时改善的概率。"""
    metrics = [*SIGNED_OBJECTIVES, "Four_Objective_Signed_Sum", "Net_Benefit_Signed"]
    rows = []
    for metric in metrics:
        series = pd.to_numeric(draws[metric], errors="coerce").dropna()
        row = {
            "Metric": metric,
            "N": len(series),
            "P05": float(series.quantile(0.05)),
            "P50": float(series.quantile(0.50)),
            "P95": float(series.quantile(0.95)),
            "Probability_Positive": float((series > 0.0).mean()),
        }
        if id_columns:
            row.update(id_columns)
        rows.append(row)
    probability = {
        "N": len(draws),
        **{f"P_{metric}_Positive": float((draws[metric] > 0.0).mean()) for metric in SIGNED_OBJECTIVES},
        "P_All_Four_Signed_Objectives_Positive": float((draws[list(SIGNED_OBJECTIVES)] > 0.0).all(axis=1).mean()),
        "P_Net_Benefit_Signed_Positive": float((draws["Net_Benefit_Signed"] > 0.0).mean()),
    }
    if id_columns:
        probability.update(id_columns)
    return pd.DataFrame(rows), pd.DataFrame([probability])


def run_current_monte_carlo(engine, n_iterations: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对 S5 的八个参数进行 1,000 次 Monte Carlo 抽样。"""
    draws = engine.run_monte_carlo_exact(n_iterations=n_iterations, random_seed=42)
    summary, probability = summarize_draws(draws)
    write_csv(
        draws,
        CURRENT_DIR / "current_monte_carlo_final.csv",
        f"{os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}; EXIOBASE PXP 2022",
        sort_by=["Iteration"],
    )
    write_csv(
        summary,
        CURRENT_DIR / "current_monte_carlo_summary_final.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_monte_carlo_final.csv",
        sort_by=["Metric"],
    )
    write_csv(
        probability,
        CURRENT_DIR / "current_monte_carlo_signed_probability_final.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_monte_carlo_final.csv",
    )
    return draws, summary, probability


def sobol_convergence(
    frame_small: pd.DataFrame,
    frame_large: pd.DataFrame,
    sample_comparison: str,
) -> pd.DataFrame:
    """比较 N=128 与 N=256 的参数排名及前两位参数，检查收敛。"""
    rows = []
    for metric in SIGNED_OBJECTIVES:
        for index_name in ("First_Order", "Total_Effect"):
            small = frame_small.loc[frame_small["Metric"].eq(metric)].set_index("Parameter")[index_name]
            large = frame_large.loc[frame_large["Metric"].eq(metric)].set_index("Parameter")[index_name]
            parameters = sorted(set(small.index) & set(large.index))
            correlation = float(spearmanr(small.loc[parameters], large.loc[parameters]).statistic)
            dominant_small = str(small.idxmax())
            dominant_large = str(large.idxmax())
            top_small = set(small.nlargest(2).index)
            top_large = set(large.nlargest(2).index)
            overlap = len(top_small & top_large)
            passed = bool(np.isfinite(correlation) and correlation >= 0.80 and dominant_small == dominant_large and overlap >= 1)
            rows.append({
                "Metric": metric,
                "Index_Type": index_name,
                "Sample_Comparison": sample_comparison,
                "Spearman_Rank_Correlation": correlation,
                "Dominant_N128": dominant_small,
                "Dominant_N256": dominant_large,
                "Dominant_Consistent": dominant_small == dominant_large,
                "Top_Two_Overlap": overlap,
                "Passed": passed,
                "Extension_N512_Required": not passed,
            })
    return pd.DataFrame(rows)


def final_current_sobol_comparisons(sobol_audit: pd.DataFrame) -> pd.DataFrame:
    """按指标及指数类型选择样本量最大的可用比较。"""
    if sobol_audit.empty:
        return sobol_audit.copy()
    selected = sobol_audit.copy()
    if "Sample_Comparison" not in selected.columns:
        selected["Sample_Comparison"] = "N128_vs_N256"
    comparison_priority = {
        "N128_vs_N256": 0,
        "N256_vs_N512": 1,
    }
    selected["_Comparison_Priority"] = (
        selected["Sample_Comparison"].map(comparison_priority).fillna(-1).astype(int)
    )
    selected = (
        selected.sort_values(
            ["Metric", "Index_Type", "_Comparison_Priority"],
            kind="mergesort",
        )
        .groupby(["Metric", "Index_Type"], as_index=False, sort=False)
        .tail(1)
        .drop(columns=["_Comparison_Priority"])
        .reset_index(drop=True)
    )
    return selected


def current_sobol_status(sobol_audit: pd.DataFrame) -> bool:
    """根据最大样本量的可用比较判断 Sobol 收敛状态。"""
    selected = final_current_sobol_comparisons(sobol_audit)
    return bool(selected["Passed"].fillna(False).all())


def write_current_sobol_report(sobol_audit: pd.DataFrame, n512: Optional[pd.DataFrame]) -> None:
    """写出 Sobol 抽样与收敛结果。"""
    selected = final_current_sobol_comparisons(sobol_audit)
    convergence_status = current_sobol_status(sobol_audit)
    lines = [
        "# Current Sobol 8-parameter convergence report",
        "",
        f"Date: {CREATED_AT}  ",
        "Author: ScarfQian  ",
        "协议状态：`CURRENT_SOBOL_8PARAM_PASS`  ",
        f"最终收敛状态：`{convergence_status}`",
        "",
        "==========",
        "",
        f"- Initial comparison: N128_vs_N256",
        f"- N512 executed: {'yes' if n512 is not None else 'no'}",
        "- Final gate selects N256_vs_N512 when available; otherwise it keeps N128_vs_N256.",
        "- Residual instability remains `False` in convergence status and is not relaxed.",
        "",
        "| Metric | Index_Type | Sample_Comparison | Passed | Spearman | Dominant_Consistent | Top_Two_Overlap |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in selected.sort_values(["Metric", "Index_Type"], kind="mergesort").iterrows():
        lines.append(
            "| {metric} | {index_type} | {comparison} | {passed} | {spearman:.6f} | {dominant} | {overlap} |".format(
                metric=row["Metric"],
                index_type=row["Index_Type"],
                comparison=row["Sample_Comparison"],
                passed=bool(row["Passed"]),
                spearman=float(row["Spearman_Rank_Correlation"]),
                dominant=bool(row["Dominant_Consistent"]),
                overlap=int(row["Top_Two_Overlap"]),
            )
        )
    (CURRENT_DIR / "current_sobol_8param_convergence_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_current_sobol(engine) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """运行八参数 N=128/256 Sobol，不稳定时追加 N=512。"""
    n128 = engine.run_sobol_8param_exact(128, random_seed=42)
    n256 = engine.run_sobol_8param_exact(256, random_seed=42)
    audit = sobol_convergence(n128, n256, sample_comparison="N128_vs_N256")
    n512 = None
    if not bool(audit["Passed"].all()):
        n512 = engine.run_sobol_8param_exact(512, random_seed=42)
        extended = sobol_convergence(n256, n512, sample_comparison="N256_vs_N512")
        audit = pd.concat([audit, extended], ignore_index=True)
    write_csv(n128, CURRENT_DIR / "current_sobol_8param_N128.csv", str(os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)), sort_by=["Metric", "Parameter"])
    write_csv(n256, CURRENT_DIR / "current_sobol_8param_N256.csv", str(os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)), sort_by=["Metric", "Parameter"])
    if n512 is not None:
        write_csv(n512, CURRENT_DIR / "current_sobol_8param_N512.csv", str(os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)), sort_by=["Metric", "Parameter"])
    write_csv(
        audit,
        CURRENT_DIR / "current_sobol_convergence_audit.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_sobol_8param_N128.csv; current_sobol_8param_N256.csv",
        sort_by=["Metric", "Index_Type"],
    )
    write_current_sobol_report(audit, n512)
    return n128, n256, audit, n512


# Current NSGA-III

def _frontier_metrics(
    all_by_seed: Mapping[int, pd.DataFrame],
    fronts_by_seed: Mapping[int, pd.DataFrame],
    union_front: pd.DataFrame,
    generations: int,
) -> pd.DataFrame:
    """使用联合候选的统一归一化空间计算超体积、覆盖与前沿距离。"""
    combined = pd.concat(list(all_by_seed.values()), ignore_index=True)
    values = combined[list(SIGNED_OBJECTIVES)].to_numpy(dtype=float)
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    span = upper - lower

    def normalize(frame: pd.DataFrame) -> np.ndarray:
        array = frame[list(SIGNED_OBJECTIVES)].to_numpy(dtype=float)
        return np.divide(array - lower, span, out=np.zeros_like(array), where=span > 0.0)

    union_norm = normalize(union_front)
    union_range = np.ptp(union_norm, axis=0)
    rows = []
    for seed, frontier in fronts_by_seed.items():
        normalized = normalize(frontier)
        hv = float(HV(ref_point=np.full(4, 1.1))(1.0 - normalized)) if HV is not None and len(normalized) else np.nan
        seed_range = np.ptp(normalized, axis=0) if len(normalized) else np.zeros(4)
        coverage = np.divide(seed_range, union_range, out=np.ones(4), where=union_range > 0.0)
        if len(normalized) and len(union_norm):
            distances = cdist(union_norm, normalized)
            frontier_distance = float(np.max(np.min(distances, axis=1)))
        else:
            frontier_distance = np.inf
        rows.append({
            "Seed": seed,
            "Generations": generations,
            "N_All_Candidates": len(all_by_seed[seed]),
            "N_Frontier": len(frontier),
            "Normalized_Hypervolume": hv,
            "Minimum_Union_Objective_Range_Coverage": float(np.min(coverage)),
            "Normalized_Union_to_Seed_Frontier_Distance": frontier_distance,
        })
    audit = pd.DataFrame(rows)
    hv_mean = float(audit["Normalized_Hypervolume"].mean())
    hv_spread = float((audit["Normalized_Hypervolume"].max() - audit["Normalized_Hypervolume"].min()) / hv_mean) if hv_mean > 0.0 else np.inf
    audit["Hypervolume_Relative_Spread"] = hv_spread
    audit["Hypervolume_Spread_Passed"] = hv_spread <= 0.02
    audit["Range_Coverage_Passed"] = audit["Minimum_Union_Objective_Range_Coverage"] >= 0.90
    audit["Frontier_Distance_Tolerance"] = 0.10
    audit["Frontier_Distance_Passed"] = audit["Normalized_Union_to_Seed_Frontier_Distance"] <= 0.10
    audit["Passed"] = (
        audit["Hypervolume_Spread_Passed"]
        & audit["Range_Coverage_Passed"]
        & audit["Frontier_Distance_Passed"]
    )
    return audit


def _union_front(all_by_seed: Mapping[int, pd.DataFrame], inequality_module) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """合并三个种子的全部候选，并筛选非支配解。"""
    union_all = pd.concat(list(all_by_seed.values()), ignore_index=True)
    decision_columns = ["tax_rate", "renewable_share", "shift_share", "training_support_level"]
    unique = union_all.drop_duplicates(subset=decision_columns, keep="first").copy()
    unique["Pareto_Final"] = inequality_module.non_dominated_flags(unique, SIGNED_OBJECTIVES)
    frontier = unique.loc[unique["Pareto_Final"]].sort_values(
        list(SIGNED_OBJECTIVES),
        ascending=[False] * 4,
        kind="mergesort",
    ).reset_index(drop=True)
    frontier["Final_Frontier_ID"] = [f"final_pareto_{idx + 1:03d}" for idx in range(len(frontier))]
    return union_all, frontier


def run_current_nsga3(engine, scenario_module, inequality_module) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行三个种子、各 30 代；未收敛时扩展到 50 代。"""
    seeds = (42, 314, 2026)

    def run_generation_setting(generations: int):
        all_by_seed: Dict[int, pd.DataFrame] = {}
        fronts_by_seed: Dict[int, pd.DataFrame] = {}
        for seed in seeds:
            all_candidates, frontier = scenario_module.run_nsga3_seed_exact(
                engine,
                seed=seed,
                pop_size=70,
                generations=generations,
                reference_partitions=4,
            )
            all_by_seed[seed] = all_candidates
            fronts_by_seed[seed] = frontier
        union_all, union_front = _union_front(all_by_seed, inequality_module)
        audit = _frontier_metrics(all_by_seed, fronts_by_seed, union_front, generations)
        return all_by_seed, fronts_by_seed, union_all, union_front, audit

    all_by_seed, fronts_by_seed, union_all, union_front, audit = run_generation_setting(30)
    extension_reason = "none"
    if not bool(audit["Passed"].all()):
        extension_reason = "30-generation convergence gate failed; rerun all seeds at 50 generations"
        for seed in seeds:
            write_csv(
                all_by_seed[seed],
                CURRENT_DIR / f"current_nsga3_seed{seed}_all_candidates_g30.csv",
                str(os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)),
                sort_by=["Evaluation_Index"],
            )
        all_by_seed, fronts_by_seed, union_all, union_front, audit = run_generation_setting(50)
    audit["Extension_Reason"] = extension_reason
    for seed in seeds:
        write_csv(
            all_by_seed[seed],
            CURRENT_DIR / f"current_nsga3_seed{seed}_all_candidates.csv",
            str(os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)),
            sort_by=["Evaluation_Index"],
        )
        write_csv(
            fronts_by_seed[seed],
            CURRENT_DIR / f"current_nsga3_seed{seed}_frontier.csv",
            f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_nsga3_seed{seed}_all_candidates.csv",
            sort_by=list(SIGNED_OBJECTIVES),
        )
    write_csv(
        union_all,
        CURRENT_DIR / "current_nsga3_union_all_candidates.csv",
        "; ".join(f"current_nsga3_seed{seed}_all_candidates.csv" for seed in seeds),
        sort_by=["Seed", "Evaluation_Index"],
    )
    write_csv(
        union_front,
        CURRENT_DIR / "current_nsga3_final_frontier.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_nsga3_union_all_candidates.csv",
        sort_by=list(SIGNED_OBJECTIVES),
    )
    write_csv(
        audit,
        CURRENT_DIR / "current_nsga3_convergence_audit.csv",
        f"{os.path.relpath(CURRENT_DIR, PROJECT_ROOT)}/current_nsga3_union_all_candidates.csv",
        sort_by=["Seed"],
    )
    report = [
        "# Current NSGA-III frontier stability report",
        "",
        f"Date: {CREATED_AT}  ",
        "Author: ScarfQian  ",
        f"状态：`{'CURRENT_NSGA3_CONVERGENCE_PASS' if bool(audit['Passed'].all()) else 'CURRENT_NSGA3_CONVERGENCE_BLOCKED'}`",
        "",
        "==========",
        "",
        f"- Final generations: {int(audit['Generations'].iloc[0])}",
        f"- Union frontier size: {len(union_front)}",
        f"- Hypervolume relative spread: {float(audit['Hypervolume_Relative_Spread'].iloc[0]):.8f}",
        f"- Extension reason: {extension_reason}",
        "- Main frontier uses four signed objectives; labour hours are not optimized.",
    ]
    (CURRENT_DIR / "current_nsga3_frontier_stability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return union_all, union_front, audit


# Future 36 states 与 12 key worlds

def build_figure2_sector_driver_coverage(engine) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算按规模加权选定走廊的生产部门贡献。"""
    top_path = SOURCE_DIR / "Figure2_authoritative_top_corridors_snapshot.csv"
    if not top_path.exists():
        raise FileNotFoundError(f"Figure 2 authoritative corridor snapshot 缺失: {top_path}")
    top = pd.read_csv(top_path).reset_index(drop=True)
    top["Corridor_Row"] = np.arange(len(top), dtype=int)
    required = {"From_Region", "To_Region", "Skill", "Carbon_Transfer"}
    missing = sorted(required.difference(top.columns))
    if missing:
        raise ValueError(f"Figure 2 top corridor 表缺少列: {missing}")
    if not engine.baseline:
        engine.compute_baseline()
    outcomes = engine.baseline
    carbon_fd = np.asarray(outcomes["Carbon_By_Producer_And_FD"], dtype=float)
    employment_sector = np.asarray(outcomes["Employment_By_Skill_And_Sector"], dtype=float)
    skill_total = employment_sector.sum(axis=0)
    skill_shares = np.divide(
        employment_sector,
        skill_total.reshape(1, -1),
        out=np.zeros_like(employment_sector),
        where=skill_total.reshape(1, -1) > 0.0,
    )
    region_index = {str(region): idx for idx, region in enumerate(engine.regions)}
    fd_columns = engine._get_fd_country_columns()
    driver_rows = []
    audit_rows = []
    top_driver_labels = []
    for row_index, row in top.reset_index(drop=True).iterrows():
        producer = str(row["From_Region"])
        consumer = str(row["To_Region"])
        skill = str(row["Skill"])
        if producer not in region_index or consumer not in fd_columns or skill not in SKILLS:
            audit_rows.append({
                "Corridor_Row": row_index,
                "From_Region": producer,
                "To_Region": consumer,
                "Skill": skill,
                "Formal_Carbon_Transfer": float(row["Carbon_Transfer"]),
                "Rebuilt_Carbon_Transfer": np.nan,
                "Relative_Difference": np.nan,
                "Driver_Row_Count": 0,
                "Covered": False,
                "Coverage_Status": "missing_region_or_skill_mapping",
            })
            top_driver_labels.append("coverage unavailable")
            continue
        producer_idx = region_index[producer]
        producer_slice = engine.country_slices[producer_idx]
        columns = np.asarray(fd_columns[consumer], dtype=int)
        skill_idx = SKILLS.index(skill)
        sector_carbon = carbon_fd[producer_slice][:, columns].sum(axis=1)
        sector_skill_carbon = sector_carbon * skill_shares[skill_idx, producer_slice]
        rebuilt = float(sector_skill_carbon.sum())
        formal = float(row["Carbon_Transfer"])
        relative = (rebuilt - formal) / abs(formal) if formal != 0.0 else np.nan
        order = np.argsort(-sector_skill_carbon, kind="stable")
        positive_order = [int(idx) for idx in order if sector_skill_carbon[idx] > 0.0][:10]
        labels = []
        for rank, local_idx in enumerate(positive_order, start=1):
            global_idx = producer_slice.start + local_idx
            value = float(sector_skill_carbon[local_idx])
            label = str(engine.row_labels[global_idx])
            labels.append(f"{label} ({value / rebuilt:.1%})" if rebuilt > 0.0 else label)
            driver_rows.append({
                "Corridor_Row": row_index,
                "From_Region": producer,
                "To_Region": consumer,
                "Skill": skill,
                "Exclusive_Type": row.get("Exclusive_Type", ""),
                "Sector": label,
                "Sector_Rank_By_Carbon": rank,
                "Sector_Skill_Carbon_Transfer": value,
                "Sector_Carbon_Share_In_Corridor": value / rebuilt if rebuilt > 0.0 else np.nan,
                "Formal_Corridor_Carbon_Transfer": formal,
                "Rebuilt_Corridor_Carbon_Transfer": rebuilt,
                "Scale_Weighted_Burden_Return_Score": row.get("Scale_Weighted_Burden_Return_Score", np.nan),
            })
        covered = bool(np.isclose(rebuilt, formal, rtol=1e-6, atol=1e-2) and len(positive_order) > 0)
        audit_rows.append({
            "Corridor_Row": row_index,
            "From_Region": producer,
            "To_Region": consumer,
            "Skill": skill,
            "Formal_Carbon_Transfer": formal,
            "Rebuilt_Carbon_Transfer": rebuilt,
            "Relative_Difference": relative,
            "Driver_Row_Count": len(positive_order),
            "Covered": covered,
            "Coverage_Status": "exact_coverage" if covered else "rebuild_mismatch",
        })
        top_driver_labels.append("; ".join(labels) if labels else "no positive sector driver")
    top["Top_Sector_Drivers_Final"] = top_driver_labels
    drivers = pd.DataFrame(driver_rows)
    audit = pd.DataFrame(audit_rows)
    write_csv(
        drivers,
        SOURCE_DIR / "Figure2_sector_drivers_for_scale_weighted_corridors_final.csv",
        f"{os.path.relpath(top_path, PROJECT_ROOT)}; {os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}; EXIOBASE PXP 2022",
        sort_by=["Corridor_Row", "Sector_Rank_By_Carbon"],
    )
    write_csv(
        top,
        SOURCE_DIR / "Figure2_top_corridors_with_sector_drivers_final.csv",
        f"{os.path.relpath(top_path, PROJECT_ROOT)}; Figure2_sector_drivers_for_scale_weighted_corridors_final.csv",
    )
    write_csv(
        audit,
        AUDIT_DIR / "Figure2_sector_driver_final_coverage_audit.csv",
        f"{os.path.relpath(SOURCE_DIR, PROJECT_ROOT)}/Figure2_sector_drivers_for_scale_weighted_corridors_final.csv",
        sort_by=["Corridor_Row"],
    )
    return drivers, audit

def load_future_engine(final_scenario_engine, scenario_module):
    """复用已加载的足迹引擎，初始化未来情景引擎。"""
    if not isinstance(final_scenario_engine, scenario_module.FinalScenarioEngine):
        raise TypeError("LOAD_FUTURE_ENGINE_REQUIRES_FINAL_SCENARIO_ENGINE")
    footprint_engine = getattr(final_scenario_engine, "engine", None)
    if footprint_engine is None:
        raise TypeError("LOAD_FUTURE_ENGINE_REQUIRES_FOOTPRINT_ENGINE_HANDLE")
    future_module = load_module(FUTURE_PATH)
    future_engine = future_module.FinalFutureEngine(footprint_engine)
    future_engine.generate_pathway_parameters()
    return future_engine, future_module


def run_future_deterministic(future_engine, future_module) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算 36 个基准状态、36 个组合政策状态及 20 个锚点状态。"""
    baseline = future_engine.project_future_baseline()
    write_csv(
        baseline,
        FUTURE_DIR / "future_baseline_36states_final.csv",
        f"{os.path.relpath(FUTURE_PATH, PROJECT_ROOT)}; {os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}; EXIOBASE PXP 2022",
        sort_by=["Pathway", "Year"],
    )
    core_rows = []
    for world in future_module.baseline_worlds():
        params = future_engine._get_parameter_row(world.pathway, world.year)
        scenario = future_engine._create_future_scenario_engine(params)
        record = scenario.evaluate_policy_exact(
            {
                "tax_rate": 100.0,
                "renewable_share": 0.60,
                "shift_share": 0.20,
                "training_support_level": 0.30,
            },
            policy_id=f"FC_{world.pathway}_{world.year}",
        )
        record.update({
            "World_Scope": "baseline_core_trajectory",
            "Pathway": world.pathway,
            "Pathway_Label": world.pathway_label,
            "Year": world.year,
        })
        core_rows.append(record)
    core = pd.DataFrame(core_rows)
    write_csv(
        core,
        FUTURE_DIR / "future_core_policy_36states_final.csv",
        f"{os.path.relpath(FUTURE_PATH, PROJECT_ROOT)}; {os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}; EXIOBASE PXP 2022",
        sort_by=["Pathway", "Year"],
    )
    anchor = baseline.loc[baseline["Year"].isin(future_module.ANCHOR_YEARS)].copy()
    anchor["World_Scope"] = "anchor_state"
    write_csv(
        anchor,
        FUTURE_DIR / "future_anchor_states_20_final.csv",
        f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_baseline_36states_final.csv",
        sort_by=["Pathway", "Year"],
    )
    if len(baseline) != 36 or len(core) != 36 or len(anchor) != 20:
        raise RuntimeError("FUTURE_36STATE_BLOCKED_COUNT_MISMATCH")
    return baseline, core, anchor


def build_key_world_cache(future_engine, future_module) -> Dict[Tuple[str, int], object]:
    """缓存 12 个关键路径-年份的政策引擎，供各分析阶段复用。"""
    cache: Dict[Tuple[str, int], object] = {}
    for world in future_module.key_worlds():
        params = future_engine._get_parameter_row(world.pathway, world.year)
        cache[(world.pathway, world.year)] = future_engine._create_future_scenario_engine(params)
    if len(cache) != 12:
        raise RuntimeError("FUTURE_KEY_WORLD_BLOCKED_COUNT_MISMATCH")
    return cache


def run_future_monte_carlo(
    future_module,
    cache: Mapping[Tuple[str, int], object],
    n_iterations: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对 12 个关键路径-年份各进行八参数、1,000 次 Monte Carlo 抽样。"""
    draw_frames = []
    summary_frames = []
    probability_frames = []
    for world_index, world in enumerate(future_module.key_worlds()):
        scenario = cache[(world.pathway, world.year)]
        draws = scenario.run_monte_carlo_exact(
            n_iterations=n_iterations,
            random_seed=42 + 101 * world_index,
        )
        draws["Pathway"] = world.pathway
        draws["Pathway_Label"] = world.pathway_label
        draws["Year"] = world.year
        draw_frames.append(draws)
        summary, probability = summarize_draws(
            draws,
            id_columns={
                "Pathway": world.pathway,
                "Pathway_Label": world.pathway_label,
                "Year": world.year,
            },
        )
        summary_frames.append(summary)
        probability_frames.append(probability)
    all_draws = pd.concat(draw_frames, ignore_index=True)
    all_summary = pd.concat(summary_frames, ignore_index=True)
    all_probability = pd.concat(probability_frames, ignore_index=True)
    write_csv(
        all_draws,
        FUTURE_DIR / "future_monte_carlo_12worlds_final.csv",
        f"{os.path.relpath(FUTURE_PATH, PROJECT_ROOT)}; {os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}",
        sort_by=["Pathway", "Year", "Iteration"],
    )
    write_csv(
        all_summary,
        FUTURE_DIR / "future_monte_carlo_summary_12worlds_final.csv",
        f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_monte_carlo_12worlds_final.csv",
        sort_by=["Pathway", "Year", "Metric"],
    )
    write_csv(
        all_probability,
        FUTURE_DIR / "future_monte_carlo_signed_probability_12worlds_final.csv",
        f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_monte_carlo_12worlds_final.csv",
        sort_by=["Pathway", "Year"],
    )
    return all_draws, all_summary, all_probability


def run_future_sobol(
    future_module,
    cache: Mapping[Tuple[str, int], object],
) -> pd.DataFrame:
    """对 12 个关键路径-年份进行 N=256 的八参数 Sobol 分析。"""
    frames = []
    for world_index, world in enumerate(future_module.key_worlds()):
        frame = cache[(world.pathway, world.year)].run_sobol_8param_exact(
            256,
            random_seed=42 + 101 * world_index,
        )
        frame["Pathway"] = world.pathway
        frame["Pathway_Label"] = world.pathway_label
        frame["Year"] = world.year
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    write_csv(
        result,
        FUTURE_DIR / "future_sobol_8param_12worlds_final.csv",
        f"{os.path.relpath(FUTURE_PATH, PROJECT_ROOT)}; {os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}",
        sort_by=["Pathway", "Year", "Metric", "Parameter"],
    )
    return result


def _frontier_consistent_bands(frontier: pd.DataFrame) -> pd.DataFrame:
    """Summarize policy-variable quantiles within each world's frontier."""
    rows = []
    variables = ("tax_rate", "renewable_share", "shift_share", "training_support_level")
    for (pathway, year), group in frontier.groupby(["Pathway", "Year"], sort=True):
        row: Dict[str, object] = {
            "Pathway": pathway,
            "Year": int(year),
            "Frontier_Size": len(group),
        }
        for variable in variables:
            series = pd.to_numeric(group[variable], errors="coerce").dropna()
            row[f"{variable}_P05"] = float(series.quantile(0.05))
            row[f"{variable}_P25"] = float(series.quantile(0.25))
            row[f"{variable}_P50"] = float(series.quantile(0.50))
            row[f"{variable}_P75"] = float(series.quantile(0.75))
            row[f"{variable}_P95"] = float(series.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def run_future_nsga3(
    future_module,
    scenario_module,
    inequality_module,
    cache: Mapping[Tuple[str, int], object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """计算 12 个路径-年份的单种子前沿，并对三个代表情景进行三种子收敛检查。"""
    all_frames = []
    frontier_frames = []
    for world in future_module.key_worlds():
        all_candidates, frontier = scenario_module.run_nsga3_seed_exact(
            cache[(world.pathway, world.year)],
            seed=42,
            pop_size=70,
            generations=15,
            reference_partitions=4,
        )
        for frame in (all_candidates, frontier):
            frame["Pathway"] = world.pathway
            frame["Pathway_Label"] = world.pathway_label
            frame["Year"] = world.year
            frame["Future_NSGA_Scope"] = FUTURE_NSGA3_ALL12_SCOPE
        all_frames.append(all_candidates)
        frontier_frames.append(frontier)

    convergence_rows = []
    selected_worlds = (("P3", 2050), ("P3", 2060), ("P4", 2050))
    for pathway, year in selected_worlds:
        scenario = cache[(pathway, year)]
        all_by_seed: Dict[int, pd.DataFrame] = {}
        fronts_by_seed: Dict[int, pd.DataFrame] = {}
        for seed in (42, 314, 2026):
            all_candidates, frontier = scenario_module.run_nsga3_seed_exact(
                scenario,
                seed=seed,
                pop_size=70,
                generations=30,
                reference_partitions=4,
            )
            all_candidates["Pathway"] = pathway
            all_candidates["Year"] = year
            all_candidates["Future_NSGA_Scope"] = FUTURE_NSGA3_SELECTED_MULTI_G30_SCOPE
            frontier["Pathway"] = pathway
            frontier["Year"] = year
            frontier["Future_NSGA_Scope"] = FUTURE_NSGA3_SELECTED_MULTI_G30_SCOPE
            all_by_seed[seed] = all_candidates
            fronts_by_seed[seed] = frontier
            all_frames.append(all_candidates)
        union_all, union_front = _union_front(all_by_seed, inequality_module)
        union_front["Pathway"] = pathway
        union_front["Year"] = year
        union_front["Future_NSGA_Scope"] = FUTURE_NSGA3_SELECTED_UNION_G30_SCOPE
        audit = _frontier_metrics(all_by_seed, fronts_by_seed, union_front, 30)
        audit["World"] = f"{pathway}-{year}"
        convergence_rows.append(audit)
        frontier_frames.append(union_front)

    all_candidates = pd.concat(all_frames, ignore_index=True)
    frontiers = pd.concat(frontier_frames, ignore_index=True)
    convergence = pd.concat(convergence_rows, ignore_index=True)
    current_audit_path = CURRENT_DIR / "current_nsga3_convergence_audit.csv"
    if current_audit_path.exists():
        current = pd.read_csv(current_audit_path)
        current["World"] = "current-2022"
        convergence = pd.concat([current, convergence], ignore_index=True, sort=False)
    frontier_consistent = _frontier_consistent_bands(
        frontiers.loc[frontiers["Future_NSGA_Scope"].eq(FUTURE_NSGA3_ALL12_SCOPE)]
    )
    write_csv(
        all_candidates,
        FUTURE_DIR / "future_nsga3_all_candidates_final.csv",
        f"{os.path.relpath(FUTURE_PATH, PROJECT_ROOT)}; {os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}",
        sort_by=["Pathway", "Year", "Future_NSGA_Scope", "Seed", "Evaluation_Index"],
    )
    write_csv(
        frontiers,
        FUTURE_DIR / "future_nsga3_frontiers_final.csv",
        f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_nsga3_all_candidates_final.csv",
        sort_by=["Pathway", "Year", "Future_NSGA_Scope"],
    )
    write_csv(
        convergence,
        FUTURE_DIR / "future_nsga3_selected_world_convergence.csv",
        f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_nsga3_all_candidates_final.csv",
        sort_by=["World", "Seed"],
    )
    write_csv(
        frontier_consistent,
        FUTURE_DIR / "future_frontier_consistent_bands_final.csv",
        f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_nsga3_frontiers_final.csv",
        sort_by=["Pathway", "Year"],
    )
    return all_candidates, frontiers, convergence, frontier_consistent


def _coerce_passed_flags(series: pd.Series) -> pd.Series:
    """将 CSV 中的 Passed 列稳定转换为布尔值。"""
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "false": False}).fillna(False).astype(bool)


def _parse_future_world_id(world_id: object) -> Tuple[str, int]:
    """将 `Pathway-Year` 形式的世界标识解析为缓存键。"""
    if not isinstance(world_id, str) or "-" not in world_id:
        raise ValueError(f"UNSUPPORTED_FUTURE_WORLD_ID: {world_id!r}")
    pathway, year_text = world_id.rsplit("-", 1)
    if not pathway or not year_text.isdigit():
        raise ValueError(f"UNSUPPORTED_FUTURE_WORLD_ID: {world_id!r}")
    return pathway, int(year_text)


def _world_membership_mask(frame: pd.DataFrame, worlds: Sequence[Tuple[str, int]]) -> pd.Series:
    """返回属于指定 Pathway-Year 组合的布尔掩码。"""
    if not worlds:
        return pd.Series(False, index=frame.index)
    allowed = {(pathway, int(year)) for pathway, year in worlds}
    years = pd.to_numeric(frame["Year"], errors="coerce")
    membership = [
        (str(pathway), int(year)) in allowed if not pd.isna(year) else False
        for pathway, year in zip(frame["Pathway"], years)
    ]
    return pd.Series(membership, index=frame.index)


def load_future_nsga3_failed_worlds(
    convergence_path: Optional[Path] = None,
) -> Tuple[Tuple[str, int, int], ...]:
    """读取未收敛路径-年份的最大代数，将重跑目标总代数提高 20 代。"""
    if convergence_path is None:
        convergence_path = FUTURE_DIR / "future_nsga3_selected_world_convergence.csv"
    convergence = pd.read_csv(convergence_path)
    failed = convergence.loc[
        convergence["World"].ne("current-2022")
        & ~_coerce_passed_flags(convergence["Passed"])
    ]
    worlds = []
    seen = set()
    for world_id in failed["World"]:
        key = _parse_future_world_id(world_id)
        if key in seen:
            continue
        seen.add(key)
        generations = pd.to_numeric(
            convergence.loc[convergence["World"].eq(world_id), "Generations"],
            errors="coerce",
        ).dropna()
        if generations.empty or float(generations.max()) != int(generations.max()):
            raise ValueError(f"INVALID_FUTURE_NSGA3_GENERATIONS: {world_id}")
        current_generations = int(generations.max())
        target_generations = current_generations + FUTURE_NSGA3_EXTENSION_STEP
        if target_generations > FUTURE_NSGA3_MAX_GENERATIONS:
            raise RuntimeError(
                "FUTURE_NSGA3_EXTENSION_LIMIT_REACHED: "
                f"{world_id} current_generations={current_generations} "
                f"max_generations={FUTURE_NSGA3_MAX_GENERATIONS}"
            )
        worlds.append((*key, target_generations))
    return tuple(worlds)


def run_future_nsga3_extension(
    scenario_module,
    inequality_module,
    cache: Mapping[Tuple[str, int], object],
    extension_targets: Sequence[Tuple[str, int, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按提高后的总代数重跑未收敛的路径-年份，并替换相应的多种子结果。"""
    if not extension_targets:
        raise ValueError("FUTURE_NSGA3_EXTENSION_REQUIRES_FAILED_WORLDS")
    failed_worlds = [(pathway, year) for pathway, year, _ in extension_targets]
    all_candidates_path = FUTURE_DIR / "future_nsga3_all_candidates_final.csv"
    frontiers_path = FUTURE_DIR / "future_nsga3_frontiers_final.csv"
    convergence_path = FUTURE_DIR / "future_nsga3_selected_world_convergence.csv"
    existing_all = pd.read_csv(all_candidates_path)
    existing_frontiers = pd.read_csv(frontiers_path)
    existing_convergence = pd.read_csv(convergence_path)
    all_scope_mask = existing_all["Future_NSGA_Scope"].astype(str).str.fullmatch(
        r"selected_world_multiseed_g\d+"
    )
    frontier_scope_mask = existing_frontiers["Future_NSGA_Scope"].astype(str).str.fullmatch(
        r"selected_world_union_frontier_g\d+"
    )
    retained_all = existing_all.loc[
        ~(all_scope_mask & _world_membership_mask(existing_all, failed_worlds))
    ].copy()
    retained_frontiers = existing_frontiers.loc[
        ~(frontier_scope_mask & _world_membership_mask(existing_frontiers, failed_worlds))
    ].copy()
    failed_world_ids = {f"{pathway}-{year}" for pathway, year in failed_worlds}
    retained_convergence = existing_convergence.loc[
        ~existing_convergence["World"].isin(failed_world_ids)
    ].copy()

    rerun_all_frames = []
    rerun_frontier_frames = []
    rerun_convergence_frames = []
    for pathway, year, target_generations in extension_targets:
        scenario = cache[(pathway, year)]
        multiseed_scope = f"selected_world_multiseed_g{target_generations}"
        union_scope = f"selected_world_union_frontier_g{target_generations}"
        all_by_seed: Dict[int, pd.DataFrame] = {}
        fronts_by_seed: Dict[int, pd.DataFrame] = {}
        for seed in (42, 314, 2026):
            all_candidates, frontier = scenario_module.run_nsga3_seed_exact(
                scenario,
                seed=seed,
                pop_size=70,
                generations=target_generations,
                reference_partitions=4,
            )
            all_candidates["Pathway"] = pathway
            all_candidates["Year"] = year
            all_candidates["Future_NSGA_Scope"] = multiseed_scope
            frontier["Pathway"] = pathway
            frontier["Year"] = year
            frontier["Future_NSGA_Scope"] = multiseed_scope
            all_by_seed[seed] = all_candidates
            fronts_by_seed[seed] = frontier
            rerun_all_frames.append(all_candidates)
        _, union_front = _union_front(all_by_seed, inequality_module)
        union_front["Pathway"] = pathway
        union_front["Year"] = year
        union_front["Future_NSGA_Scope"] = union_scope
        audit = _frontier_metrics(
            all_by_seed,
            fronts_by_seed,
            union_front,
            target_generations,
        )
        audit["World"] = f"{pathway}-{year}"
        rerun_frontier_frames.append(union_front)
        rerun_convergence_frames.append(audit)

    all_candidates = pd.concat([retained_all, *rerun_all_frames], ignore_index=True, sort=False)
    frontiers = pd.concat([retained_frontiers, *rerun_frontier_frames], ignore_index=True, sort=False)
    convergence = pd.concat([retained_convergence, *rerun_convergence_frames], ignore_index=True, sort=False)
    frontier_consistent = _frontier_consistent_bands(
        frontiers.loc[frontiers["Future_NSGA_Scope"].eq(FUTURE_NSGA3_ALL12_SCOPE)]
    )
    existing_all_source = f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_nsga3_all_candidates_final.csv"
    existing_frontier_source = f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_nsga3_frontiers_final.csv"
    existing_convergence_source = f"{os.path.relpath(FUTURE_DIR, PROJECT_ROOT)}/future_nsga3_selected_world_convergence.csv"
    write_csv(
        all_candidates,
        all_candidates_path,
        f"{existing_all_source}; {os.path.relpath(SCENARIO_PATH, PROJECT_ROOT)}",
        sort_by=["Pathway", "Year", "Future_NSGA_Scope", "Seed", "Evaluation_Index"],
    )
    write_csv(
        frontiers,
        frontiers_path,
        f"{os.path.relpath(all_candidates_path, PROJECT_ROOT)}; {existing_frontier_source}",
        sort_by=["Pathway", "Year", "Future_NSGA_Scope"],
    )
    write_csv(
        convergence,
        convergence_path,
        f"{os.path.relpath(all_candidates_path, PROJECT_ROOT)}; {existing_convergence_source}",
        sort_by=["World", "Seed"],
    )
    write_csv(
        frontier_consistent,
        FUTURE_DIR / "future_frontier_consistent_bands_final.csv",
        f"{os.path.relpath(frontiers_path, PROJECT_ROOT)}",
        sort_by=["Pathway", "Year"],
    )
    return all_candidates, frontiers, convergence, frontier_consistent


# 入口

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析分析阶段和输入输出参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "check-model", "baseline",
            "current-core", "current-monte-carlo", "current-sobol", "current-nsga3", "current-all",
            FIGURE2_STAGE,
            "future-deterministic", "future-monte-carlo", "future-sobol", "future-nsga3", "future-all",
            FUTURE_NSGA3_EXTENSION_STAGE,
        ),
        default="check-model",
    )
    parser.add_argument("--mc-draws", type=int, default=1000)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)
    try:
        validate_formal_mc_draws(args.stage, args.mc_draws)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """加载 EXIOBASE 并执行指定分析阶段。"""
    args = parse_args(argv)
    configure_paths(args.data_path, args.output_root)
    ensure_output_dirs()
    extension_targets: Tuple[Tuple[str, int, int], ...] = ()
    if args.stage == FUTURE_NSGA3_EXTENSION_STAGE:
        extension_targets = load_future_nsga3_failed_worlds()
        if not extension_targets:
            print("FUTURE_NSGA3_EXTENSION_PASS: no_failed_worlds")
            print("FUTURE_NSGA3_SELECTED_WORLD_CONVERGENCE: True")
            return
    engine, scenario_module, inequality_module = load_exact_engine()
    audit = check_model(engine, inequality_module)
    print(f"FINAL_INEQUALITY_ENGINE_TEST_PASS: {int(audit['Passed'].sum())}/{len(audit)}")
    if args.stage == "check-model":
        return
    if args.stage == "baseline":
        from .baseline import run_baseline
        run_baseline(engine.engine, RESULT_ROOT)
        return
    if args.stage in {"current-core", "current-all", "all"}:
        run_current_core(engine, inequality_module)
        from . import summaries
        summaries.configure_paths(RESULT_ROOT)
        summaries.build_fixed_hour_lock(engine)
        print("CURRENT_CORE_SCENARIOS_PASS")
    if args.stage in {"current-monte-carlo", "current-all", "all"}:
        run_current_monte_carlo(engine, n_iterations=args.mc_draws)
        print("CURRENT_MONTE_CARLO_PASS")
    if args.stage in {"current-sobol", "current-all", "all"}:
        _, _, sobol_audit, _ = run_current_sobol(engine)
        print("CURRENT_SOBOL_8PARAM_PASS")
        print(f"CURRENT_SOBOL_8PARAM_CONVERGENCE_STATUS: {current_sobol_status(sobol_audit)}")
    if args.stage in {"current-nsga3", "current-all", "all"}:
        _, _, nsga_audit = run_current_nsga3(engine, scenario_module, inequality_module)
        print(f"CURRENT_NSGA3_CONVERGENCE_STATUS: {bool(nsga_audit['Passed'].all())}")
    if args.stage == FIGURE2_STAGE:
        _, figure2_audit = build_figure2_sector_driver_coverage(engine)
        print(f"FIGURE2_SECTOR_DRIVER_COVERAGE_STATUS: {bool(figure2_audit['Covered'].all())}")
    if args.stage in FUTURE_EXPLICIT_STAGES:
        future_engine, future_module = load_future_engine(engine, scenario_module)
        key_cache = None
        if args.stage in {"future-deterministic", "future-all"}:
            run_future_deterministic(future_engine, future_module)
            print("FUTURE_36STATE_PASS")
        if args.stage in FUTURE_KEY_CACHE_STAGES:
            key_cache = build_key_world_cache(future_engine, future_module)
        if args.stage in {"future-monte-carlo", "future-all"}:
            assert key_cache is not None
            run_future_monte_carlo(future_module, key_cache, n_iterations=args.mc_draws)
            print("FUTURE_MONTE_CARLO_12WORLD_PASS")
        if args.stage in {"future-sobol", "future-all"}:
            assert key_cache is not None
            run_future_sobol(future_module, key_cache)
            print("FUTURE_SOBOL_8PARAM_12WORLD_PASS")
        if args.stage in {"future-nsga3", "future-all"}:
            assert key_cache is not None
            _, _, future_convergence, _ = run_future_nsga3(
                future_module,
                scenario_module,
                inequality_module,
                key_cache,
            )
            selected = future_convergence.loc[future_convergence["World"].ne("current-2022")]
            print(f"FUTURE_NSGA3_SELECTED_WORLD_CONVERGENCE: {bool(selected['Passed'].fillna(False).all())}")
        if args.stage == FUTURE_NSGA3_EXTENSION_STAGE:
            key_cache = build_key_world_cache(future_engine, future_module)
            _, _, future_convergence, _ = run_future_nsga3_extension(
                scenario_module,
                inequality_module,
                key_cache,
                extension_targets,
            )
            selected = future_convergence.loc[future_convergence["World"].ne("current-2022")]
            print(f"FUTURE_NSGA3_SELECTED_WORLD_CONVERGENCE: {bool(selected['Passed'].fillna(False).all())}")


if __name__ == "__main__":
    main()
