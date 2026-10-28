"""Tests for attribution, account boundaries and policy objectives."""
from pathlib import Path

import numpy as np


def test_primary_index_is_ce_cw_mean(scenario_module, fake_footprint):
    """主政策不平等指标只是 CE/CW 均值。"""
    engine = scenario_module.FinalScenarioEngine(fake_footprint)
    baseline = engine.compute_baseline()
    expected = 0.5 * (baseline["System_CE_Gini"] + baseline["System_CW_Gini"])
    assert np.isclose(baseline["Policy_Inequality_Index"], expected)
    assert np.isclose(baseline["Composite_Inequality_Index"], expected)


def test_exact_fast_physical_totals_match_full(scenario_module, fake_footprint):
    """快速代数路径与全量 MRIO 路径的物理总量必须一致。"""
    engine = scenario_module.FinalScenarioEngine(fake_footprint)
    full = engine.compute_outcomes()
    fast = engine.compute_outcomes_exact_fast(
        engine.Y,
        engine.co2_emissions,
        engine.employment["data"],
        engine.wage["data"],
    )
    for field in ("Carbon_Footprint", "Employment_Total", "Wage_Total", "Hours_Total"):
        assert np.isclose(full[field], fast[field])


def test_aggregate_policy_outcomes_match_full_matrix_state(scenario_module, fake_footprint):
    """完整 Y 冲击后的 aggregate exact 路径必须与全量矩阵一致。"""
    engine = scenario_module.FinalScenarioEngine(fake_footprint)
    params = {
        "tax_rate": 120.0,
        "renewable_share": 0.7,
        "shift_share": 0.22,
        "training_support_level": 0.4,
    }
    uncertainty = {
        "theta_tax": 0.75,
        "epsilon_highcarbon": -0.45,
        "eta_renewable_co2": 0.7,
        "lambda_green_jobs": 1.3,
        "xi_shift_effectiveness": 0.9,
        "omega_lowcarbon_substitution": 0.8,
        "rho_reskilling": 0.85,
        "omega_wage_support": 0.95,
    }
    y_full, co2_full, e_full, w_full = engine._build_policy_state(params, uncertainty)
    full = engine.compute_outcomes(Y=y_full, co2_emissions=co2_full, employment_data=e_full, wage_data=w_full)
    aggregate = engine.compute_outcomes_exact_fast(y_full, co2_full, e_full, w_full)
    for field in ("Carbon_Footprint", "Employment_Total", "Wage_Total", "Hours_Total"):
        assert np.isclose(full[field], aggregate[field])


def test_skill_wage_excludes_residual_categories(scenario_module, fake_footprint):
    """政策工资只汇总 Low/Medium/High，不强制分配 residual。"""
    engine = scenario_module.FinalScenarioEngine(fake_footprint)
    outcomes = engine.compute_outcomes_exact_fast(
        engine.Y,
        engine.co2_emissions,
        engine.employment["data"],
        engine.wage["data"],
    )
    output_total = float((engine.L @ engine.Y.sum(axis=1)).sum())
    assert np.isclose(outcomes["Wage_Total"], 6.0 * output_total)
    assert outcomes["Wage_Total"] < float((fake_footprint.wage["data"] * (engine.L @ engine.Y.sum(axis=1))).sum())


def test_lock_disparity_zero_at_equal_lock(scenario_module, fake_footprint):
    """各技能锁定率相等时 disparity 必须为零。"""
    engine = scenario_module.FinalScenarioEngine(fake_footprint)
    outcomes = engine.compute_outcomes()
    summary = engine._compute_scenario_inequality_summary(outcomes)
    assert np.isclose(summary["Employment_Lock_Disparity"], 0.0)
    assert np.isclose(summary["Hours_Lock_Disparity"], 0.0)
