"""Tests for attribution, account boundaries and policy objectives."""
from pathlib import Path

import numpy as np


def test_signed_objectives_preserve_negative_outcomes(scenario_module, fake_footprint):
    """负的就业或工资改善不得被主目标裁剪。"""
    engine = scenario_module.FinalScenarioEngine(fake_footprint)
    engine.baseline = {"Policy_Inequality_Index": 0.4}
    metrics = engine._extract_transition_metrics({
        "Carbon_Change": 0.1,
        "Employment_Change": -0.2,
        "Wage_Change": -0.3,
        "Policy_Inequality_Index": 0.5,
    })
    assert metrics["Carbon_Reduction_Signed"] == -0.1
    assert metrics["Employment_Gain_Signed"] == -0.2
    assert metrics["Wage_Gain_Signed"] == -0.3
    assert np.isclose(metrics["Inequality_Reduction_Signed"], -0.1)
    assert metrics["Employment_Gain_Clipped"] == 0.0


def test_main_nsga_problem_has_four_signed_objectives(scenario_module):
    """主前沿严格保持四目标。"""
    assert scenario_module.SIGNED_OBJECTIVES == (
        "Carbon_Reduction_Signed",
        "Employment_Gain_Signed",
        "Wage_Gain_Signed",
        "Inequality_Reduction_Signed",
    )
    source = Path(scenario_module.__file__).read_text(encoding="utf-8")
    assert "n_obj=4" in source
    assert "Hours_" not in source[source.index('out["F"]'):source.index('out["F"]') + 500]


def test_broad_and_legacy_do_not_define_frontier(scenario_module):
    """broad/legacy sensitivity 字段不得决定 main Pareto membership。"""
    source = Path(scenario_module.__file__).read_text(encoding="utf-8")
    objective_block = source[source.index('out["F"]'):source.index('out["F"]') + 500]
    assert "Broad_Structural_Inequality_Index" not in objective_block
    assert "Legacy_Five_Component_Index" not in objective_block

