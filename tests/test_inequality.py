"""Tests for attribution, account boundaries and policy objectives."""
from pathlib import Path

import numpy as np


def test_attributed_plus_unattributed_equals_full(inequality_module):
    """归属碳与未归属碳必须守恒。"""
    carbon = np.array([12.0, 9.0])
    employment = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    result = inequality_module.attribute_carbon_by_skill(carbon, employment)
    assert np.isclose(result.attributed_carbon + result.unattributed_carbon, carbon.sum())
    assert np.isclose(result.unattributed_carbon, 9.0)


def test_no_forced_one_third_allocation(inequality_module):
    """零技能就业部门不得三等分。"""
    result = inequality_module.attribute_carbon_by_skill(
        np.array([9.0]),
        np.zeros((3, 1)),
    )
    assert np.allclose(result.attributed[:, 0], 0.0)
    assert np.isclose(result.unattributed_by_sector[0], 9.0)


def test_zero_resource_positive_carbon_sorted_at_infinity(inequality_module):
    """零资源正碳单元必须位于排序末端。"""
    result = inequality_module.exact_bivariate_gini(
        np.array([1.0, 0.0, 2.0]),
        np.array([1.0, 3.0, 1.0]),
        unit_ids=["A", "B", "C"],
    )
    assert result.trace.iloc[-1]["Unit_ID"] == "B"
    assert np.isinf(result.trace.iloc[-1]["Sorting_Ratio"])


def test_stable_unit_id_tie_breaking(inequality_module):
    """相同强度单元按 Unit_ID 稳定排序。"""
    result = inequality_module.exact_bivariate_gini(
        np.ones(3),
        np.ones(3),
        unit_ids=["C", "A", "B"],
    )
    assert result.trace["Unit_ID"].tolist() == ["A", "B", "C"]


def test_eq9_trace_identity(inequality_module):
    """轨迹项之和必须独立重建 Eq. (9)。"""
    result = inequality_module.exact_bivariate_gini(
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0]),
        unit_ids=["A", "B", "C"],
    )
    rebuilt = float(result.trace["Eq9_Term"].sum() - 1.0)
    assert np.isclose(rebuilt, result.raw_value)
    assert np.isclose(np.clip(rebuilt, 0.0, 1.0), result.value)
