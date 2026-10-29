#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不平等公共函数。

本模块是 Eq. (9)、技能碳归属和 Pareto 判定的唯一公共实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# 规范常量

SPECIFICATION_ID = "country_skill_mismatch_mean_v1"
ENGINE_VERSION = "0716-final-inequality-v1"
EXACT_STATUS = "exact_exiobase_policy_engine"
SKILLS = ("Low", "Medium", "High")
SIGNED_OBJECTIVES = (
    "Carbon_Reduction_Signed",
    "Employment_Gain_Signed",
    "Wage_Gain_Signed",
    "Inequality_Reduction_Signed",
)


@dataclass(frozen=True)
class SkillCarbonAttribution:
    """技能碳归属结果及部门级覆盖率审计。"""

    attributed: np.ndarray
    attributable_by_sector: np.ndarray
    unattributed_by_sector: np.ndarray
    employment_total_by_sector: np.ndarray
    eligible_by_sector: np.ndarray

    @property
    def total_carbon(self) -> float:
        """返回非负全部碳宇宙。"""
        return float(self.attributable_by_sector.sum() + self.unattributed_by_sector.sum())

    @property
    def attributed_carbon(self) -> float:
        """返回可归属技能碳。"""
        return float(self.attributed.sum())

    @property
    def unattributed_carbon(self) -> float:
        """返回零技能就业部门中保留的未归属碳。"""
        return float(self.unattributed_by_sector.sum())

    @property
    def coverage(self) -> float:
        """返回全部非负碳中的技能归属比例。"""
        total = self.total_carbon
        return self.attributed_carbon / total if total > 0.0 else np.nan


@dataclass(frozen=True)
class BivariateGiniResult:
    """Eq. (9) 结果与可审计排序轨迹。"""

    value: float
    raw_value: float
    trace: pd.DataFrame


# 基础分配函数

def nonnegative_gini(values: np.ndarray) -> float:
    """计算非负值的传统 Gini，空或全零输入返回 0。"""
    array = np.sort(np.clip(np.asarray(values, dtype=float).reshape(-1), 0.0, None))
    if array.size == 0:
        return 0.0
    total = float(array.sum())
    if total <= 0.0:
        return 0.0
    cumulative = np.cumsum(array)
    value = (array.size + 1.0 - 2.0 * float(cumulative.sum()) / total) / array.size
    return float(np.clip(value, 0.0, 1.0))


def attribute_carbon_by_skill(
    carbon_by_sector: np.ndarray,
    employment_by_skill_and_sector: np.ndarray,
) -> SkillCarbonAttribution:
    """
    按部门内技能就业份额归属非负碳，不对零就业部门强制三等分。

    负的数值残差不进入排放责任宇宙。EXIOBASE 燃烧 CO2 正式输入
    为非负值，因此该裁剪只是数值边界保护。
    """
    carbon = np.clip(np.asarray(carbon_by_sector, dtype=float).reshape(-1), 0.0, None)
    employment = np.clip(np.asarray(employment_by_skill_and_sector, dtype=float), 0.0, None)
    if employment.ndim != 2 or employment.shape[1] != carbon.size:
        raise ValueError("employment 必须为 skill x sector 矩阵，且部门数与 carbon 一致。")

    sector_employment = employment.sum(axis=0)
    eligible = sector_employment > 0.0
    shares = np.divide(
        employment,
        sector_employment.reshape(1, -1),
        out=np.zeros_like(employment, dtype=float),
        where=eligible.reshape(1, -1),
    )
    attributed = shares * carbon.reshape(1, -1)
    attributable = np.where(eligible, carbon, 0.0)
    unattributed = np.where(~eligible, carbon, 0.0)
    return SkillCarbonAttribution(
        attributed=attributed,
        attributable_by_sector=attributable,
        unattributed_by_sector=unattributed,
        employment_total_by_sector=sector_employment,
        eligible_by_sector=eligible,
    )


def _validated_unit_ids(unit_ids: Sequence[str] | None, size: int) -> list[str]:
    """构造或校验唯一 Unit_ID，保证排序规则可复现。"""
    ids = [f"unit_{idx:04d}" for idx in range(size)] if unit_ids is None else [str(v) for v in unit_ids]
    if len(ids) != size:
        raise ValueError("unit_ids 长度必须与观察单元数一致。")
    if len(set(ids)) != len(ids):
        raise ValueError("unit_ids 必须唯一，否则无法稳定处理并列。")
    return ids


def exact_bivariate_gini(
    resource_values: np.ndarray,
    carbon_values: np.ndarray,
    unit_ids: Sequence[str] | None = None,
    metric: str = "System_CE_Gini",
) -> BivariateGiniResult:
    """
    使用 attributable-carbon universe 精确计算 Eq. (9)。

    排序为 carbon/resource 升序；零资源且正碳单元强度为 +inf；
    并列使用 Unit_ID 升序的稳定规则。资源分母使用全资源宇宙，
    碳分母仅使用可归属碳。
    """
    resource = np.clip(np.asarray(resource_values, dtype=float).reshape(-1), 0.0, None)
    carbon = np.clip(np.asarray(carbon_values, dtype=float).reshape(-1), 0.0, None)
    if resource.size == 0 or carbon.size != resource.size:
        raise ValueError("resource 与 carbon 必须是等长非空向量。")
    ids = _validated_unit_ids(unit_ids, resource.size)

    intensity = np.divide(
        carbon,
        resource,
        out=np.zeros_like(carbon, dtype=float),
        where=resource > 0.0,
    )
    intensity[(resource == 0.0) & (carbon > 0.0)] = np.inf
    frame = pd.DataFrame(
        {
            "Metric": metric,
            "Unit_ID": ids,
            "Resource": resource,
            "Carbon_Attributed": carbon,
            "Sorting_Ratio": intensity,
        }
    ).sort_values(
        ["Sorting_Ratio", "Unit_ID"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame["Rank"] = np.arange(1, len(frame) + 1, dtype=int)

    resource_total = float(frame["Resource"].sum())
    carbon_total = float(frame["Carbon_Attributed"].sum())
    if resource_total <= 0.0 or carbon_total <= 0.0:
        frame["D_Resource_Share"] = 0.0
        frame["V_Carbon_Share"] = 0.0
        frame["T_Cumulative_Carbon_Share"] = 0.0
        frame["Eq9_Term"] = 0.0
        raw_value = 0.0
    else:
        frame["D_Resource_Share"] = frame["Resource"] / resource_total
        frame["V_Carbon_Share"] = frame["Carbon_Attributed"] / carbon_total
        frame["T_Cumulative_Carbon_Share"] = frame["V_Carbon_Share"].cumsum()
        frame["Eq9_Term"] = (
            frame["D_Resource_Share"] * frame["V_Carbon_Share"]
            + 2.0
            * frame["D_Resource_Share"]
            * (1.0 - frame["T_Cumulative_Carbon_Share"])
        )
        raw_value = float(frame["Eq9_Term"].sum() - 1.0)

    value = float(np.clip(raw_value, 0.0, 1.0))
    frame["Eq9_Raw_Value"] = raw_value
    frame["Eq9_Value"] = value
    frame["Resource_Denominator"] = resource_total
    frame["Attributed_Carbon_Denominator"] = carbon_total
    frame["Primary_Inequality_Specification"] = SPECIFICATION_ID
    frame["Exact_or_Surrogate"] = EXACT_STATUS
    return BivariateGiniResult(value=value, raw_value=raw_value, trace=frame)


# Pareto 与审计工具

def non_dominated_flags(
    frame: pd.DataFrame,
    objective_columns: Iterable[str] = SIGNED_OBJECTIVES,
) -> pd.Series:
    """按所有目标最大化的规则返回非支配标记。"""
    columns = list(objective_columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Pareto 判定缺少目标列: {missing}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pareto 目标必须全部为有限值。")
    flags = np.ones(len(values), dtype=bool)
    for idx, row in enumerate(values):
        no_worse = np.all(values >= row, axis=1)
        strictly_better = np.any(values > row, axis=1)
        no_worse[idx] = False
        flags[idx] = not bool(np.any(no_worse & strictly_better))
    return pd.Series(flags, index=frame.index, name="Pareto_Final")


def normalized_objective_matrix(
    frame: pd.DataFrame,
    objective_columns: Iterable[str] = SIGNED_OBJECTIVES,
) -> np.ndarray:
    """在给定候选宇宙内将四目标线性映射到 [0, 1]。"""
    columns = list(objective_columns)
    values = frame[columns].to_numpy(dtype=float)
    lower = np.min(values, axis=0)
    upper = np.max(values, axis=0)
    span = upper - lower
    return np.divide(values - lower, span, out=np.zeros_like(values), where=span > 0.0)
