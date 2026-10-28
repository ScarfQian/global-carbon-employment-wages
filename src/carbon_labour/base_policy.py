#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情景模拟引擎

功能：
1. 统一基准值与情景值的 CF / EF / WF 口径
2. 构建 6 个核心情景（S0-S5）
3. 提供 Monte Carlo、Sobol、Pareto、情景矩阵分析
4. 逐情景重算扩展版三重不平等摘要指标（补充 transmission 与 coupling）
"""

from __future__ import annotations

import itertools
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .footprint import FootprintEngine


from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


class _ScenarioNSGA3Problem(ElementwiseProblem):
    """将四个政策杠杆封装为 NSGA-III 可求解的问题。"""

    def __init__(self, scenario_engine: "ScenarioEngine") -> None:
        self.scenario_engine = scenario_engine
        bounds = scenario_engine._get_policy_parameter_bounds()
        xl = np.array([bounds[name][0] for name in scenario_engine.policy_parameter_order], dtype=float)
        xu = np.array([bounds[name][1] for name in scenario_engine.policy_parameter_order], dtype=float)
        super().__init__(n_var=len(scenario_engine.policy_parameter_order), n_obj=4, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        evaluation = self.scenario_engine._evaluate_policy_vector(np.asarray(x, dtype=float))
        out["F"] = np.array(
            [
                -evaluation["Carbon_Reduction"],
                -evaluation["Employment_Gain"],
                -evaluation["Wage_Gain"],
                -evaluation["Inequality_Reduction"],
            ],
            dtype=float,
        )

class ScenarioEngine:
    """面向公正转型的情景模拟引擎。"""

    # ==========
    # 初始化与元数据解析
    # ==========

    def __init__(self, footprint_engine):
        """
        初始化情景模拟引擎。

        参数：
            footprint_engine: 已经完成数据加载的 FootprintEngine 实例。
        """
        self.engine = footprint_engine
        self.mrio = getattr(footprint_engine, "mrio", None)
        self.L = np.asarray(footprint_engine.L, dtype=float)
        self.Y = np.asarray(footprint_engine.Y, dtype=float)
        self.x = np.asarray(footprint_engine.x, dtype=float).reshape(-1)
        self.regions = list(getattr(footprint_engine, "regions", []))
        self.n_products = int(getattr(footprint_engine, "n_products", self.Y.shape[0]))
        self.co2_emissions = np.asarray(footprint_engine.co2_emissions, dtype=float).reshape(-1)

        self.employment = {
            "data": np.asarray(footprint_engine.employment["data"], dtype=float),
            "categories": list(footprint_engine.employment["categories"]),
        }
        self.wage = {
            "data": np.asarray(footprint_engine.wage["data"], dtype=float),
            "categories": list(footprint_engine.wage["categories"]),
        }

        self.skill_order = ("Low", "Medium", "High")
        self.policy_parameter_order = (
            "tax_rate",
            "renewable_share",
            "shift_share",
            "training_support_level",
        )
        self.row_labels = self._build_row_labels()
        self.employment_skill_rows = self._build_skill_rows(self.employment["categories"])
        self.wage_skill_rows = self._build_skill_rows(self.wage["categories"])
        self.employment_skill_matrix = self._aggregate_skill_matrix(
            self.employment["data"],
            self.employment_skill_rows,
        )
        self.wage_skill_matrix = self._aggregate_skill_matrix(
            self.wage["data"],
            self.wage_skill_rows,
        )
        self.fd_region_labels = self._build_fd_region_labels()
        self.country_slices = self._get_country_slices()
        self.sector_masks = self._build_sector_masks()
        self.inequality_columns = [
            "Consumption_Herfindahl",
            "Consumption_Gini_CF",
            "Consumption_Gini_Emp",
            "Consumption_Gini_Wage",
            "Production_Lock_Gap",
            "Transmission_Carbon_Gini",
            "Transmission_Employment_Gini",
            "Transmission_Wage_Gini",
            "Transmission_Carbon_Employment_Coupling",
            "Transmission_Carbon_Wage_Coupling",
            "Transmission_Employment_Wage_Coupling",
            "System_CE_Gini",
            "System_CW_Gini",
            "System_Carbon_Employment_Coupling",
            "System_Carbon_Wage_Coupling",
            "System_Employment_Wage_Coupling",
            "Appendix_M_Gini",
            "Composite_Inequality_Index",
        ]

        self.baseline: Dict = {}
        self.scenarios: Dict[str, object] = {}

    def _build_row_labels(self) -> List[str]:
        """构造与 9800 行一致的产品标签。"""
        if self.mrio is not None and hasattr(self.mrio, "Y"):
            return [str(v) for v in self.mrio.Y.index.get_level_values(1).tolist()]

        products = list(getattr(self.engine, "products", []))
        if products:
            if len(products) == self.Y.shape[0]:
                return [str(v) for v in products]
            if self.regions and len(products) * len(self.regions) == self.Y.shape[0]:
                return [str(products[i % len(products)]) for i in range(self.Y.shape[0])]

        return [f"sector_{i}" for i in range(self.Y.shape[0])]

    @staticmethod
    def _build_skill_rows(categories: Iterable[str]) -> Dict[str, List[int]]:
        """根据类别名识别低/中/高技能行索引。"""
        skill_rows = {"Low": [], "Medium": [], "High": []}
        for idx, category in enumerate(categories):
            lower = str(category).lower()
            if "low" in lower:
                skill_rows["Low"].append(idx)
            elif "medium" in lower:
                skill_rows["Medium"].append(idx)
            elif "high" in lower:
                skill_rows["High"].append(idx)
        return skill_rows

    def _aggregate_skill_matrix(
        self,
        matrix: np.ndarray,
        skill_rows: Dict[str, List[int]],
    ) -> np.ndarray:
        """将技能相关系数矩阵聚合为 Low/Medium/High 三行。"""
        aggregated = np.zeros((len(self.skill_order), matrix.shape[1]), dtype=float)
        for idx, skill in enumerate(self.skill_order):
            rows = skill_rows.get(skill, [])
            if rows:
                aggregated[idx, :] = np.asarray(matrix[rows, :], dtype=float).sum(axis=0)
        return aggregated

    def _build_fd_region_labels(self) -> List[str]:
        """构造最终需求列对应的国家标签。"""
        if self.mrio is not None and hasattr(self.mrio, "Y"):
            columns = self.mrio.Y.columns
            if isinstance(columns, pd.MultiIndex):
                return [str(v) for v in columns.get_level_values(0).tolist()]
            return [str(v) for v in columns.tolist()]

        if not self.regions:
            return [f"FD_{idx}" for idx in range(self.Y.shape[1])]
        if len(self.regions) == 1:
            return [self.regions[0]] * self.Y.shape[1]
        if self.Y.shape[1] == len(self.regions):
            return [str(v) for v in self.regions]
        if self.Y.shape[1] % len(self.regions) == 0:
            repeat_count = self.Y.shape[1] // len(self.regions)
            return [str(region) for region in self.regions for _ in range(repeat_count)]
        return [str(self.regions[idx % len(self.regions)]) for idx in range(self.Y.shape[1])]

    def _get_country_slices(self) -> List[slice]:
        """返回各国家对应的产品行切片。"""
        if not self.regions:
            return [slice(0, self.Y.shape[0])]
        return [
            slice(idx * self.n_products, (idx + 1) * self.n_products)
            for idx in range(len(self.regions))
        ]

    def _get_fd_country_columns(self) -> Dict[str, np.ndarray]:
        """返回最终需求国家到列索引的映射。"""
        labels = np.asarray(self.fd_region_labels, dtype=object)
        return {
            region: np.where(labels == region)[0]
            for region in pd.unique(labels)
        }

    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """计算传统基尼系数。"""
        array = np.sort(np.asarray(values, dtype=float).reshape(-1))
        if array.size == 0:
            return 0.0
        array = np.clip(array, 0.0, None)
        total = float(array.sum())
        if total <= 0.0:
            return 0.0
        cumsum = np.cumsum(array)
        gini = (array.size + 1 - 2 * np.sum(cumsum) / total) / array.size
        return float(np.clip(gini, 0.0, 1.0))

    @staticmethod
    def _compute_bivariate_gini(resource_values: np.ndarray, carbon_values: np.ndarray) -> float:
        """按照 Lorenz / M-Gini 逻辑计算碳对资源分配的系统失衡。"""
        resource = np.clip(np.asarray(resource_values, dtype=float).reshape(-1), 0.0, None)
        carbon = np.clip(np.asarray(carbon_values, dtype=float).reshape(-1), 0.0, None)
        if resource.size == 0 or carbon.size == 0 or resource.size != carbon.size:
            return 0.0
        if resource.sum() <= 0.0 or carbon.sum() <= 0.0:
            return 0.0

        intensity = np.divide(carbon, resource, out=np.zeros_like(carbon), where=resource > 0)
        order = np.argsort(intensity)
        resource_sorted = resource[order]
        carbon_sorted = carbon[order]
        resource_share = resource_sorted / resource_sorted.sum()
        carbon_share = carbon_sorted / carbon_sorted.sum()
        cumulative_carbon = np.cumsum(carbon_share)
        value = np.sum(resource_share * carbon_share + 2.0 * resource_share * (1.0 - cumulative_carbon)) - 1.0
        return float(np.clip(value, 0.0, 1.0))

    def _allocate_carbon_by_skill_and_sector(
        self,
        carbon_by_sector: np.ndarray,
        employment_by_skill_and_sector: np.ndarray,
    ) -> np.ndarray:
        """按照部门内就业结构把碳责任分配到技能群体。"""
        sector_totals = employment_by_skill_and_sector.sum(axis=0)
        shares = np.divide(
            employment_by_skill_and_sector,
            sector_totals.reshape(1, -1),
            out=np.zeros_like(employment_by_skill_and_sector, dtype=float),
            where=sector_totals.reshape(1, -1) > 0,
        )
        zero_mask = sector_totals <= 0
        if np.any(zero_mask):
            shares[:, zero_mask] = 1.0 / len(self.skill_order)
        return shares * carbon_by_sector.reshape(1, -1)

    @staticmethod
    def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
        """安全计算相关系数。"""
        left_array = np.asarray(left, dtype=float).reshape(-1)
        right_array = np.asarray(right, dtype=float).reshape(-1)
        if left_array.size == 0 or right_array.size == 0:
            return 0.0
        if left_array.size != right_array.size:
            return 0.0
        if np.allclose(left_array, left_array[0]) or np.allclose(right_array, right_array[0]):
            return 0.0
        value = np.corrcoef(left_array, right_array)[0, 1]
        if not np.isfinite(value):
            return 0.0
        return float(value)

    def _extract_inequality_columns(self, result: Dict[str, float]) -> Dict[str, float]:
        """从结果字典中提取统一的不平等摘要字段。"""
        return {
            column: float(result.get(column, 0.0))
            for column in self.inequality_columns
        }

    def _keyword_mask(self, keywords: Iterable[str]) -> np.ndarray:
        """按关键词匹配部门掩码。"""
        labels = [label.lower() for label in self.row_labels]
        mask = np.zeros(len(labels), dtype=bool)
        for keyword in keywords:
            mask |= np.array([keyword in label for label in labels], dtype=bool)
        return mask

    def _quantile_mask(self, quantile: float, high: bool) -> np.ndarray:
        """按碳强度分位数构造掩码。"""
        positive = self.co2_emissions > 0
        if not np.any(positive):
            return np.zeros_like(self.co2_emissions, dtype=bool)
        threshold = np.quantile(self.co2_emissions[positive], quantile)
        if high:
            return positive & (self.co2_emissions >= threshold)
        return positive & (self.co2_emissions <= threshold)

    def _build_sector_masks(self) -> Dict[str, np.ndarray]:
        """构造情景所需的核心部门掩码。"""
        fossil_mask = self._keyword_mask([
            "coal", "coke", "oil", "petroleum", "gas", "fuel", "lignite"
        ])
        electricity_mask = self._keyword_mask([
            "electricity", "steam", "power", "grid"
        ])
        green_investment_mask = self._keyword_mask([
            "electricity", "electrical", "construction", "machinery",
            "transport", "rail", "bus", "services"
        ])

        high_carbon_mask = self._quantile_mask(0.70, high=True)
        low_carbon_mask = self._quantile_mask(0.35, high=False)
        energy_mask = fossil_mask | electricity_mask

        if not np.any(energy_mask):
            energy_mask = high_carbon_mask.copy()
        if not np.any(fossil_mask):
            fossil_mask = energy_mask.copy()
        if not np.any(low_carbon_mask):
            low_carbon_mask = ~high_carbon_mask
        if not np.any(green_investment_mask):
            green_investment_mask = low_carbon_mask.copy()

        return {
            "fossil": fossil_mask,
            "electricity": electricity_mask,
            "energy": energy_mask,
            "high_carbon": high_carbon_mask,
            "low_carbon": low_carbon_mask,
            "green_investment": green_investment_mask,
        }

    # ==========
    # 通用计算工具
    # ==========

    @staticmethod
    def _safe_pct_change(new_value: float, old_value: float) -> float:
        """安全计算百分比变化。"""
        if abs(old_value) < 1e-12:
            return 0.0
        return float((new_value - old_value) / old_value)

    @staticmethod
    def _ensure_non_negative(matrix: np.ndarray) -> np.ndarray:
        """将数值裁剪到非负区间。"""
        return np.clip(matrix, 0.0, None)

    @staticmethod
    def _normalize_weights(values: np.ndarray) -> np.ndarray:
        """归一化权重，避免全零。"""
        total = float(values.sum())
        if total <= 0:
            return np.full_like(values, 1.0 / len(values), dtype=float)
        return values / total

    @staticmethod
    def _scale_block(matrix: np.ndarray, row_indices: List[int], col_mask: np.ndarray, factor: float) -> None:
        """对指定行列块做原位缩放。"""
        if not row_indices:
            return
        col_indices = np.where(col_mask)[0]
        if col_indices.size == 0:
            return
        matrix[np.ix_(row_indices, col_indices)] *= factor


    @staticmethod
    def _scale_positive_final_demand(matrix: np.ndarray, row_mask: np.ndarray, factor) -> None:
        """仅缩放正的最终需求单元，避免负库存/调整项被错误回拉。"""
        row_indices = np.where(row_mask)[0]
        if row_indices.size == 0:
            return

        block = matrix[row_indices, :]
        factor_array = np.asarray(factor, dtype=float)
        if factor_array.ndim == 0:
            scaled_block = block * float(factor_array)
        else:
            scaled_block = block * factor_array.reshape(-1, 1)
        matrix[row_indices, :] = np.where(block > 0, scaled_block, block)

    def _redistribute_released_demand(
        self,
        baseline_Y: np.ndarray,
        reduced_Y: np.ndarray,
        target_mask: np.ndarray,
        substitution_share: float,
    ) -> np.ndarray:
        """将释放出的预算按目标部门权重重新分配。"""
        substitution_share = float(np.clip(substitution_share, 0.0, 1.2))
        target_indices = np.where(target_mask)[0]
        if target_indices.size == 0 or substitution_share <= 0:
            return reduced_Y

        released_budget = np.clip((baseline_Y - reduced_Y).sum(axis=0), 0.0, None)
        target_weights = self._normalize_weights(baseline_Y[target_indices, :].sum(axis=1))

        adjusted_Y = reduced_Y.copy()
        adjusted_Y[target_indices, :] += np.outer(target_weights, released_budget * substitution_share)
        return adjusted_Y

    def compute_outcomes(
        self,
        Y: Optional[np.ndarray] = None,
        co2_emissions: Optional[np.ndarray] = None,
        employment_data: Optional[np.ndarray] = None,
        wage_data: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        统一计算基准值或情景值的 CF / EF / WF。

        参数：
            Y: 情景下的最终需求矩阵。
            co2_emissions: 情景下的排放系数向量。
            employment_data: 情景下的就业系数矩阵。
            wage_data: 情景下的工资系数矩阵。
        """
        Y_matrix = self.Y if Y is None else np.asarray(Y, dtype=float)
        co2_vector = self.co2_emissions if co2_emissions is None else np.asarray(co2_emissions, dtype=float).reshape(-1)
        employment_matrix = self.employment["data"] if employment_data is None else np.asarray(employment_data, dtype=float)
        wage_matrix = self.wage["data"] if wage_data is None else np.asarray(wage_data, dtype=float)

        employment_skill_matrix = self._aggregate_skill_matrix(employment_matrix, self.employment_skill_rows)
        wage_skill_matrix = self._aggregate_skill_matrix(wage_matrix, self.wage_skill_rows)
        employment_total_intensity = employment_matrix.sum(axis=0)
        wage_total_intensity = wage_matrix.sum(axis=0)

        induced_output = self.L @ Y_matrix
        output_by_sector = induced_output.sum(axis=1)
        carbon_by_producer_and_fd = co2_vector.reshape(-1, 1) * induced_output
        employment_by_producer_and_fd = employment_total_intensity.reshape(-1, 1) * induced_output
        wage_by_producer_and_fd = wage_total_intensity.reshape(-1, 1) * induced_output
        carbon_by_producer = carbon_by_producer_and_fd.sum(axis=1)
        carbon_vector = carbon_by_producer_and_fd.sum(axis=0)

        employment_by_skill_and_fd = employment_skill_matrix @ induced_output
        wage_by_skill_and_fd = wage_skill_matrix @ induced_output
        employment_by_skill_and_sector = employment_skill_matrix * output_by_sector.reshape(1, -1)
        wage_by_skill_and_sector = wage_skill_matrix * output_by_sector.reshape(1, -1)

        employment_by_skill = {
            skill: float(employment_by_skill_and_fd[idx, :].sum())
            for idx, skill in enumerate(self.skill_order)
        }
        wage_by_skill = {
            skill: float(wage_by_skill_and_fd[idx, :].sum())
            for idx, skill in enumerate(self.skill_order)
        }

        return {
            "Carbon_Footprint": float(carbon_vector.sum()),
            "Carbon_By_Final_Demand": carbon_vector,
            "Carbon_By_Producer": carbon_by_producer,
            "Carbon_By_Producer_And_FD": carbon_by_producer_and_fd,
            "Employment_By_Producer_And_FD": employment_by_producer_and_fd,
            "Wage_By_Producer_And_FD": wage_by_producer_and_fd,
            "Output_By_Sector": output_by_sector,
            "Employment_Total": float(sum(employment_by_skill.values())),
            "Employment_By_Skill": employment_by_skill,
            "Employment_By_Skill_And_FD": employment_by_skill_and_fd,
            "Employment_By_Skill_And_Sector": employment_by_skill_and_sector,
            "Wage_Total": float(sum(wage_by_skill.values())),
            "Wage_By_Skill": wage_by_skill,
            "Wage_By_Skill_And_FD": wage_by_skill_and_fd,
            "Wage_By_Skill_And_Sector": wage_by_skill_and_sector,
        }

    def _compute_mrio_gini_from_country_vectors(
        self,
        output_by_country: np.ndarray,
        carbon_by_country: np.ndarray,
    ) -> float:
        """根据国家产出与碳排放向量计算 MRIO M-Gini。"""
        output_array = np.asarray(output_by_country, dtype=float).reshape(-1)
        carbon_array = np.asarray(carbon_by_country, dtype=float).reshape(-1)
        output_array = np.clip(output_array, 0.0, None)
        carbon_array = np.clip(carbon_array, 0.0, None)
        if output_array.sum() <= 0.0 or carbon_array.sum() <= 0.0:
            return 0.0

        intensity = np.divide(
            carbon_array,
            output_array,
            out=np.zeros_like(carbon_array),
            where=output_array > 0,
        )
        order = np.argsort(intensity)
        output_sorted = output_array[order]
        carbon_sorted = carbon_array[order]
        value_share = output_sorted / output_sorted.sum()
        carbon_share = carbon_sorted / carbon_sorted.sum()
        cumulative_carbon = np.cumsum(carbon_share)
        m_gini = float(np.sum(value_share * carbon_share + 2.0 * value_share * (1.0 - cumulative_carbon)) - 1.0)
        return float(np.clip(m_gini, 0.0, 1.0))

    def _compute_scenario_inequality_summary(self, outcomes: Dict) -> Dict[str, float]:
        """基于情景结果重算三重不平等摘要指标。"""
        carbon_by_sector = np.asarray(outcomes["Carbon_By_Producer"], dtype=float)
        output_by_sector = np.asarray(outcomes["Output_By_Sector"], dtype=float)
        employment_by_skill_and_sector = np.asarray(outcomes["Employment_By_Skill_And_Sector"], dtype=float)
        wage_by_skill_and_sector = np.asarray(outcomes["Wage_By_Skill_And_Sector"], dtype=float)
        carbon_by_skill_and_sector = self._allocate_carbon_by_skill_and_sector(
            carbon_by_sector,
            employment_by_skill_and_sector,
        )

        carbon_by_skill = carbon_by_skill_and_sector.sum(axis=1)
        employment_by_skill = employment_by_skill_and_sector.sum(axis=1)
        wage_by_skill = wage_by_skill_and_sector.sum(axis=1)

        total_carbon_by_skill = float(carbon_by_skill.sum())
        if total_carbon_by_skill > 0.0:
            carbon_shares = carbon_by_skill / total_carbon_by_skill
        else:
            carbon_shares = np.zeros_like(carbon_by_skill)
        consumption_herfindahl = float(np.sum(carbon_shares ** 2))
        consumption_gini_cf = self._compute_gini(carbon_by_skill)
        consumption_gini_emp = self._compute_gini(employment_by_skill)
        consumption_gini_wage = self._compute_gini(wage_by_skill)

        direct_carbon = np.clip(carbon_by_sector, 0.0, None)
        positive_carbon = direct_carbon[direct_carbon > 0.0]
        if positive_carbon.size > 0:
            threshold_high = float(np.quantile(positive_carbon, 0.80))
            high_carbon_mask = direct_carbon >= threshold_high
        else:
            high_carbon_mask = np.zeros_like(direct_carbon, dtype=bool)
        production_lock = {}
        for idx, skill in enumerate(self.skill_order):
            skill_employment = employment_by_skill_and_sector[idx, :]
            total_employment = float(skill_employment.sum())
            high_carbon_employment = float(skill_employment[high_carbon_mask].sum())
            production_lock[skill] = high_carbon_employment / total_employment if total_employment > 0.0 else 0.0
        high_lock = production_lock.get("High", 0.0)
        production_lock_gap = float(production_lock.get("Low", 0.0) / high_lock) if high_lock > 0.0 else 0.0

        n_regions = len(self.country_slices)
        carbon_transfer = np.zeros((n_regions, n_regions), dtype=float)
        employment_transfer = np.zeros((n_regions, n_regions), dtype=float)
        wage_transfer = np.zeros((n_regions, n_regions), dtype=float)
        fd_country_columns = self._get_fd_country_columns()
        carbon_by_producer_and_fd = np.asarray(outcomes["Carbon_By_Producer_And_FD"], dtype=float)
        employment_by_producer_and_fd = np.asarray(outcomes["Employment_By_Producer_And_FD"], dtype=float)
        wage_by_producer_and_fd = np.asarray(outcomes["Wage_By_Producer_And_FD"], dtype=float)
        employment_country_skill = np.zeros((n_regions, len(self.skill_order)), dtype=float)
        wage_country_skill = np.zeros((n_regions, len(self.skill_order)), dtype=float)
        carbon_country_skill = np.zeros((n_regions, len(self.skill_order)), dtype=float)
        output_by_country = np.zeros(n_regions, dtype=float)
        carbon_by_country = np.zeros(n_regions, dtype=float)

        for region_idx, country_slice in enumerate(self.country_slices):
            output_by_country[region_idx] = float(output_by_sector[country_slice].sum())
            carbon_by_country[region_idx] = float(carbon_by_sector[country_slice].sum())
            employment_country_skill[region_idx, :] = employment_by_skill_and_sector[:, country_slice].sum(axis=1)
            wage_country_skill[region_idx, :] = wage_by_skill_and_sector[:, country_slice].sum(axis=1)
            carbon_country_skill[region_idx, :] = carbon_by_skill_and_sector[:, country_slice].sum(axis=1)
            for demand_idx, demand_country in enumerate(self.regions[:n_regions]):
                demand_columns = fd_country_columns.get(demand_country, np.array([], dtype=int))
                if demand_columns.size == 0:
                    continue
                carbon_transfer[region_idx, demand_idx] = float(
                    carbon_by_producer_and_fd[country_slice][:, demand_columns].sum()
                )
                employment_transfer[region_idx, demand_idx] = float(
                    employment_by_producer_and_fd[country_slice][:, demand_columns].sum()
                )
                wage_transfer[region_idx, demand_idx] = float(
                    wage_by_producer_and_fd[country_slice][:, demand_columns].sum()
                )

        net_carbon_transfer = carbon_transfer.sum(axis=1) - carbon_transfer.sum(axis=0)
        net_employment_transfer = employment_transfer.sum(axis=1) - employment_transfer.sum(axis=0)
        net_wage_transfer = wage_transfer.sum(axis=1) - wage_transfer.sum(axis=0)
        transmission_carbon_gini = self._compute_gini(np.abs(net_carbon_transfer))
        transmission_employment_gini = self._compute_gini(np.abs(net_employment_transfer))
        transmission_wage_gini = self._compute_gini(np.abs(net_wage_transfer))

        export_carbon_country_skill = np.zeros_like(employment_country_skill)
        for region_idx in range(n_regions):
            export_carbon = float(carbon_transfer[region_idx, :].sum())
            employment_weights = self._normalize_weights(employment_country_skill[region_idx, :])
            export_carbon_country_skill[region_idx, :] = export_carbon * employment_weights

        transmission_ce_coupling = self._safe_corr(
            export_carbon_country_skill.reshape(-1),
            employment_country_skill.reshape(-1),
        )
        transmission_cw_coupling = self._safe_corr(
            export_carbon_country_skill.reshape(-1),
            wage_country_skill.reshape(-1),
        )
        transmission_ew_coupling = self._safe_corr(
            np.abs(net_employment_transfer),
            np.abs(net_wage_transfer),
        )
        system_ce_gini = self._compute_bivariate_gini(
            employment_country_skill.reshape(-1),
            carbon_country_skill.reshape(-1),
        )
        system_cw_gini = self._compute_bivariate_gini(
            wage_country_skill.reshape(-1),
            carbon_country_skill.reshape(-1),
        )
        system_ce_coupling = self._safe_corr(
            carbon_country_skill.reshape(-1),
            employment_country_skill.reshape(-1),
        )
        system_cw_coupling = self._safe_corr(
            carbon_country_skill.reshape(-1),
            wage_country_skill.reshape(-1),
        )
        system_ew_coupling = self._safe_corr(
            employment_country_skill.reshape(-1),
            wage_country_skill.reshape(-1),
        )
        appendix_m_gini = self._compute_mrio_gini_from_country_vectors(output_by_country, carbon_by_country)
        consumption_score = float(np.mean([consumption_gini_cf, consumption_gini_wage]))
        composite_index = float(np.mean([
            consumption_score,
            min(production_lock_gap, 3.0) / 3.0,
            transmission_carbon_gini,
            system_ce_gini,
            system_cw_gini,
        ]))

        return {
            "Consumption_Herfindahl": consumption_herfindahl,
            "Consumption_Gini_CF": consumption_gini_cf,
            "Consumption_Gini_Emp": consumption_gini_emp,
            "Consumption_Gini_Wage": consumption_gini_wage,
            "Production_Lock_Gap": production_lock_gap,
            "Transmission_Carbon_Gini": transmission_carbon_gini,
            "Transmission_Employment_Gini": transmission_employment_gini,
            "Transmission_Wage_Gini": transmission_wage_gini,
            "Transmission_Carbon_Employment_Coupling": transmission_ce_coupling,
            "Transmission_Carbon_Wage_Coupling": transmission_cw_coupling,
            "Transmission_Employment_Wage_Coupling": transmission_ew_coupling,
            "System_CE_Gini": system_ce_gini,
            "System_CW_Gini": system_cw_gini,
            "System_Carbon_Employment_Coupling": system_ce_coupling,
            "System_Carbon_Wage_Coupling": system_cw_coupling,
            "System_Employment_Wage_Coupling": system_ew_coupling,
            "Appendix_M_Gini": appendix_m_gini,
            "Composite_Inequality_Index": composite_index,
        }

    def _build_result_record(self, scenario_name: str, outcomes: Dict, metadata: Optional[Dict] = None) -> Dict:
        """将情景结果展平成统一记录。"""
        if not self.baseline:
            self.baseline = self.compute_outcomes()

        baseline = self.baseline
        record = {
            "Scenario": scenario_name,
            "Carbon_Footprint": outcomes["Carbon_Footprint"],
            "Carbon_Change": self._safe_pct_change(outcomes["Carbon_Footprint"], baseline["Carbon_Footprint"]),
            "Employment_Total": outcomes["Employment_Total"],
            "Employment_Change": self._safe_pct_change(outcomes["Employment_Total"], baseline["Employment_Total"]),
            "Wage_Total": outcomes["Wage_Total"],
            "Wage_Change": self._safe_pct_change(outcomes["Wage_Total"], baseline["Wage_Total"]),
        }

        for skill in self.skill_order:
            emp_value = outcomes["Employment_By_Skill"].get(skill, 0.0)
            wage_value = outcomes["Wage_By_Skill"].get(skill, 0.0)
            record[f"Employment_{skill}"] = emp_value
            record[f"Employment_{skill}_Change"] = self._safe_pct_change(
                emp_value,
                baseline["Employment_By_Skill"].get(skill, 0.0),
            )
            record[f"Wage_{skill}"] = wage_value
            record[f"Wage_{skill}_Change"] = self._safe_pct_change(
                wage_value,
                baseline["Wage_By_Skill"].get(skill, 0.0),
            )

        record.update(self._compute_scenario_inequality_summary(outcomes))
        record.update(self._extract_transition_metrics(record))
        if metadata:
            record.update(metadata)
        return record
    def _extract_transition_metrics(self, result: Dict) -> Dict[str, float]:
        """将情景结果转换为四目标优化可用的收益指标。"""
        carbon_reduction = max(-float(result["Carbon_Change"]), 0.0)
        employment_gain = max(float(result["Employment_Change"]), 0.0)
        wage_gain = max(float(result["Wage_Change"]), 0.0)
        baseline_inequality = float(self.baseline.get("Composite_Inequality_Index", 0.0)) if self.baseline else 0.0
        current_inequality = float(result.get("Composite_Inequality_Index", baseline_inequality))
        inequality_reduction = max(baseline_inequality - current_inequality, 0.0)
        return {
            "Carbon_Reduction": carbon_reduction,
            "Employment_Gain": employment_gain,
            "Wage_Gain": wage_gain,
            "Inequality_Reduction": inequality_reduction,
            "Transition_Benefit_Score": carbon_reduction + employment_gain + wage_gain + inequality_reduction,
        }

    @staticmethod
    def _build_monte_carlo_summary(draws: pd.DataFrame) -> pd.DataFrame:
        """根据 Monte Carlo 抽样结果构造 p5/p50/p95 汇总表。"""
        if draws.empty:
            return pd.DataFrame()

        metric_columns = (
            "theta_tax",
            "epsilon_highcarbon",
            "eta_renewable_co2",
            "lambda_green_jobs",
            "Carbon_Change",
            "Employment_Change",
            "Wage_Change",
            "Composite_Inequality_Index",
            "Carbon_Reduction",
            "Employment_Gain",
            "Wage_Gain",
            "Inequality_Reduction",
            "Transition_Benefit_Score",
            "Net_Benefit_Score",
        )
        summary = {
            "Draw_Count": int(len(draws)),
            "Policy_Cost_Index": float(draws["Policy_Cost_Index"].iloc[0]) if "Policy_Cost_Index" in draws.columns else np.nan,
        }

        for column in metric_columns:
            if column not in draws.columns:
                continue
            series = pd.to_numeric(draws[column], errors="coerce").dropna()
            if series.empty:
                continue
            summary[f"{column}_Mean"] = float(series.mean())
            summary[f"{column}_P05"] = float(series.quantile(0.05))
            summary[f"{column}_P50"] = float(series.quantile(0.50))
            summary[f"{column}_P95"] = float(series.quantile(0.95))

        return pd.DataFrame([summary])

    @staticmethod
    def _compute_policy_cost_index(
        tax_rate: float,
        renewable_share: float,
        shift_share: float,
        training_support_level: float,
    ) -> float:
        """将四类政策强度标准化为单一成本指数。"""
        tax_norm = np.clip((float(tax_rate) - 50.0) / 100.0, 0.0, 1.0)
        renewable_norm = np.clip((float(renewable_share) - 0.30) / 0.60, 0.0, 1.0)
        shift_norm = np.clip((float(shift_share) - 0.10) / 0.20, 0.0, 1.0)
        training_norm = np.clip(float(training_support_level) / 0.60, 0.0, 1.0)
        return float(
            0.35 * tax_norm
            + 0.30 * renewable_norm
            + 0.20 * shift_norm
            + 0.15 * training_norm
        )

    @staticmethod
    def _classify_policy_profile(
        carbon_reduction: float,
        employment_gain: float,
        wage_gain: float,
        policy_cost_index: float,
    ) -> str:
        """为情景矩阵与帕累托结果打上易读标签。"""
        if carbon_reduction >= 0.20 and employment_gain >= 0.10 and wage_gain >= 0.05:
            return "balanced_high_impact"
        if carbon_reduction >= 0.15 and employment_gain < 0.05:
            return "carbon_first"
        if employment_gain >= 0.10 and carbon_reduction < 0.10:
            return "jobs_first"
        if policy_cost_index >= 0.75:
            return "high_cost_push"
        return "incremental_transition"

    @staticmethod
    def _select_pareto_front(df: pd.DataFrame) -> pd.DataFrame:
        """基于四目标非支配规则筛选 Pareto 前沿。"""
        objective_columns = [
            "Carbon_Reduction",
            "Employment_Gain",
            "Wage_Gain",
            "Inequality_Reduction",
        ]
        pareto_rows = []
        for idx, row in df.iterrows():
            dominated = False
            for jdx, other in df.iterrows():
                if idx == jdx:
                    continue
                no_worse = all(float(other[column]) >= float(row[column]) for column in objective_columns)
                strictly_better = any(float(other[column]) > float(row[column]) for column in objective_columns)
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                row_dict = row.to_dict()
                row_dict["Pareto_Front"] = 1
                pareto_rows.append(row_dict)

        if not pareto_rows:
            return pd.DataFrame(columns=list(df.columns) + ["Pareto_Front"])

        pareto_df = pd.DataFrame(pareto_rows)
        sort_columns = [
            column
            for column in [
                "Net_Benefit_Score",
                "Inequality_Reduction",
                "Carbon_Reduction",
                "Employment_Gain",
                "Wage_Gain",
            ]
            if column in pareto_df.columns
        ]
        if sort_columns:
            pareto_df = pareto_df.sort_values(
                sort_columns,
                ascending=[False] * len(sort_columns),
            )
        return pareto_df.reset_index(drop=True)

    @staticmethod
    def _get_policy_parameter_bounds() -> Dict[str, Tuple[float, float]]:
        """返回正式多目标优化使用的政策变量边界。"""
        return {
            "tax_rate": (50.0, 150.0),
            "renewable_share": (0.30, 0.90),
            "shift_share": (0.10, 0.30),
            "training_support_level": (0.00, 0.60),
        }

    def _vector_to_policy_params(self, vector: np.ndarray) -> Dict[str, float]:
        """将优化向量映射为受边界约束的政策参数。"""
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if vector.size != len(self.policy_parameter_order):
            raise ValueError("政策向量维度与参数数量不一致。")

        bounds = self._get_policy_parameter_bounds()
        params = {}
        for idx, name in enumerate(self.policy_parameter_order):
            lower, upper = bounds[name]
            params[name] = float(np.clip(vector[idx], lower, upper))
        return params

    def _evaluate_policy_point(
        self,
        params: Dict[str, float],
        optimization_method: str,
    ) -> Dict[str, float]:
        """执行单个政策点评估，并返回可用于 Pareto 的完整记录。"""
        result = self.simulate_policy_package(**params)
        transition_metrics = self._extract_transition_metrics(result)
        policy_cost_index = self._compute_policy_cost_index(**params)

        record = {name: float(params[name]) for name in self.policy_parameter_order}
        record.update(
            {
                "Carbon_Change": result["Carbon_Change"],
                "Employment_Change": result["Employment_Change"],
                "Wage_Change": result["Wage_Change"],
                "Policy_Cost_Index": policy_cost_index,
                "Optimization_Method": optimization_method,
            }
        )
        record.update(self._extract_inequality_columns(result))
        record.update(transition_metrics)
        record["Net_Benefit_Score"] = transition_metrics["Transition_Benefit_Score"] - policy_cost_index
        record["Policy_Profile"] = self._classify_policy_profile(
            transition_metrics["Carbon_Reduction"],
            transition_metrics["Employment_Gain"],
            transition_metrics["Wage_Gain"],
            policy_cost_index,
        )
        return record

    def _evaluate_policy_vector(self, vector: np.ndarray) -> Dict[str, float]:
        """供 NSGA-III 调用的向量化政策点评估入口。"""
        params = self._vector_to_policy_params(vector)
        return self._evaluate_policy_point(params, optimization_method="nsga3")

    def _run_sampled_pareto_optimization(
        self,
        n_points: int,
        random_seed: int,
    ) -> pd.DataFrame:
        """使用随机采样近似 Pareto 前沿，作为正式算法的对照方案。"""
        rng = np.random.default_rng(random_seed)
        bounds = self._get_policy_parameter_bounds()
        samples = []
        for _ in range(n_points):
            params = {
                name: float(rng.uniform(lower, upper))
                for name, (lower, upper) in bounds.items()
            }
            samples.append(self._evaluate_policy_point(params, optimization_method="sampled"))

        df = pd.DataFrame(samples)
        return self._select_pareto_front(df)

    def _run_nsga3_pareto_optimization(
        self,
        n_points: int,
        random_seed: int,
        pop_size: Optional[int] = None,
        n_generations: Optional[int] = None,
        n_partitions: int = 3,
    ) -> pd.DataFrame:
        """使用 NSGA-III 执行正式四目标多目标优化。"""
        if NSGA3 is None or minimize is None or get_reference_directions is None:
            raise ImportError("当前环境缺少 pymoo，无法执行 NSGA-III 多目标优化。")

        ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=n_partitions)
        ref_dir_count = int(len(ref_dirs))
        effective_pop_size = max(int(pop_size or ref_dir_count), ref_dir_count)

        if n_generations is None:
            target_evaluations = max(int(n_points), effective_pop_size)
            effective_generations = max(10, int(np.ceil(target_evaluations / effective_pop_size)))
        else:
            effective_generations = max(int(n_generations), 1)

        problem = _ScenarioNSGA3Problem(self)
        algorithm = NSGA3(pop_size=effective_pop_size, ref_dirs=ref_dirs)
        result = minimize(
            problem,
            algorithm,
            termination=("n_gen", effective_generations),
            seed=random_seed,
            save_history=False,
            verbose=False,
        )

        if result.X is None:
            return pd.DataFrame()

        decision_matrix = np.atleast_2d(np.asarray(result.X, dtype=float))
        records = []
        for vector in decision_matrix:
            record = self._evaluate_policy_vector(vector)
            record["NSGA3_Population_Size"] = effective_pop_size
            record["NSGA3_Generations"] = effective_generations
            record["NSGA3_Reference_Directions"] = ref_dir_count
            records.append(record)

        df = pd.DataFrame(records)
        if df.empty:
            return df

        df = df.drop_duplicates(subset=list(self.policy_parameter_order)).reset_index(drop=True)
        return self._select_pareto_front(df)

    # ==========
    # 基准情景
    # ==========

    def compute_baseline(self) -> Dict:
        """计算基准情景，并统一使用足迹口径。"""
        print("\n" + "=" * 70)
        print("【计算基准情景】")
        print("=" * 70)

        self.baseline = self.compute_outcomes()
        self.baseline.update(self._compute_scenario_inequality_summary(self.baseline))

        print(f"  ✓ 基准碳足迹: {self.baseline['Carbon_Footprint']:.2e}")
        print(f"  ✓ 基准总就业: {self.baseline['Employment_Total']:.2e}")
        print(f"  ✓ 基准总工资: {self.baseline['Wage_Total']:.2e}")

        self.scenarios["baseline"] = pd.DataFrame([self._build_result_record("S0_baseline", self.baseline)])
        return self.baseline

    # ==========
    # 情景冲击构造器
    # ==========

    def _apply_carbon_tax_y_shock(
        self,
        base_Y: np.ndarray,
        tax_rate: float,
        theta_tax: float,
        epsilon_highcarbon: float,
        substitution_share: float = 0.7,
    ) -> np.ndarray:
        """将碳税映射为高碳需求收缩 + 低碳替代。"""
        high_carbon_mask = self.sector_masks["high_carbon"]
        if not np.any(high_carbon_mask):
            return np.asarray(base_Y, dtype=float).copy()

        relative_intensity = np.zeros_like(self.co2_emissions, dtype=float)
        high_carbon_intensity = np.maximum(self.co2_emissions[high_carbon_mask], 0.0)
        positive_intensity = high_carbon_intensity[high_carbon_intensity > 0]
        if positive_intensity.size > 0:
            scale = float(np.quantile(positive_intensity, 0.75))
            scale = max(scale, float(np.median(positive_intensity)), 1e-12)
            relative_intensity[high_carbon_mask] = np.log1p(high_carbon_intensity / scale)
            max_relative = float(relative_intensity[high_carbon_mask].max())
            if max_relative > 0:
                relative_intensity[high_carbon_mask] /= max_relative

        tax_scalar = max(tax_rate, 0.0) / 100.0
        price_change = np.clip(theta_tax, 0.0, 1.5) * tax_scalar * np.clip(relative_intensity, 0.0, 1.0)
        demand_change = np.clip(epsilon_highcarbon * price_change, -0.35, 0.0)

        reduced_Y = np.asarray(base_Y, dtype=float).copy()
        self._scale_positive_final_demand(
            reduced_Y,
            high_carbon_mask,
            1.0 + demand_change[high_carbon_mask],
        )

        return self._redistribute_released_demand(
            baseline_Y=np.asarray(base_Y, dtype=float),
            reduced_Y=reduced_Y,
            target_mask=self.sector_masks["low_carbon"],
            substitution_share=substitution_share,
        )

    def _apply_consumption_shift_y_shock(
        self,
        base_Y: np.ndarray,
        shift_share: float,
        substitution_share: float = 1.0,
    ) -> np.ndarray:
        """将消费结构转型映射为预算内的需求替代。"""
        shift_share = float(np.clip(shift_share, 0.0, 0.6))
        high_carbon_mask = self.sector_masks["high_carbon"]
        reduced_Y = np.asarray(base_Y, dtype=float).copy()
        self._scale_positive_final_demand(reduced_Y, high_carbon_mask, 1.0 - shift_share)

        return self._redistribute_released_demand(
            baseline_Y=np.asarray(base_Y, dtype=float),
            reduced_Y=reduced_Y,
            target_mask=self.sector_masks["low_carbon"],
            substitution_share=substitution_share,
        )

    def _apply_energy_transition_shocks(
        self,
        base_co2: np.ndarray,
        base_Y: np.ndarray,
        base_E: np.ndarray,
        renewable_share: float,
        eta_renewable_co2: float,
        lambda_green_jobs: float,
        investment_boost: float = 0.20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造能源转型的定向 S_co2 shock + Y shock + 绿色就业乘数。"""
        renewable_share = float(np.clip(renewable_share, 0.0, 1.0))
        eta_renewable_co2 = float(np.clip(eta_renewable_co2, 0.0, 1.0))
        lambda_green_jobs = float(max(lambda_green_jobs, 1.0))

        co2_new = base_co2.copy()
        fossil_mask = self.sector_masks["fossil"]
        electricity_mask = self.sector_masks["electricity"]
        energy_mask = self.sector_masks["energy"]

        co2_new[fossil_mask] *= (1.0 - eta_renewable_co2 * renewable_share)
        co2_new[electricity_mask] *= (1.0 - 0.5 * eta_renewable_co2 * renewable_share)
        co2_new[~energy_mask] *= (1.0 - 0.05 * eta_renewable_co2 * renewable_share)
        co2_new = self._ensure_non_negative(co2_new)

        fossil_reduction = min(0.30 * renewable_share, 0.40)
        reduced_Y = np.asarray(base_Y, dtype=float).copy()
        self._scale_positive_final_demand(reduced_Y, fossil_mask, 1.0 - fossil_reduction)
        Y_new = self._redistribute_released_demand(
            baseline_Y=np.asarray(base_Y, dtype=float),
            reduced_Y=reduced_Y,
            target_mask=self.sector_masks["green_investment"],
            substitution_share=1.0,
        )
        self._scale_positive_final_demand(
            Y_new,
            self.sector_masks["green_investment"],
            1.0 + investment_boost * renewable_share,
        )

        employment_new = base_E.copy()
        green_bonus = max(lambda_green_jobs - 1.0, 0.0)
        self._scale_block(
            employment_new,
            self.employment_skill_rows["Low"],
            self.sector_masks["green_investment"],
            1.0 + green_bonus * renewable_share * 0.8,
        )
        self._scale_block(
            employment_new,
            self.employment_skill_rows["Medium"],
            self.sector_masks["green_investment"],
            1.0 + green_bonus * renewable_share,
        )
        self._scale_block(
            employment_new,
            self.employment_skill_rows["High"],
            self.sector_masks["green_investment"],
            1.0 + green_bonus * renewable_share * 0.5,
        )

        return co2_new, Y_new, self._ensure_non_negative(employment_new)

    def _apply_just_transition_shocks(
        self,
        base_E: np.ndarray,
        base_W: np.ndarray,
        training_support_level: float,
        wage_support_strength: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """构造公正转型的 E/W shock。"""
        training_support_level = float(np.clip(training_support_level, 0.0, 1.0))
        employment_new = base_E.copy()
        wage_new = base_W.copy()

        low_rows = self.employment_skill_rows["Low"]
        medium_rows = self.employment_skill_rows["Medium"]
        high_rows = self.employment_skill_rows["High"]
        low_wage_rows = self.wage_skill_rows["Low"]
        medium_wage_rows = self.wage_skill_rows["Medium"]

        high_carbon_mask = self.sector_masks["high_carbon"]
        green_mask = self.sector_masks["green_investment"]

        self._scale_block(employment_new, low_rows, high_carbon_mask, 1.0 - 0.05 * training_support_level)
        self._scale_block(employment_new, low_rows, green_mask, 1.0 + 0.06 * training_support_level)
        self._scale_block(employment_new, medium_rows, green_mask, 1.0 + 0.10 * training_support_level)
        self._scale_block(employment_new, high_rows, green_mask, 1.0 + 0.03 * training_support_level)

        self._scale_block(wage_new, low_wage_rows, high_carbon_mask, 1.0 + wage_support_strength * training_support_level)
        self._scale_block(wage_new, medium_wage_rows, high_carbon_mask, 1.0 + 0.5 * wage_support_strength * training_support_level)
        self._scale_block(wage_new, low_wage_rows, green_mask, 1.0 + 0.05 * training_support_level)

        return self._ensure_non_negative(employment_new), self._ensure_non_negative(wage_new)

    # ==========
    # 六个核心情景
    # ==========

    def simulate_carbon_tax(
        self,
        tax_rate: float,
        theta_tax: float = 0.8,
        epsilon_highcarbon: float = -0.5,
        substitution_share: float = 0.7,
    ) -> Dict:
        """S1：碳税情景。"""
        if not self.baseline:
            self.compute_baseline()

        Y_new = self._apply_carbon_tax_y_shock(
            base_Y=self.Y,
            tax_rate=tax_rate,
            theta_tax=theta_tax,
            epsilon_highcarbon=epsilon_highcarbon,
            substitution_share=substitution_share,
        )
        outcomes = self.compute_outcomes(Y=Y_new)
        result = self._build_result_record(
            "S1_carbon_tax",
            outcomes,
            metadata={
                "Tax_Rate": float(tax_rate),
                "theta_tax": float(theta_tax),
                "epsilon_highcarbon": float(epsilon_highcarbon),
            },
        )
        self.scenarios["carbon_tax"] = pd.DataFrame([result])
        return result

    def simulate_energy_transition(
        self,
        renewable_share: float,
        eta_renewable_co2: float = 0.75,
        lambda_green_jobs: float = 1.25,
        investment_boost: float = 0.20,
    ) -> Dict:
        """S2：能源转型情景。"""
        if not self.baseline:
            self.compute_baseline()

        co2_new, Y_new, E_new = self._apply_energy_transition_shocks(
            base_co2=self.co2_emissions,
            base_Y=self.Y,
            base_E=self.employment["data"],
            renewable_share=renewable_share,
            eta_renewable_co2=eta_renewable_co2,
            lambda_green_jobs=lambda_green_jobs,
            investment_boost=investment_boost,
        )
        outcomes = self.compute_outcomes(Y=Y_new, co2_emissions=co2_new, employment_data=E_new)
        result = self._build_result_record(
            "S2_energy_transition",
            outcomes,
            metadata={
                "Renewable_Share": float(renewable_share),
                "eta_renewable_co2": float(eta_renewable_co2),
                "lambda_green_jobs": float(lambda_green_jobs),
            },
        )
        self.scenarios["energy_transition"] = pd.DataFrame([result])
        return result

    def simulate_consumption_shift(
        self,
        shift_share: float,
        substitution_share: float = 1.0,
        xi_shift_effectiveness: float = 1.0,
        omega_lowcarbon_substitution: float = 1.0,
    ) -> Dict:
        """S3：消费结构转型情景。"""
        if not self.baseline:
            self.compute_baseline()

        effective_shift_share = float(np.clip(shift_share, 0.0, 0.6)) * float(np.clip(xi_shift_effectiveness, 0.0, 1.0))
        effective_substitution_share = float(np.clip(substitution_share, 0.0, 1.0)) * float(np.clip(omega_lowcarbon_substitution, 0.0, 1.0))
        Y_new = self._apply_consumption_shift_y_shock(
            base_Y=self.Y,
            shift_share=effective_shift_share,
            substitution_share=effective_substitution_share,
        )
        outcomes = self.compute_outcomes(Y=Y_new)
        result = self._build_result_record(
            "S3_consumption_shift",
            outcomes,
            metadata={
                "Consumption_Shift_Share": float(shift_share),
                "xi_shift_effectiveness": float(np.clip(xi_shift_effectiveness, 0.0, 1.0)),
                "omega_lowcarbon_substitution": float(np.clip(omega_lowcarbon_substitution, 0.0, 1.0)),
            },
        )
        self.scenarios["consumption_shift"] = pd.DataFrame([result])
        return result

    def simulate_just_transition(
        self,
        training_support_level: float,
        wage_support_strength: float = 0.15,
        rho_reskilling: float = 1.0,
        omega_wage_support: float = 1.0,
    ) -> Dict:
        """S4：公正转型配套情景。"""
        if not self.baseline:
            self.compute_baseline()

        effective_training_support = float(np.clip(training_support_level, 0.0, 1.0)) * float(np.clip(rho_reskilling, 0.0, 1.0))
        effective_wage_support = float(max(wage_support_strength, 0.0)) * float(np.clip(omega_wage_support, 0.0, 1.0))
        E_new, W_new = self._apply_just_transition_shocks(
            base_E=self.employment["data"],
            base_W=self.wage["data"],
            training_support_level=effective_training_support,
            wage_support_strength=effective_wage_support,
        )
        outcomes = self.compute_outcomes(employment_data=E_new, wage_data=W_new)
        result = self._build_result_record(
            "S4_just_transition",
            outcomes,
            metadata={
                "Training_Support_Level": float(training_support_level),
                "rho_reskilling": float(np.clip(rho_reskilling, 0.0, 1.0)),
                "omega_wage_support": float(np.clip(omega_wage_support, 0.0, 1.0)),
            },
        )
        self.scenarios["just_transition"] = pd.DataFrame([result])
        return result

    def simulate_policy_package(
        self,
        tax_rate: float = 100.0,
        renewable_share: float = 0.60,
        shift_share: float = 0.20,
        training_support_level: float = 0.30,
        theta_tax: float = 0.8,
        epsilon_highcarbon: float = -0.5,
        eta_renewable_co2: float = 0.75,
        lambda_green_jobs: float = 1.25,
        xi_shift_effectiveness: float = 1.0,
        omega_lowcarbon_substitution: float = 1.0,
        rho_reskilling: float = 1.0,
        omega_wage_support: float = 1.0,
    ) -> Dict:
        """S5：综合政策包情景。"""
        if not self.baseline:
            self.compute_baseline()

        Y_new = self._apply_carbon_tax_y_shock(
            base_Y=self.Y,
            tax_rate=tax_rate,
            theta_tax=theta_tax,
            epsilon_highcarbon=epsilon_highcarbon,
            substitution_share=0.7,
        )
        co2_new, Y_new, E_new = self._apply_energy_transition_shocks(
            base_co2=self.co2_emissions,
            base_Y=Y_new,
            base_E=self.employment["data"],
            renewable_share=renewable_share,
            eta_renewable_co2=eta_renewable_co2,
            lambda_green_jobs=lambda_green_jobs,
            investment_boost=0.20,
        )
        Y_new = self._apply_consumption_shift_y_shock(
            base_Y=Y_new,
            shift_share=float(np.clip(shift_share, 0.0, 0.6)) * float(np.clip(xi_shift_effectiveness, 0.0, 1.0)),
            substitution_share=float(np.clip(omega_lowcarbon_substitution, 0.0, 1.0)),
        )
        E_new, W_new = self._apply_just_transition_shocks(
            base_E=E_new,
            base_W=self.wage["data"],
            training_support_level=float(np.clip(training_support_level, 0.0, 1.0)) * float(np.clip(rho_reskilling, 0.0, 1.0)),
            wage_support_strength=0.15 * float(np.clip(omega_wage_support, 0.0, 1.0)),
        )

        outcomes = self.compute_outcomes(
            Y=Y_new,
            co2_emissions=co2_new,
            employment_data=E_new,
            wage_data=W_new,
        )
        result = self._build_result_record(
            "S5_policy_package",
            outcomes,
            metadata={
                "Tax_Rate": float(tax_rate),
                "Renewable_Share": float(renewable_share),
                "Consumption_Shift_Share": float(shift_share),
                "Training_Support_Level": float(training_support_level),
                "theta_tax": float(theta_tax),
                "epsilon_highcarbon": float(epsilon_highcarbon),
                "eta_renewable_co2": float(eta_renewable_co2),
                "lambda_green_jobs": float(lambda_green_jobs),
                "xi_shift_effectiveness": float(np.clip(xi_shift_effectiveness, 0.0, 1.0)),
                "omega_lowcarbon_substitution": float(np.clip(omega_lowcarbon_substitution, 0.0, 1.0)),
                "rho_reskilling": float(np.clip(rho_reskilling, 0.0, 1.0)),
                "omega_wage_support": float(np.clip(omega_wage_support, 0.0, 1.0)),
            },
        )
        self.scenarios["policy_package"] = pd.DataFrame([result])
        return result

    def run_core_scenarios(self) -> pd.DataFrame:
        """一次性运行 S0-S5 六个核心情景。"""
        if not self.baseline:
            self.compute_baseline()

        baseline_record = self._build_result_record("S0_baseline", self.baseline)
        records = [
            baseline_record,
            self.simulate_carbon_tax(tax_rate=100.0),
            self.simulate_energy_transition(renewable_share=0.60),
            self.simulate_consumption_shift(shift_share=0.20),
            self.simulate_just_transition(training_support_level=0.30),
            self.simulate_policy_package(),
        ]
        df = pd.DataFrame(records)
        self.scenarios["core_scenarios"] = df
        self.scenarios["triple_inequality"] = df[["Scenario", *self.inequality_columns]].copy()
        return df
    # ==========
    # Monte Carlo / Sobol / Pareto / 情景矩阵
    # ==========

    def monte_carlo_simulation(
        self,
        n_iterations: int = 1000,
        random_seed: int = 42,
        package_strengths: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """对 S5 综合政策包执行 Monte Carlo 不确定性分析。"""
        if not self.baseline:
            self.compute_baseline()

        strengths = {
            "tax_rate": 100.0,
            "renewable_share": 0.60,
            "shift_share": 0.20,
            "training_support_level": 0.30,
        }
        if package_strengths:
            strengths.update(package_strengths)

        policy_cost_index = self._compute_policy_cost_index(**strengths)
        rng = np.random.default_rng(random_seed)
        records = []
        for iteration in range(n_iterations):
            params = {
                "theta_tax": float(rng.uniform(0.6, 1.0)),
                "epsilon_highcarbon": float(rng.uniform(-0.8, -0.2)),
                "eta_renewable_co2": float(rng.uniform(0.5, 0.9)),
                "lambda_green_jobs": float(rng.uniform(1.1, 1.5)),
                "xi_shift_effectiveness": float(rng.uniform(0.6, 1.0)),
                "omega_lowcarbon_substitution": float(rng.uniform(0.6, 1.0)),
                "rho_reskilling": float(rng.uniform(0.6, 1.0)),
                "omega_wage_support": float(rng.uniform(0.6, 1.0)),
            }
            result = self.simulate_policy_package(**strengths, **params)
            transition_metrics = self._extract_transition_metrics(result)
            row = {"Iteration": iteration, "Policy_Cost_Index": policy_cost_index}
            row.update(params)
            row.update({
                "Carbon_Footprint": result["Carbon_Footprint"],
                "Carbon_Change": result["Carbon_Change"],
                "Employment_Total": result["Employment_Total"],
                "Employment_Change": result["Employment_Change"],
                "Wage_Total": result["Wage_Total"],
                "Wage_Change": result["Wage_Change"],
            })
            row.update(self._extract_inequality_columns(result))
            row.update(transition_metrics)
            row["Net_Benefit_Score"] = transition_metrics["Transition_Benefit_Score"] - policy_cost_index
            records.append(row)

        df = pd.DataFrame(records)
        self.scenarios["monte_carlo"] = df
        self.scenarios["monte_carlo_summary"] = self._build_monte_carlo_summary(df)
        return df
    def sobol_sensitivity_analysis(
        self,
        n_samples: int = 256,
        random_seed: int = 42,
        package_strengths: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """基于 Saltelli/Jansen 估计的一阶与总效应敏感性指数。"""
        if not self.baseline:
            self.compute_baseline()

        strengths = {
            "tax_rate": 100.0,
            "renewable_share": 0.60,
            "shift_share": 0.20,
            "training_support_level": 0.30,
        }
        if package_strengths:
            strengths.update(package_strengths)

        bounds = {
            "theta_tax": (0.6, 1.0),
            "epsilon_highcarbon": (-0.8, -0.2),
            "eta_renewable_co2": (0.5, 0.9),
            "lambda_green_jobs": (1.1, 1.5),
        }
        metric_map = {
            "Carbon": "Carbon_Change",
            "Employment": "Employment_Change",
            "Wage": "Wage_Change",
            "Inequality": "Composite_Inequality_Index",
        }
        param_names = list(bounds.keys())
        n_params = len(param_names)
        rng = np.random.default_rng(random_seed)
        matrix_a = rng.random((n_samples, n_params))
        matrix_b = rng.random((n_samples, n_params))

        def scale_sample(sample_vector: np.ndarray) -> Dict[str, float]:
            scaled = {}
            for idx, param_name in enumerate(param_names):
                low, high = bounds[param_name]
                scaled[param_name] = float(low + (high - low) * sample_vector[idx])
            return scaled

        values_a = {metric: np.zeros(n_samples, dtype=float) for metric in metric_map}
        values_b = {metric: np.zeros(n_samples, dtype=float) for metric in metric_map}
        values_ab = {
            param_name: {metric: np.zeros(n_samples, dtype=float) for metric in metric_map}
            for param_name in param_names
        }

        for sample_idx in range(n_samples):
            params_a = scale_sample(matrix_a[sample_idx])
            params_b = scale_sample(matrix_b[sample_idx])
            result_a = self.simulate_policy_package(**strengths, **params_a)
            result_b = self.simulate_policy_package(**strengths, **params_b)

            for metric_name, result_key in metric_map.items():
                values_a[metric_name][sample_idx] = float(result_a[result_key])
                values_b[metric_name][sample_idx] = float(result_b[result_key])

            for param_idx, param_name in enumerate(param_names):
                mixed_sample = matrix_a[sample_idx].copy()
                mixed_sample[param_idx] = matrix_b[sample_idx, param_idx]
                mixed_params = scale_sample(mixed_sample)
                mixed_result = self.simulate_policy_package(**strengths, **mixed_params)
                for metric_name, result_key in metric_map.items():
                    values_ab[param_name][metric_name][sample_idx] = float(mixed_result[result_key])

        rows = []
        for param_name in param_names:
            row = {"Parameter": param_name}
            for metric_name in metric_map:
                merged_values = np.concatenate([values_a[metric_name], values_b[metric_name]])
                variance = float(np.var(merged_values, ddof=1))
                if variance <= 1e-12:
                    first_order = 0.0
                    total_order = 0.0
                else:
                    first_order = float(
                        np.mean(
                            values_b[metric_name]
                            * (values_ab[param_name][metric_name] - values_a[metric_name])
                        )
                        / variance
                    )
                    total_order = float(
                        0.5
                        * np.mean(
                            (values_a[metric_name] - values_ab[param_name][metric_name]) ** 2
                        )
                        / variance
                    )
                    first_order = float(np.clip(first_order, 0.0, 1.0))
                    total_order = float(np.clip(total_order, 0.0, 1.0))

                row[f"{metric_name}_Index"] = first_order
                row[f"{metric_name}_Total_Index"] = total_order
            rows.append(row)

        df = pd.DataFrame(rows)
        self.scenarios["sobol"] = df
        return df
    def pareto_optimization(
        self,
        n_points: int = 200,
        random_seed: int = 42,
        method: str = "nsga3",
        pop_size: Optional[int] = None,
        n_generations: Optional[int] = None,
        n_partitions: int = 3,
    ) -> pd.DataFrame:
        """执行四目标 Pareto 优化，默认使用 NSGA-III，保留 sampled 作为对照。"""
        if not self.baseline:
            self.compute_baseline()

        normalized_method = str(method).strip().lower()
        if normalized_method == "sampled":
            pareto_df = self._run_sampled_pareto_optimization(
                n_points=n_points,
                random_seed=random_seed,
            )
        elif normalized_method == "nsga3":
            pareto_df = self._run_nsga3_pareto_optimization(
                n_points=n_points,
                random_seed=random_seed,
                pop_size=pop_size,
                n_generations=n_generations,
                n_partitions=n_partitions,
            )
        else:
            raise ValueError(f"不支持的 Pareto 优化方法: {method}")

        self.scenarios["pareto"] = pareto_df
        return pareto_df
    def scenario_matrix_analysis(
        self,
        tax_rates: Optional[List[float]] = None,
        renewable_shares: Optional[List[float]] = None,
        shift_shares: Optional[List[float]] = None,
        training_levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """构建四维政策强度矩阵，并输出收益/成本对照指标。"""
        if not self.baseline:
            self.compute_baseline()

        tax_rates = tax_rates or [50.0, 100.0, 150.0]
        renewable_shares = renewable_shares or [0.30, 0.60, 0.90]
        shift_shares = shift_shares or [0.10, 0.20, 0.30]
        training_levels = training_levels or [0.00, 0.30, 0.60]

        rows = []
        for tax_rate, renewable_share, shift_share, training_support_level in itertools.product(
            tax_rates,
            renewable_shares,
            shift_shares,
            training_levels,
        ):
            result = self.simulate_policy_package(
                tax_rate=tax_rate,
                renewable_share=renewable_share,
                shift_share=shift_share,
                training_support_level=training_support_level,
            )
            transition_metrics = self._extract_transition_metrics(result)
            policy_cost_index = self._compute_policy_cost_index(
                tax_rate=tax_rate,
                renewable_share=renewable_share,
                shift_share=shift_share,
                training_support_level=training_support_level,
            )
            row = {
                "Tax_Rate": tax_rate,
                "Renewable_Share": renewable_share,
                "Consumption_Shift_Share": shift_share,
                "Training_Support_Level": training_support_level,
                "Carbon_Change": result["Carbon_Change"],
                "Employment_Change": result["Employment_Change"],
                "Wage_Change": result["Wage_Change"],
                "Carbon_Reduction": transition_metrics["Carbon_Reduction"],
                "Employment_Gain": transition_metrics["Employment_Gain"],
                "Wage_Gain": transition_metrics["Wage_Gain"],
                "Inequality_Reduction": transition_metrics["Inequality_Reduction"],
                "Transition_Benefit_Score": transition_metrics["Transition_Benefit_Score"],
                "Policy_Cost_Index": policy_cost_index,
                "Net_Benefit_Score": transition_metrics["Transition_Benefit_Score"] - policy_cost_index,
                "Policy_Profile": self._classify_policy_profile(
                    transition_metrics["Carbon_Reduction"],
                    transition_metrics["Employment_Gain"],
                    transition_metrics["Wage_Gain"],
                    policy_cost_index,
                ),
            }
            row.update(self._extract_inequality_columns(result))
            rows.append(row)

        df = pd.DataFrame(rows)
        self.scenarios["scenario_matrix"] = df
        return df

    @staticmethod
    def _build_single_policy_scan_summary(scan_df: pd.DataFrame) -> pd.DataFrame:
        """将单政策强度扫描结果压缩为可写作用的摘要表。"""
        if scan_df.empty:
            return pd.DataFrame()

        summary_rows = []
        metric_names = (
            "Carbon_Reduction",
            "Employment_Gain",
            "Wage_Gain",
            "Inequality_Reduction",
            "Net_Benefit_Score",
        )
        for scenario_family, group in scan_df.groupby("Scenario_Family", sort=False):
            group = group.sort_values("Scan_Value").reset_index(drop=True)
            row = {
                "Scenario_Family": scenario_family,
                "Scan_Parameter": str(group["Scan_Parameter"].iloc[0]),
                "Scan_Min": float(group["Scan_Value"].min()),
                "Scan_Max": float(group["Scan_Value"].max()),
                "Terminal_Carbon_Reduction": float(group["Carbon_Reduction"].iloc[-1]),
                "Terminal_Employment_Gain": float(group["Employment_Gain"].iloc[-1]),
                "Terminal_Wage_Gain": float(group["Wage_Gain"].iloc[-1]),
                "Terminal_Inequality_Reduction": float(group["Inequality_Reduction"].iloc[-1]),
                "Terminal_Net_Benefit_Score": float(group["Net_Benefit_Score"].iloc[-1]),
            }
            for metric_name in metric_names:
                best_position = int(group[metric_name].astype(float).to_numpy().argmax())
                best_row = group.iloc[best_position]
                row[f"Peak_{metric_name}"] = float(best_row[metric_name])
                row[f"At_{metric_name}"] = float(best_row["Scan_Value"])
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def single_policy_strength_scan(
        self,
        scan_points: int = 11,
        max_tax_rate: float = 150.0,
        max_renewable_share: float = 0.90,
        max_shift_share: float = 0.30,
        max_training_level: float = 0.60,
    ) -> pd.DataFrame:
        """对 S1-S5 执行强度扫描，避免只依赖单一中心参数。"""
        if not self.baseline:
            self.compute_baseline()
        if int(scan_points) < 2:
            raise ValueError("scan_points 至少应为 2。")

        scan_specs = (
            ("S1_carbon_tax", "tax_rate", np.linspace(0.0, float(max_tax_rate), int(scan_points))),
            ("S2_energy_transition", "renewable_share", np.linspace(0.0, float(max_renewable_share), int(scan_points))),
            ("S3_consumption_shift", "shift_share", np.linspace(0.0, float(max_shift_share), int(scan_points))),
            ("S4_just_transition", "training_support_level", np.linspace(0.0, float(max_training_level), int(scan_points))),
            ("S5_policy_package", "package_scale", np.linspace(0.0, 1.0, int(scan_points))),
        )

        rows = []
        for scenario_family, scan_parameter, grid in scan_specs:
            max_value = float(np.max(grid)) if len(grid) else 0.0
            for scan_index, raw_value in enumerate(grid):
                scan_value = float(raw_value)
                if scenario_family == "S1_carbon_tax":
                    tax_rate = scan_value
                    renewable_share = 0.0
                    shift_share = 0.0
                    training_support_level = 0.0
                    result = self.simulate_carbon_tax(tax_rate=tax_rate)
                elif scenario_family == "S2_energy_transition":
                    tax_rate = 0.0
                    renewable_share = scan_value
                    shift_share = 0.0
                    training_support_level = 0.0
                    result = self.simulate_energy_transition(renewable_share=renewable_share)
                elif scenario_family == "S3_consumption_shift":
                    tax_rate = 0.0
                    renewable_share = 0.0
                    shift_share = scan_value
                    training_support_level = 0.0
                    result = self.simulate_consumption_shift(shift_share=shift_share)
                elif scenario_family == "S4_just_transition":
                    tax_rate = 0.0
                    renewable_share = 0.0
                    shift_share = 0.0
                    training_support_level = scan_value
                    result = self.simulate_just_transition(training_support_level=training_support_level)
                else:
                    tax_rate = 100.0 * scan_value
                    renewable_share = 0.60 * scan_value
                    shift_share = 0.20 * scan_value
                    training_support_level = 0.30 * scan_value
                    result = self.simulate_policy_package(
                        tax_rate=tax_rate,
                        renewable_share=renewable_share,
                        shift_share=shift_share,
                        training_support_level=training_support_level,
                    )

                transition_metrics = self._extract_transition_metrics(result)
                policy_cost_index = self._compute_policy_cost_index(
                    tax_rate=tax_rate,
                    renewable_share=renewable_share,
                    shift_share=shift_share,
                    training_support_level=training_support_level,
                )
                row = {
                    "Scenario_Family": scenario_family,
                    "Scan_Parameter": scan_parameter,
                    "Scan_Index": int(scan_index),
                    "Scan_Value": scan_value,
                    "Scan_Share_Of_Max": float(scan_value / max_value) if max_value > 0 else 0.0,
                    "Tax_Rate": tax_rate,
                    "Renewable_Share": renewable_share,
                    "Consumption_Shift_Share": shift_share,
                    "Training_Support_Level": training_support_level,
                    "Carbon_Change": float(result["Carbon_Change"]),
                    "Employment_Change": float(result["Employment_Change"]),
                    "Wage_Change": float(result["Wage_Change"]),
                    "Carbon_Reduction": float(transition_metrics["Carbon_Reduction"]),
                    "Employment_Gain": float(transition_metrics["Employment_Gain"]),
                    "Wage_Gain": float(transition_metrics["Wage_Gain"]),
                    "Inequality_Reduction": float(transition_metrics["Inequality_Reduction"]),
                    "Transition_Benefit_Score": float(transition_metrics["Transition_Benefit_Score"]),
                    "Policy_Cost_Index": float(policy_cost_index),
                    "Net_Benefit_Score": float(transition_metrics["Transition_Benefit_Score"] - policy_cost_index),
                    "Policy_Profile": self._classify_policy_profile(
                        transition_metrics["Carbon_Reduction"],
                        transition_metrics["Employment_Gain"],
                        transition_metrics["Wage_Gain"],
                        policy_cost_index,
                    ),
                }
                row.update(self._extract_inequality_columns(result))
                rows.append(row)

        df = pd.DataFrame(rows)
        self.scenarios["single_policy_scan"] = df
        self.scenarios["single_policy_scan_summary"] = self._build_single_policy_scan_summary(df)
        return df

    @staticmethod
    def _build_single_policy_uncertainty_summary(draw_df: pd.DataFrame) -> pd.DataFrame:
        """将单政策扫描 Monte Carlo 结果汇总为 p05/p50/p95 区间表。"""
        if draw_df.empty:
            return pd.DataFrame()

        summary_rows = []
        metric_names = (
            "Carbon_Reduction",
            "Employment_Gain",
            "Wage_Gain",
            "Inequality_Reduction",
            "Net_Benefit_Score",
        )
        group_columns = [
            "Scenario_Family",
            "Scan_Parameter",
            "Scan_Index",
            "Scan_Value",
            "Scan_Share_Of_Max",
            "Tax_Rate",
            "Renewable_Share",
            "Consumption_Shift_Share",
            "Training_Support_Level",
        ]
        for keys, group in draw_df.groupby(group_columns, sort=False):
            row = {column: value for column, value in zip(group_columns, keys)}
            row["Draw_Count"] = int(len(group))
            for metric_name in metric_names:
                series = pd.to_numeric(group[metric_name], errors="coerce").dropna()
                row[f"{metric_name}_P05"] = float(series.quantile(0.05))
                row[f"{metric_name}_P50"] = float(series.quantile(0.50))
                row[f"{metric_name}_P95"] = float(series.quantile(0.95))
                row[f"{metric_name}_Mean"] = float(series.mean())
            summary_rows.append(row)

        return pd.DataFrame(summary_rows)

    def single_policy_strength_uncertainty_scan(
        self,
        scan_points: int = 11,
        n_iterations: int = 100,
        random_seed: int = 42,
        max_tax_rate: float = 150.0,
        max_renewable_share: float = 0.90,
        max_shift_share: float = 0.30,
        max_training_level: float = 0.60,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """对 S1-S5 的强度扫描叠加响应参数不确定性，生成结果包络。"""
        if not self.baseline:
            self.compute_baseline()
        if int(scan_points) < 2:
            raise ValueError("scan_points 至少应为 2。")
        if int(n_iterations) < 1:
            raise ValueError("n_iterations 至少应为 1。")

        scan_specs = (
            ("S1_carbon_tax", "tax_rate", np.linspace(0.0, float(max_tax_rate), int(scan_points))),
            ("S2_energy_transition", "renewable_share", np.linspace(0.0, float(max_renewable_share), int(scan_points))),
            ("S3_consumption_shift", "shift_share", np.linspace(0.0, float(max_shift_share), int(scan_points))),
            ("S4_just_transition", "training_support_level", np.linspace(0.0, float(max_training_level), int(scan_points))),
            ("S5_policy_package", "package_scale", np.linspace(0.0, 1.0, int(scan_points))),
        )
        rng = np.random.default_rng(random_seed)
        draw_rows = []

        for scenario_family, scan_parameter, grid in scan_specs:
            max_value = float(np.max(grid)) if len(grid) else 0.0
            for scan_index, raw_value in enumerate(grid):
                scan_value = float(raw_value)
                if scenario_family == "S1_carbon_tax":
                    tax_rate = scan_value
                    renewable_share = 0.0
                    shift_share = 0.0
                    training_support_level = 0.0
                elif scenario_family == "S2_energy_transition":
                    tax_rate = 0.0
                    renewable_share = scan_value
                    shift_share = 0.0
                    training_support_level = 0.0
                elif scenario_family == "S3_consumption_shift":
                    tax_rate = 0.0
                    renewable_share = 0.0
                    shift_share = scan_value
                    training_support_level = 0.0
                elif scenario_family == "S4_just_transition":
                    tax_rate = 0.0
                    renewable_share = 0.0
                    shift_share = 0.0
                    training_support_level = scan_value
                else:
                    tax_rate = 100.0 * scan_value
                    renewable_share = 0.60 * scan_value
                    shift_share = 0.20 * scan_value
                    training_support_level = 0.30 * scan_value

                for iteration in range(int(n_iterations)):
                    theta_tax = 1.0
                    epsilon_highcarbon = -0.5
                    eta_renewable_co2 = 0.75
                    lambda_green_jobs = 1.25
                    xi_shift_effectiveness = 1.0
                    omega_lowcarbon_substitution = 1.0
                    rho_reskilling = 1.0
                    omega_wage_support = 1.0

                    if scenario_family in {"S1_carbon_tax", "S5_policy_package"}:
                        theta_tax = float(rng.uniform(0.6, 1.0))
                        epsilon_highcarbon = float(rng.uniform(-0.8, -0.2))
                    if scenario_family in {"S2_energy_transition", "S5_policy_package"}:
                        eta_renewable_co2 = float(rng.uniform(0.5, 0.9))
                        lambda_green_jobs = float(rng.uniform(1.1, 1.5))
                    if scenario_family in {"S3_consumption_shift", "S5_policy_package"}:
                        xi_shift_effectiveness = float(rng.uniform(0.6, 1.0))
                        omega_lowcarbon_substitution = float(rng.uniform(0.6, 1.0))
                    if scenario_family in {"S4_just_transition", "S5_policy_package"}:
                        rho_reskilling = float(rng.uniform(0.6, 1.0))
                        omega_wage_support = float(rng.uniform(0.6, 1.0))

                    if scenario_family == "S1_carbon_tax":
                        result = self.simulate_carbon_tax(
                            tax_rate=tax_rate,
                            theta_tax=theta_tax,
                            epsilon_highcarbon=epsilon_highcarbon,
                        )
                    elif scenario_family == "S2_energy_transition":
                        result = self.simulate_energy_transition(
                            renewable_share=renewable_share,
                            eta_renewable_co2=eta_renewable_co2,
                            lambda_green_jobs=lambda_green_jobs,
                        )
                    elif scenario_family == "S3_consumption_shift":
                        result = self.simulate_consumption_shift(
                            shift_share=shift_share,
                            xi_shift_effectiveness=xi_shift_effectiveness,
                            omega_lowcarbon_substitution=omega_lowcarbon_substitution,
                        )
                    elif scenario_family == "S4_just_transition":
                        result = self.simulate_just_transition(
                            training_support_level=training_support_level,
                            rho_reskilling=rho_reskilling,
                            omega_wage_support=omega_wage_support,
                        )
                    else:
                        result = self.simulate_policy_package(
                            tax_rate=tax_rate,
                            renewable_share=renewable_share,
                            shift_share=shift_share,
                            training_support_level=training_support_level,
                            theta_tax=theta_tax,
                            epsilon_highcarbon=epsilon_highcarbon,
                            eta_renewable_co2=eta_renewable_co2,
                            lambda_green_jobs=lambda_green_jobs,
                            xi_shift_effectiveness=xi_shift_effectiveness,
                            omega_lowcarbon_substitution=omega_lowcarbon_substitution,
                            rho_reskilling=rho_reskilling,
                            omega_wage_support=omega_wage_support,
                        )

                    transition_metrics = self._extract_transition_metrics(result)
                    policy_cost_index = self._compute_policy_cost_index(
                        tax_rate=tax_rate,
                        renewable_share=renewable_share,
                        shift_share=shift_share,
                        training_support_level=training_support_level,
                    )
                    row = {
                        "Scenario_Family": scenario_family,
                        "Scan_Parameter": scan_parameter,
                        "Scan_Index": int(scan_index),
                        "Scan_Value": scan_value,
                        "Scan_Share_Of_Max": float(scan_value / max_value) if max_value > 0 else 0.0,
                        "Tax_Rate": tax_rate,
                        "Renewable_Share": renewable_share,
                        "Consumption_Shift_Share": shift_share,
                        "Training_Support_Level": training_support_level,
                        "Iteration": int(iteration),
                        "theta_tax": theta_tax,
                        "epsilon_highcarbon": epsilon_highcarbon,
                        "eta_renewable_co2": eta_renewable_co2,
                        "lambda_green_jobs": lambda_green_jobs,
                        "xi_shift_effectiveness": xi_shift_effectiveness,
                        "omega_lowcarbon_substitution": omega_lowcarbon_substitution,
                        "rho_reskilling": rho_reskilling,
                        "omega_wage_support": omega_wage_support,
                        "Carbon_Reduction": float(transition_metrics["Carbon_Reduction"]),
                        "Employment_Gain": float(transition_metrics["Employment_Gain"]),
                        "Wage_Gain": float(transition_metrics["Wage_Gain"]),
                        "Inequality_Reduction": float(transition_metrics["Inequality_Reduction"]),
                        "Net_Benefit_Score": float(transition_metrics["Transition_Benefit_Score"] - policy_cost_index),
                    }
                    draw_rows.append(row)

        draw_df = pd.DataFrame(draw_rows)
        summary_df = self._build_single_policy_uncertainty_summary(draw_df)
        self.scenarios["single_policy_scan_uncertainty_draws"] = draw_df
        self.scenarios["single_policy_scan_uncertainty_summary"] = summary_df
        return draw_df, summary_df

    # ==========
    # 结果保存
    # ==========
    def save_results(self, output_path: str) -> None:
        """保存所有情景结果。"""
        csv_dir = os.path.join(output_path, "csv")
        json_dir = os.path.join(output_path, "json")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)
        for name, data in self.scenarios.items():
            file_path = os.path.join(csv_dir, f"scenario_{name}.csv")
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
                print(f"  ✓ 保存: {file_path}")
            elif isinstance(data, dict):
                pd.DataFrame([data]).to_csv(file_path, index=False)
                print(f"  ✓ 保存: {file_path}")
