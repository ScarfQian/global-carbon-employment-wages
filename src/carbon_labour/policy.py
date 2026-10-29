#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四目标情景引擎。

复用基准引擎的物理冲击链，使用最终不平等定义、
有符号四目标和可审计的精确评价接口。劳动小时仍为次级诊断。
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import base_policy as _BASE
from . import inequality as _INEQ

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions


# 模型规范

ENGINE_VERSION = "0716-final-scenario-engine-v1"
SPECIFICATION_ID = "country_skill_mismatch_mean_v1"
EXACT_STATUS = "exact_exiobase_policy_engine"
SKILLS = ("Low", "Medium", "High")
UNCERTAINTY_BOUNDS = {
    "theta_tax": (0.6, 1.0),
    "epsilon_highcarbon": (-0.8, -0.2),
    "eta_renewable_co2": (0.5, 0.9),
    "lambda_green_jobs": (1.1, 1.5),
    "xi_shift_effectiveness": (0.6, 1.0),
    "omega_lowcarbon_substitution": (0.6, 1.0),
    "rho_reskilling": (0.6, 1.0),
    "omega_wage_support": (0.6, 1.0),
}
SIGNED_OBJECTIVES = (
    "Carbon_Reduction_Signed",
    "Employment_Gain_Signed",
    "Wage_Gain_Signed",
    "Inequality_Reduction_Signed",
)



class FinalScenarioEngine(_BASE.ScenarioEngine):
    """保留原物理冲击链、使用最终不平等规范的四目标引擎。"""

    def __init__(self, footprint_engine):
        super().__init__(footprint_engine)
        self.primary_inequality_specification = SPECIFICATION_ID
        self.exact_or_surrogate = EXACT_STATUS
        self.hours = {
            "data": np.asarray(footprint_engine.hours["data"], dtype=float),
            "categories": list(footprint_engine.hours["categories"]),
        }
        self.hours_skill_rows = self._build_skill_rows(self.hours["categories"])
        self.hours_skill_matrix = self._aggregate_skill_matrix(
            self.hours["data"],
            self.hours_skill_rows,
        )
        self.inequality_columns = [
            "Skill_Carbon_Herfindahl",
            "Skill_Carbon_Gini",
            "Skill_Employment_Gini",
            "Skill_Wage_Gini",
            "Net_Carbon_Transfer_Gini",
            "Net_Employment_Transfer_Gini",
            "Net_Wage_Transfer_Gini",
            "Employment_Lock_Low",
            "Employment_Lock_Medium",
            "Employment_Lock_High",
            "Employment_Lock_Disparity",
            "Employment_Lock_Direction",
            "Hours_Lock_Low",
            "Hours_Lock_Medium",
            "Hours_Lock_High",
            "Hours_Lock_Disparity",
            "System_CE_Gini",
            "System_CW_Gini",
            "Policy_Inequality_Index",
            "Composite_Inequality_Index",
            "Broad_Structural_Inequality_Index",
            "Legacy_Five_Component_Index",
            "Supplementary_Synthetic_Transmission_Coupling",
            "Attributed_Carbon",
            "Unattributed_Carbon",
            "Attribution_Coverage",
            # 兼容字段，仅用于表格连接。
            "Consumption_Herfindahl",
            "Consumption_Gini_CF",
            "Consumption_Gini_Emp",
            "Consumption_Gini_Wage",
            "Production_Lock_Gap",
            "Transmission_Carbon_Gini",
            "Transmission_Employment_Gini",
            "Transmission_Wage_Gini",
        ]

    def _allocate_carbon_by_skill_and_sector(
        self,
        carbon_by_sector: np.ndarray,
        employment_by_skill_and_sector: np.ndarray,
    ) -> np.ndarray:
        """按就业份额归属碳，零技能就业部门保留为未归属。"""
        return _INEQ.attribute_carbon_by_skill(
            carbon_by_sector,
            employment_by_skill_and_sector,
        ).attributed

    @staticmethod
    def _compute_bivariate_gini(
        resource_values: np.ndarray,
        carbon_values: np.ndarray,
        unit_ids: Optional[Sequence[str]] = None,
    ) -> float:
        """调用公共 Eq. (9) 实现，不在 ScenarioEngine 内复制公式。"""
        return _INEQ.exact_bivariate_gini(
            resource_values,
            carbon_values,
            unit_ids=unit_ids,
        ).value

    def _policy_hours_skill_intensity(self, employment_skill_matrix: np.ndarray) -> np.ndarray:
        """用 baseline hours-per-worker 保持规则生成政策工时强度。"""
        omega = np.divide(
            self.hours_skill_matrix,
            self.employment_skill_matrix,
            out=np.zeros_like(self.hours_skill_matrix, dtype=float),
            where=self.employment_skill_matrix > 0.0,
        )
        return omega * np.asarray(employment_skill_matrix, dtype=float)

    def _attach_hours(
        self,
        outcomes: Dict,
        employment_matrix: np.ndarray,
        wage_matrix: np.ndarray,
    ) -> Dict:
        """在物理 outcomes 中附加工时次级诊断所需数组。"""
        output = np.asarray(outcomes["Output_By_Sector"], dtype=float).reshape(-1)
        employment_skill = self._aggregate_skill_matrix(employment_matrix, self.employment_skill_rows)
        wage_skill = self._aggregate_skill_matrix(wage_matrix, self.wage_skill_rows)
        hours_skill = self._policy_hours_skill_intensity(employment_skill)
        hours_sector = hours_skill * output.reshape(1, -1)
        outcomes["Hours_By_Skill_And_Sector"] = hours_sector
        outcomes["Hours_By_Skill"] = {
            skill: float(hours_sector[idx, :].sum())
            for idx, skill in enumerate(SKILLS)
        }
        outcomes["Hours_Total"] = float(hours_sector.sum())
        outcomes["Employment_Intensity_By_Skill"] = employment_skill
        outcomes["Wage_Intensity_By_Skill"] = wage_skill
        outcomes["Hours_Intensity_By_Skill"] = hours_skill
        outcomes["Hours_Method"] = "baseline hours-per-worker x policy employment intensity"
        return outcomes

    def compute_outcomes(
        self,
        Y: Optional[np.ndarray] = None,
        co2_emissions: Optional[np.ndarray] = None,
        employment_data: Optional[np.ndarray] = None,
        wage_data: Optional[np.ndarray] = None,
    ) -> Dict:
        """调用基准引擎的全量精确 MRIO 计算，并附加工时数组。"""
        employment_matrix = self.employment["data"] if employment_data is None else np.asarray(employment_data, dtype=float)
        wage_matrix = self.wage["data"] if wage_data is None else np.asarray(wage_data, dtype=float)
        outcomes = super().compute_outcomes(
            Y=Y,
            co2_emissions=co2_emissions,
            employment_data=employment_matrix,
            wage_data=wage_matrix,
        )
        return self._attach_hours(outcomes, employment_matrix, wage_matrix)

    def compute_outcomes_exact_fast(
        self,
        Y: np.ndarray,
        co2_emissions: np.ndarray,
        employment_data: np.ndarray,
        wage_data: np.ndarray,
    ) -> Dict:
        """
        通过 ``L @ sum(Y, axis=1)`` 计算与全量矩阵完全相同的产出总量。

        该路径仅省略不进入主目标的 bilateral diagnostic arrays，
        不使用拟合、代理或插值，因此仍是 exact engine evaluation。
        """
        y_matrix = np.asarray(Y, dtype=float)
        co2 = np.asarray(co2_emissions, dtype=float).reshape(-1)
        employment_matrix = np.asarray(employment_data, dtype=float)
        wage_matrix = np.asarray(wage_data, dtype=float)
        final_demand_total = y_matrix if y_matrix.ndim == 1 else y_matrix.sum(axis=1)
        output = np.asarray(self.L, dtype=float) @ final_demand_total
        employment_skill = self._aggregate_skill_matrix(employment_matrix, self.employment_skill_rows)
        wage_skill = self._aggregate_skill_matrix(wage_matrix, self.wage_skill_rows)
        employment_sector = employment_skill * output.reshape(1, -1)
        wage_sector = wage_skill * output.reshape(1, -1)
        outcomes = {
            "Carbon_Footprint": float(np.dot(co2, output)),
            "Carbon_By_Producer": co2 * output,
            "Output_By_Sector": output,
            "Employment_Total": float(employment_sector.sum()),
            "Employment_By_Skill": {
                skill: float(employment_sector[idx, :].sum())
                for idx, skill in enumerate(SKILLS)
            },
            "Employment_By_Skill_And_Sector": employment_sector,
            "Wage_Total": float(wage_sector.sum()),
            "Wage_By_Skill": {
                skill: float(wage_sector[idx, :].sum())
                for idx, skill in enumerate(SKILLS)
            },
            "Wage_By_Skill_And_Sector": wage_sector,
            "Network_Diagnostics_Computed": False,
        }
        return self._attach_hours(outcomes, employment_matrix, wage_matrix)

    def _country_skill_arrays(
        self,
        employment_sector: np.ndarray,
        wage_sector: np.ndarray,
        carbon_sector: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """将部门数组聚合为 49 x 3 country-skill cells。"""
        n_regions = len(self.country_slices)
        employment = np.zeros((n_regions, len(SKILLS)), dtype=float)
        wage = np.zeros_like(employment)
        carbon = np.zeros_like(employment)
        unit_ids: list[str] = []
        for region_idx, country_slice in enumerate(self.country_slices):
            employment[region_idx, :] = employment_sector[:, country_slice].sum(axis=1)
            wage[region_idx, :] = wage_sector[:, country_slice].sum(axis=1)
            carbon[region_idx, :] = carbon_sector[:, country_slice].sum(axis=1)
            region = str(self.regions[region_idx])
            unit_ids.extend(f"{region}|{skill}" for skill in SKILLS)
        return employment.reshape(-1), wage.reshape(-1), carbon.reshape(-1), unit_ids

    def _net_transfer_diagnostics(self, outcomes: Mapping[str, object]) -> Dict[str, float]:
        """仅在全量 producer-final-demand 数组存在时计算净转移集中度。"""
        required = (
            "Carbon_By_Producer_And_FD",
            "Employment_By_Producer_And_FD",
            "Wage_By_Producer_And_FD",
        )
        if not all(name in outcomes for name in required):
            return {
                "Net_Carbon_Transfer_Gini": np.nan,
                "Net_Employment_Transfer_Gini": np.nan,
                "Net_Wage_Transfer_Gini": np.nan,
                "Supplementary_Synthetic_Transmission_Coupling": np.nan,
            }

        n_regions = len(self.country_slices)
        fd_columns = self._get_fd_country_columns()
        matrices = {
            "Carbon": np.asarray(outcomes[required[0]], dtype=float),
            "Employment": np.asarray(outcomes[required[1]], dtype=float),
            "Wage": np.asarray(outcomes[required[2]], dtype=float),
        }
        transfers = {
            name: np.zeros((n_regions, n_regions), dtype=float)
            for name in matrices
        }
        for producer_idx, country_slice in enumerate(self.country_slices):
            for consumer_idx, consumer in enumerate(self.regions[:n_regions]):
                columns = fd_columns.get(consumer, np.array([], dtype=int))
                if columns.size == 0:
                    continue
                for name, matrix in matrices.items():
                    transfers[name][producer_idx, consumer_idx] = float(
                        matrix[country_slice][:, columns].sum()
                    )

        net = {
            name: matrix.sum(axis=1) - matrix.sum(axis=0)
            for name, matrix in transfers.items()
        }
        employment_country_skill = np.zeros((n_regions, len(SKILLS)), dtype=float)
        employment_sector = np.asarray(outcomes["Employment_By_Skill_And_Sector"], dtype=float)
        for region_idx, country_slice in enumerate(self.country_slices):
            employment_country_skill[region_idx, :] = employment_sector[:, country_slice].sum(axis=1)
        synthetic_export_carbon = np.zeros_like(employment_country_skill)
        for region_idx in range(n_regions):
            weights = self._normalize_weights(employment_country_skill[region_idx, :])
            synthetic_export_carbon[region_idx, :] = float(transfers["Carbon"][region_idx, :].sum()) * weights

        return {
            "Net_Carbon_Transfer_Gini": _INEQ.nonnegative_gini(np.abs(net["Carbon"])),
            "Net_Employment_Transfer_Gini": _INEQ.nonnegative_gini(np.abs(net["Employment"])),
            "Net_Wage_Transfer_Gini": _INEQ.nonnegative_gini(np.abs(net["Wage"])),
            "Supplementary_Synthetic_Transmission_Coupling": self._safe_corr(
                synthetic_export_carbon.reshape(-1),
                employment_country_skill.reshape(-1),
            ),
        }

    def _compute_scenario_inequality_summary(self, outcomes: Dict) -> Dict[str, float]:
        """计算最终 primary index、机制诊断和 supplementary sensitivity indices。"""
        carbon_by_sector = np.asarray(outcomes["Carbon_By_Producer"], dtype=float)
        output_by_sector = np.asarray(outcomes["Output_By_Sector"], dtype=float)
        employment_sector = np.asarray(outcomes["Employment_By_Skill_And_Sector"], dtype=float)
        wage_sector = np.asarray(outcomes["Wage_By_Skill_And_Sector"], dtype=float)
        hours_sector = np.asarray(outcomes["Hours_By_Skill_And_Sector"], dtype=float)
        attribution = _INEQ.attribute_carbon_by_skill(carbon_by_sector, employment_sector)
        carbon_skill_sector = attribution.attributed

        carbon_by_skill = carbon_skill_sector.sum(axis=1)
        employment_by_skill = employment_sector.sum(axis=1)
        wage_by_skill = wage_sector.sum(axis=1)
        carbon_total = float(carbon_by_skill.sum())
        carbon_shares = carbon_by_skill / carbon_total if carbon_total > 0.0 else np.zeros_like(carbon_by_skill)

        country_employment, country_wage, country_carbon, unit_ids = self._country_skill_arrays(
            employment_sector,
            wage_sector,
            carbon_skill_sector,
        )
        ce_result = _INEQ.exact_bivariate_gini(
            country_employment,
            country_carbon,
            unit_ids=unit_ids,
            metric="System_CE_Gini",
        )
        cw_result = _INEQ.exact_bivariate_gini(
            country_wage,
            country_carbon,
            unit_ids=unit_ids,
            metric="System_CW_Gini",
        )
        policy_index = 0.5 * (ce_result.value + cw_result.value)

        positive_carbon = np.clip(carbon_by_sector, 0.0, None)
        positive_values = positive_carbon[positive_carbon > 0.0]
        threshold = float(np.quantile(positive_values, 0.80)) if positive_values.size else np.inf
        high_carbon_mask = (positive_carbon > 0.0) & (positive_carbon >= threshold)
        employment_lock: Dict[str, float] = {}
        hours_lock: Dict[str, float] = {}
        for idx, skill in enumerate(SKILLS):
            employment_total = float(employment_sector[idx, :].sum())
            hours_total = float(hours_sector[idx, :].sum())
            employment_lock[skill] = (
                float(employment_sector[idx, high_carbon_mask].sum()) / employment_total
                if employment_total > 0.0 else 0.0
            )
            hours_lock[skill] = (
                float(hours_sector[idx, high_carbon_mask].sum()) / hours_total
                if hours_total > 0.0 else 0.0
            )
        employment_disparity = float(max(employment_lock.values()) - min(employment_lock.values()))
        employment_direction = float(
            0.5 * (employment_lock["Low"] + employment_lock["Medium"])
            - employment_lock["High"]
        )
        hours_disparity = float(max(hours_lock.values()) - min(hours_lock.values()))

        transfer = self._net_transfer_diagnostics(outcomes)
        skill_carbon_gini = _INEQ.nonnegative_gini(carbon_by_skill)
        skill_employment_gini = _INEQ.nonnegative_gini(employment_by_skill)
        skill_wage_gini = _INEQ.nonnegative_gini(wage_by_skill)
        net_carbon_gini = float(transfer["Net_Carbon_Transfer_Gini"])
        broad = (
            0.25
            * (
                0.5 * (skill_carbon_gini + skill_wage_gini)
                + employment_disparity
                + net_carbon_gini
                + policy_index
            )
            if np.isfinite(net_carbon_gini) else np.nan
        )
        high_lock = employment_lock["High"]
        deprecated_lock_gap = employment_lock["Low"] / high_lock if high_lock > 0.0 else 0.0
        legacy = (
            float(np.mean([
                0.5 * (skill_carbon_gini + skill_wage_gini),
                min(deprecated_lock_gap, 3.0) / 3.0,
                net_carbon_gini,
                ce_result.value,
                cw_result.value,
            ]))
            if np.isfinite(net_carbon_gini) else np.nan
        )

        result = {
            "Skill_Carbon_Herfindahl": float(np.sum(carbon_shares ** 2)),
            "Skill_Carbon_Gini": skill_carbon_gini,
            "Skill_Employment_Gini": skill_employment_gini,
            "Skill_Wage_Gini": skill_wage_gini,
            **transfer,
            "Employment_Lock_Low": employment_lock["Low"],
            "Employment_Lock_Medium": employment_lock["Medium"],
            "Employment_Lock_High": employment_lock["High"],
            "Employment_Lock_Disparity": employment_disparity,
            "Employment_Lock_Direction": employment_direction,
            "Hours_Lock_Low": hours_lock["Low"],
            "Hours_Lock_Medium": hours_lock["Medium"],
            "Hours_Lock_High": hours_lock["High"],
            "Hours_Lock_Disparity": hours_disparity,
            "System_CE_Gini": ce_result.value,
            "System_CW_Gini": cw_result.value,
            "Policy_Inequality_Index": policy_index,
            "Composite_Inequality_Index": policy_index,
            "Broad_Structural_Inequality_Index": broad,
            "Legacy_Five_Component_Index": legacy,
            "Attributed_Carbon": attribution.attributed_carbon,
            "Unattributed_Carbon": attribution.unattributed_carbon,
            "Attribution_Coverage": attribution.coverage,
            "Zero_Skill_Employment_Positive_Carbon_Sectors": int(
                np.sum((~attribution.eligible_by_sector) & (np.clip(carbon_by_sector, 0.0, None) > 0.0))
            ),
            "High_Carbon_Threshold_80": threshold,
            "High_Carbon_Sector_Count": int(high_carbon_mask.sum()),
            "Primary_Inequality_Specification": SPECIFICATION_ID,
            "Exact_or_Surrogate": EXACT_STATUS,
            "Synthetic_Transmission_Metadata": "mechanically_employment_weighted",
            "Labour_Hours_Role": "secondary_diagnostic_not_objective",
            # Deprecated aliases.
            "Consumption_Herfindahl": float(np.sum(carbon_shares ** 2)),
            "Consumption_Gini_CF": skill_carbon_gini,
            "Consumption_Gini_Emp": skill_employment_gini,
            "Consumption_Gini_Wage": skill_wage_gini,
            "Production_Lock_Gap": deprecated_lock_gap,
            "Transmission_Carbon_Gini": net_carbon_gini,
            "Transmission_Employment_Gini": float(transfer["Net_Employment_Transfer_Gini"]),
            "Transmission_Wage_Gini": float(transfer["Net_Wage_Transfer_Gini"]),
        }
        if not np.isclose(result["Policy_Inequality_Index"], 0.5 * (result["System_CE_Gini"] + result["System_CW_Gini"])):
            raise AssertionError("Policy_Inequality_Index 与 CE/CW 均值不一致。")
        if not np.isclose(result["Composite_Inequality_Index"], result["Policy_Inequality_Index"]):
            raise AssertionError("Composite_Inequality_Index 别名与 primary index 不一致。")
        return result

    @staticmethod
    def _safe_positive_ratio(numerator: float, denominator: float) -> float:
        """分母为正时返回比率，否则返回 NaN。"""
        return float(numerator / denominator) if denominator > 0.0 else np.nan

    def _attach_hour_record_fields(self, record: Dict, outcomes: Mapping[str, object]) -> Dict:
        """将 labour-hour secondary diagnostics 附加到扁平政策记录。"""
        hours_total = float(outcomes["Hours_Total"])
        employment_total = float(outcomes["Employment_Total"])
        wage_total = float(outcomes["Wage_Total"])
        baseline_hours = float(self.baseline.get("Hours_Total", hours_total)) if self.baseline else hours_total
        baseline_employment = float(self.baseline.get("Employment_Total", employment_total)) if self.baseline else employment_total
        baseline_wage = float(self.baseline.get("Wage_Total", wage_total)) if self.baseline else wage_total
        hours_per_worker = self._safe_positive_ratio(hours_total, employment_total)
        baseline_hours_per_worker = self._safe_positive_ratio(baseline_hours, baseline_employment)
        wage_per_hour = self._safe_positive_ratio(wage_total, hours_total)
        baseline_wage_per_hour = self._safe_positive_ratio(baseline_wage, baseline_hours)
        record.update({
            "Hours_Total": hours_total,
            "Hours_Change": self._safe_pct_change(hours_total, baseline_hours),
            "Hours_Per_Worker": hours_per_worker,
            "Hours_Per_Worker_Change": self._safe_pct_change(hours_per_worker, baseline_hours_per_worker),
            "Wage_Per_Hour": wage_per_hour,
            "Wage_Per_Hour_Change": self._safe_pct_change(wage_per_hour, baseline_wage_per_hour),
            "Hours_Unit": "M.hours",
            "Employment_Unit": "thousand persons",
            "Wage_Unit": "M.EUR",
            "Hours_Per_Worker_Unit": "1000 hours/person",
            "Wage_Per_Hour_Unit": "EUR/hour",
            "Hours_Method": str(outcomes.get("Hours_Method", "baseline hours-per-worker x policy employment intensity")),
            "Labour_Hours_Optimized": False,
        })
        for idx, skill in enumerate(SKILLS):
            skill_hours = float(outcomes["Hours_By_Skill"][skill])
            record[f"Hours_{skill}"] = skill_hours
            baseline_lock = float(self.baseline.get(f"Hours_Lock_{skill}", record.get(f"Hours_Lock_{skill}", np.nan))) if self.baseline else float(record.get(f"Hours_Lock_{skill}", np.nan))
            current_lock = float(record.get(f"Hours_Lock_{skill}", np.nan))
            record[f"Hours_Lock_Reduction_{skill}"] = baseline_lock - current_lock
        return record

    def _build_result_record(
        self,
        scenario_name: str,
        outcomes: Dict,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """构建包含主四目标和次级工时诊断的精确记录。"""
        record = super()._build_result_record(scenario_name, outcomes, metadata=metadata)
        record["Primary_Inequality_Specification"] = SPECIFICATION_ID
        record["Engine_Version"] = ENGINE_VERSION
        record["Exact_or_Surrogate"] = EXACT_STATUS
        record["Objective_Count"] = 4
        return self._attach_hour_record_fields(record, outcomes)

    def _extract_transition_metrics(self, result: Dict) -> Dict[str, float]:
        """保留负改善的 signed 四目标，clipped 字段仅作补充诊断。"""
        carbon_signed = -float(result["Carbon_Change"])
        employment_signed = float(result["Employment_Change"])
        wage_signed = float(result["Wage_Change"])
        baseline_index = float(self.baseline.get("Policy_Inequality_Index", 0.0)) if self.baseline else 0.0
        current_index = float(result.get("Policy_Inequality_Index", baseline_index))
        inequality_signed = baseline_index - current_index
        signed_sum = carbon_signed + employment_signed + wage_signed + inequality_signed
        return {
            "Carbon_Reduction_Signed": carbon_signed,
            "Employment_Gain_Signed": employment_signed,
            "Wage_Gain_Signed": wage_signed,
            "Inequality_Reduction_Signed": inequality_signed,
            "Carbon_Reduction_Clipped": max(carbon_signed, 0.0),
            "Employment_Gain_Clipped": max(employment_signed, 0.0),
            "Wage_Gain_Clipped": max(wage_signed, 0.0),
            "Inequality_Reduction_Clipped": max(inequality_signed, 0.0),
            "Four_Objective_Signed_Sum": signed_sum,
            # 兼容列保留有符号值，不裁剪负值。
            "Carbon_Reduction": carbon_signed,
            "Employment_Gain": employment_signed,
            "Wage_Gain": wage_signed,
            "Inequality_Reduction": inequality_signed,
            "Transition_Benefit_Score": signed_sum,
        }

    def _build_policy_state(
        self,
        params: Mapping[str, float],
        uncertainty: Optional[Mapping[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """严格复用 S5 冲击顺序，返回 Y、CO2、employment 和 wage 数组。"""
        u = {
            "theta_tax": 0.8,
            "epsilon_highcarbon": -0.5,
            "eta_renewable_co2": 0.75,
            "lambda_green_jobs": 1.25,
            "xi_shift_effectiveness": 1.0,
            "omega_lowcarbon_substitution": 1.0,
            "rho_reskilling": 1.0,
            "omega_wage_support": 1.0,
        }
        if uncertainty:
            u.update({key: float(value) for key, value in uncertainty.items()})
        y_new = self._apply_carbon_tax_y_shock(
            base_Y=self.Y,
            tax_rate=float(params["tax_rate"]),
            theta_tax=u["theta_tax"],
            epsilon_highcarbon=u["epsilon_highcarbon"],
            substitution_share=0.7,
        )
        co2_new, y_new, employment_new = self._apply_energy_transition_shocks(
            base_co2=self.co2_emissions,
            base_Y=y_new,
            base_E=self.employment["data"],
            renewable_share=float(params["renewable_share"]),
            eta_renewable_co2=u["eta_renewable_co2"],
            lambda_green_jobs=u["lambda_green_jobs"],
            investment_boost=0.20,
        )
        y_new = self._apply_consumption_shift_y_shock(
            base_Y=y_new,
            shift_share=float(np.clip(params["shift_share"], 0.0, 0.6)) * float(np.clip(u["xi_shift_effectiveness"], 0.0, 1.0)),
            substitution_share=float(np.clip(u["omega_lowcarbon_substitution"], 0.0, 1.0)),
        )
        employment_new, wage_new = self._apply_just_transition_shocks(
            base_E=employment_new,
            base_W=self.wage["data"],
            training_support_level=float(np.clip(params["training_support_level"], 0.0, 1.0)) * float(np.clip(u["rho_reskilling"], 0.0, 1.0)),
            wage_support_strength=0.15 * float(np.clip(u["omega_wage_support"], 0.0, 1.0)),
        )
        return y_new, co2_new, employment_new, wage_new

    def evaluate_policy_exact(
        self,
        params: Mapping[str, float],
        uncertainty: Optional[Mapping[str, float]] = None,
        policy_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """使用完整 Y 冲击链和 exact aggregate algebra 评价单个政策向量。"""
        if not self.baseline:
            self.compute_baseline()
        normalized = {
            name: float(params[name])
            for name in self.policy_parameter_order
        }
        y_new, co2_new, employment_new, wage_new = self._build_policy_state(normalized, uncertainty)
        outcomes = self.compute_outcomes_exact_fast(
            y_new,
            co2_new,
            employment_new,
            wage_new,
        )
        record: Dict[str, float] = {
            "Policy_ID": policy_id or "policy_vector",
            **normalized,
            "Carbon_Footprint": float(outcomes["Carbon_Footprint"]),
            "Carbon_Change": self._safe_pct_change(outcomes["Carbon_Footprint"], self.baseline["Carbon_Footprint"]),
            "Employment_Total": float(outcomes["Employment_Total"]),
            "Employment_Change": self._safe_pct_change(outcomes["Employment_Total"], self.baseline["Employment_Total"]),
            "Wage_Total": float(outcomes["Wage_Total"]),
            "Wage_Change": self._safe_pct_change(outcomes["Wage_Total"], self.baseline["Wage_Total"]),
        }
        for skill in SKILLS:
            record[f"Employment_{skill}"] = float(outcomes["Employment_By_Skill"][skill])
            record[f"Wage_{skill}"] = float(outcomes["Wage_By_Skill"][skill])
            record[f"Hours_{skill}"] = float(outcomes["Hours_By_Skill"][skill])
        record["Hours_Total"] = float(outcomes["Hours_Total"])
        record.update(self._compute_scenario_inequality_summary(outcomes))
        record.update(self._extract_transition_metrics(record))
        self._attach_hour_record_fields(record, outcomes)
        policy_cost = self._compute_policy_cost_index(**normalized)
        record["Policy_Cost_Index"] = policy_cost
        record["Net_Benefit_Signed"] = record["Four_Objective_Signed_Sum"] - policy_cost
        record["Primary_Inequality_Specification"] = SPECIFICATION_ID
        record["Exact_or_Surrogate"] = EXACT_STATUS
        record["Objective_Count"] = 4
        record["Labour_Hours_Optimized"] = False
        if uncertainty:
            record.update({name: float(value) for name, value in uncertainty.items()})
        return record

    def evaluate_policy_vector_exact(self, vector: np.ndarray, policy_id: Optional[str] = None) -> Dict[str, float]:
        """将四维决策向量转换为参数并精确评价。"""
        return self.evaluate_policy_exact(
            self._vector_to_policy_params(np.asarray(vector, dtype=float)),
            policy_id=policy_id,
        )

    def run_monte_carlo_exact(
        self,
        n_iterations: int = 1000,
        random_seed: int = 42,
        package_strengths: Optional[Mapping[str, float]] = None,
    ) -> pd.DataFrame:
        """对 S5 执行 8 参数精确 Monte Carlo，保留 signed 改善。"""
        strengths = {
            "tax_rate": 100.0,
            "renewable_share": 0.60,
            "shift_share": 0.20,
            "training_support_level": 0.30,
        }
        if package_strengths:
            strengths.update({key: float(value) for key, value in package_strengths.items()})
        rng = np.random.default_rng(random_seed)
        rows = []
        for iteration in range(int(n_iterations)):
            uncertainty = {
                name: float(rng.uniform(low, high))
                for name, (low, high) in UNCERTAINTY_BOUNDS.items()
            }
            row = self.evaluate_policy_exact(
                strengths,
                uncertainty=uncertainty,
                policy_id=f"mc_{iteration:04d}",
            )
            row["Iteration"] = iteration
            rows.append(row)
        return pd.DataFrame(rows)

    def run_sobol_8param_exact(
        self,
        n_samples: int,
        random_seed: int = 42,
        package_strengths: Optional[Mapping[str, float]] = None,
    ) -> pd.DataFrame:
        """使用 Saltelli/Jansen A、B、AB 设计计算全部 8 参数一阶和总效应。"""
        strengths = {
            "tax_rate": 100.0,
            "renewable_share": 0.60,
            "shift_share": 0.20,
            "training_support_level": 0.30,
        }
        if package_strengths:
            strengths.update({key: float(value) for key, value in package_strengths.items()})
        names = list(UNCERTAINTY_BOUNDS)
        n = int(n_samples)
        rng = np.random.default_rng(random_seed)
        matrix_a = rng.random((n, len(names)))
        matrix_b = rng.random((n, len(names)))

        def scale(vector: np.ndarray) -> Dict[str, float]:
            return {
                name: float(UNCERTAINTY_BOUNDS[name][0] + (UNCERTAINTY_BOUNDS[name][1] - UNCERTAINTY_BOUNDS[name][0]) * vector[idx])
                for idx, name in enumerate(names)
            }

        values_a = {metric: np.zeros(n) for metric in SIGNED_OBJECTIVES}
        values_b = {metric: np.zeros(n) for metric in SIGNED_OBJECTIVES}
        values_ab = {
            name: {metric: np.zeros(n) for metric in SIGNED_OBJECTIVES}
            for name in names
        }
        for sample_idx in range(n):
            result_a = self.evaluate_policy_exact(strengths, uncertainty=scale(matrix_a[sample_idx]))
            result_b = self.evaluate_policy_exact(strengths, uncertainty=scale(matrix_b[sample_idx]))
            for metric in SIGNED_OBJECTIVES:
                values_a[metric][sample_idx] = float(result_a[metric])
                values_b[metric][sample_idx] = float(result_b[metric])
            for parameter_idx, parameter in enumerate(names):
                mixed = matrix_a[sample_idx].copy()
                mixed[parameter_idx] = matrix_b[sample_idx, parameter_idx]
                result_ab = self.evaluate_policy_exact(strengths, uncertainty=scale(mixed))
                for metric in SIGNED_OBJECTIVES:
                    values_ab[parameter][metric][sample_idx] = float(result_ab[metric])

        rows = []
        for parameter in names:
            for metric in SIGNED_OBJECTIVES:
                merged = np.concatenate([values_a[metric], values_b[metric]])
                variance = float(np.var(merged, ddof=1))
                if variance <= 1e-16:
                    first_order = 0.0
                    total_effect = 0.0
                else:
                    first_order = float(
                        np.mean(values_b[metric] * (values_ab[parameter][metric] - values_a[metric])) / variance
                    )
                    total_effect = float(
                        0.5 * np.mean((values_a[metric] - values_ab[parameter][metric]) ** 2) / variance
                    )
                rows.append({
                    "Parameter": parameter,
                    "Metric": metric,
                    "First_Order": first_order,
                    "Total_Effect": total_effect,
                    "N_Base": n,
                    "N_Parameters": len(names),
                    "N_Exact_Evaluations": n * (2 + len(names)),
                    "Primary_Inequality_Specification": SPECIFICATION_ID,
                    "Exact_or_Surrogate": EXACT_STATUS,
                })
        return pd.DataFrame(rows)


# 四目标 NSGA-III 全候选记录

class _RecordedFinalNSGA3Problem(ElementwiseProblem):
    """记录每个被 exact engine 评价的四维政策向量。"""

    def __init__(self, engine: FinalScenarioEngine, seed: int, pop_size: int):
        self.engine = engine
        self.seed = int(seed)
        self.pop_size = int(pop_size)
        self.records: list[Dict[str, float]] = []
        bounds = engine._get_policy_parameter_bounds()
        xl = np.array([bounds[name][0] for name in engine.policy_parameter_order], dtype=float)
        xu = np.array([bounds[name][1] for name in engine.policy_parameter_order], dtype=float)
        super().__init__(n_var=4, n_obj=4, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        evaluation_index = len(self.records) + 1
        record = self.engine.evaluate_policy_vector_exact(
            np.asarray(x, dtype=float),
            policy_id=f"seed{self.seed}_eval{evaluation_index:05d}",
        )
        record["Seed"] = self.seed
        record["Evaluation_Index"] = evaluation_index
        record["Generation_Approx"] = int((evaluation_index - 1) // self.pop_size + 1)
        self.records.append(record)
        out["F"] = -np.array([record[column] for column in SIGNED_OBJECTIVES], dtype=float)


def run_nsga3_seed_exact(
    engine: FinalScenarioEngine,
    seed: int,
    pop_size: int = 70,
    generations: int = 30,
    reference_partitions: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行单种子 signed four-objective NSGA-III，返回全候选与前沿。"""
    if NSGA3 is None or minimize is None or get_reference_directions is None:
        raise ImportError("当前环境缺少 pymoo，无法执行正式 NSGA-III。")
    if not engine.baseline:
        engine.compute_baseline()
    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=int(reference_partitions))
    problem = _RecordedFinalNSGA3Problem(engine, seed=seed, pop_size=pop_size)
    algorithm = NSGA3(pop_size=int(pop_size), ref_dirs=ref_dirs)
    minimize(
        problem,
        algorithm,
        termination=("n_gen", int(generations)),
        seed=int(seed),
        save_history=False,
        verbose=False,
    )
    all_candidates = pd.DataFrame(problem.records)
    if all_candidates.empty:
        return all_candidates, all_candidates.copy()
    unique = all_candidates.drop_duplicates(subset=list(engine.policy_parameter_order), keep="first").copy()
    unique["Pareto_Final"] = _INEQ.non_dominated_flags(unique, SIGNED_OBJECTIVES)
    frontier = unique.loc[unique["Pareto_Final"]].sort_values(
        list(SIGNED_OBJECTIVES),
        ascending=[False] * len(SIGNED_OBJECTIVES),
        kind="mergesort",
    ).reset_index(drop=True)
    all_candidates["NSGA3_Population_Size"] = int(pop_size)
    all_candidates["NSGA3_Generations"] = int(generations)
    all_candidates["NSGA3_Reference_Partitions"] = int(reference_partitions)
    frontier["NSGA3_Population_Size"] = int(pop_size)
    frontier["NSGA3_Generations"] = int(generations)
    frontier["NSGA3_Reference_Partitions"] = int(reference_partitions)
    return all_candidates, frontier


ScenarioEngine = FinalScenarioEngine


__all__ = [
    "FinalScenarioEngine",
    "ScenarioEngine",
    "run_nsga3_seed_exact",
    "SIGNED_OBJECTIVES",
    "UNCERTAINTY_BOUNDS",
    "SPECIFICATION_ID",
    "ENGINE_VERSION",
    "EXACT_STATUS",
]
