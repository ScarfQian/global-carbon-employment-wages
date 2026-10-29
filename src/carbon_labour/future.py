#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""四目标未来分析引擎。

包含路径参数、插值和未来快照逻辑。场景计算依赖本包的
``policy`` 模块，不读取既有结果文件，也不把劳动工时纳入优化目标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import policy as _FINAL_SCENARIO_MODULE


FinalScenarioEngine = _FINAL_SCENARIO_MODULE.FinalScenarioEngine

ENGINE_VERSION = "0716-final-future-engine-v1"
EXACT_STATUS = str(_FINAL_SCENARIO_MODULE.EXACT_STATUS)
SURROGATE_STATUS = "not_used"
OBJECTIVE_COLUMNS: Tuple[str, ...] = tuple(_FINAL_SCENARIO_MODULE.SIGNED_OBJECTIVES)
LABOUR_HOURS_OPTIMIZED = False

if len(OBJECTIVE_COLUMNS) != 4 or any("hours" in name.lower() for name in OBJECTIVE_COLUMNS):
    raise ValueError("最终未来引擎必须使用四个非工时优化目标。")


# 未来路径基础配置

YEAR_GRID: Tuple[int, ...] = (2022, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060)
ANCHOR_YEARS: Tuple[int, ...] = (2022, 2030, 2040, 2050, 2060)
KEY_YEARS: Tuple[int, ...] = (2030, 2050, 2060)
PATHWAY_ORDER: Tuple[str, ...] = ("P1", "P2", "P3", "P4")

PATHWAY_LABELS: Dict[str, str] = {
    "P1": "Gradual integrated transition",
    "P2": "Efficiency-first transition",
    "P3": "Justice-first transition",
    "P4": "Delayed carbon lock-in",
}

PROGRESS_ANCHORS: Dict[str, Dict[int, Dict[str, float]]] = {
    "P1": {
        2022: {"q_struct": 0.0, "q_decarb": 0.0, "q_justice": 0.0},
        2030: {"q_struct": 0.25, "q_decarb": 0.30, "q_justice": 0.20},
        2040: {"q_struct": 0.50, "q_decarb": 0.60, "q_justice": 0.45},
        2050: {"q_struct": 0.75, "q_decarb": 0.85, "q_justice": 0.70},
        2060: {"q_struct": 1.00, "q_decarb": 1.00, "q_justice": 0.85},
    },
    "P2": {
        2022: {"q_struct": 0.0, "q_decarb": 0.0, "q_justice": 0.0},
        2030: {"q_struct": 0.40, "q_decarb": 0.45, "q_justice": 0.10},
        2040: {"q_struct": 0.70, "q_decarb": 0.80, "q_justice": 0.20},
        2050: {"q_struct": 0.90, "q_decarb": 0.95, "q_justice": 0.30},
        2060: {"q_struct": 1.00, "q_decarb": 1.00, "q_justice": 0.35},
    },
    "P3": {
        2022: {"q_struct": 0.0, "q_decarb": 0.0, "q_justice": 0.0},
        2030: {"q_struct": 0.35, "q_decarb": 0.35, "q_justice": 0.35},
        2040: {"q_struct": 0.65, "q_decarb": 0.65, "q_justice": 0.65},
        2050: {"q_struct": 0.85, "q_decarb": 0.85, "q_justice": 0.85},
        2060: {"q_struct": 1.00, "q_decarb": 1.00, "q_justice": 1.00},
    },
    "P4": {
        2022: {"q_struct": 0.0, "q_decarb": 0.0, "q_justice": 0.0},
        2030: {"q_struct": 0.10, "q_decarb": 0.08, "q_justice": 0.05},
        2040: {"q_struct": 0.25, "q_decarb": 0.20, "q_justice": 0.10},
        2050: {"q_struct": 0.65, "q_decarb": 0.60, "q_justice": 0.25},
        2060: {"q_struct": 1.00, "q_decarb": 1.00, "q_justice": 0.35},
    },
}

FD_TOTAL_GROWTH_ANCHORS: Dict[str, Dict[int, float]] = {
    "P1": {2022: 1.00, 2030: 1.08, 2040: 1.18, 2050: 1.28, 2060: 1.38},
    "P2": {2022: 1.00, 2030: 1.08, 2040: 1.18, 2050: 1.28, 2060: 1.38},
    "P3": {2022: 1.00, 2030: 1.08, 2040: 1.18, 2050: 1.28, 2060: 1.38},
    "P4": {2022: 1.00, 2030: 1.07, 2040: 1.15, 2050: 1.23, 2060: 1.30},
}

TERMINAL_TARGETS: Dict[str, Dict[str, float]] = {
    "P1": {
        "fd_highcarbon_multiplier": 0.60,
        "fd_green_multiplier": 1.60,
        "fd_service_multiplier": 1.30,
        "co2_highcarbon_multiplier": 0.32,
        "co2_other_multiplier": 0.72,
        "emp_low_green_multiplier": 1.24,
        "emp_medium_green_multiplier": 1.35,
        "emp_high_green_multiplier": 1.30,
        "emp_brown_multiplier": 0.84,
        "wage_low_multiplier": 1.13,
        "wage_medium_multiplier": 1.18,
        "wage_high_multiplier": 1.24,
        "jt_buffer": 0.45,
    },
    "P2": {
        "fd_highcarbon_multiplier": 0.42,
        "fd_green_multiplier": 1.82,
        "fd_service_multiplier": 1.24,
        "co2_highcarbon_multiplier": 0.15,
        "co2_other_multiplier": 0.62,
        "emp_low_green_multiplier": 0.90,
        "emp_medium_green_multiplier": 1.36,
        "emp_high_green_multiplier": 1.56,
        "emp_brown_multiplier": 0.64,
        "wage_low_multiplier": 1.04,
        "wage_medium_multiplier": 1.18,
        "wage_high_multiplier": 1.38,
        "jt_buffer": 0.12,
    },
    "P3": {
        "fd_highcarbon_multiplier": 0.46,
        "fd_green_multiplier": 1.88,
        "fd_service_multiplier": 1.34,
        "co2_highcarbon_multiplier": 0.22,
        "co2_other_multiplier": 0.66,
        "emp_low_green_multiplier": 1.50,
        "emp_medium_green_multiplier": 1.52,
        "emp_high_green_multiplier": 1.34,
        "emp_brown_multiplier": 0.80,
        "wage_low_multiplier": 1.32,
        "wage_medium_multiplier": 1.28,
        "wage_high_multiplier": 1.27,
        "jt_buffer": 0.75,
    },
    "P4": {
        "fd_highcarbon_multiplier": 0.62,
        "fd_green_multiplier": 1.52,
        "fd_service_multiplier": 1.18,
        "co2_highcarbon_multiplier": 0.38,
        "co2_other_multiplier": 0.76,
        "emp_low_green_multiplier": 0.95,
        "emp_medium_green_multiplier": 1.18,
        "emp_high_green_multiplier": 1.22,
        "emp_brown_multiplier": 0.82,
        "wage_low_multiplier": 1.00,
        "wage_medium_multiplier": 1.09,
        "wage_high_multiplier": 1.20,
        "jt_buffer": 0.12,
    },
}


# 世界索引接口


@dataclass(frozen=True)
class FutureWorld:
    """一个可审计的路径、年份和运行范围组合。"""

    scope: str
    pathway: str
    pathway_label: str
    year: int


def _validate_selection(
    selected: Sequence[Any],
    allowed: Sequence[Any],
    label: str,
) -> Tuple[Any, ...]:
    """校验世界索引选择，拒绝范围外值。"""
    normalized = tuple(selected)
    invalid = tuple(value for value in normalized if value not in allowed)
    if invalid:
        raise ValueError(f"{label}超出允许范围: {invalid}")
    return normalized


def _build_worlds(
    scope: str,
    allowed_years: Sequence[int],
    pathways: Optional[Sequence[str]],
    years: Optional[Sequence[int]],
) -> Tuple[FutureWorld, ...]:
    """按指定范围生成稳定有序的未来世界索引。"""
    selected_pathways = PATHWAY_ORDER if pathways is None else _validate_selection(
        pathways,
        PATHWAY_ORDER,
        "路径",
    )
    selected_years = tuple(allowed_years) if years is None else _validate_selection(
        years,
        allowed_years,
        "年份",
    )
    return tuple(
        FutureWorld(scope, pathway, PATHWAY_LABELS[pathway], int(year))
        for pathway in selected_pathways
        for year in selected_years
    )


def baseline_worlds(
    pathways: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
) -> Tuple[FutureWorld, ...]:
    """返回完整年份网格的 baseline 世界，默认共 36 个。"""
    return _build_worlds("baseline", YEAR_GRID, pathways, years)


def core_worlds(
    pathways: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
) -> Tuple[FutureWorld, ...]:
    """返回锚点年份的核心情景世界，默认共 20 个。"""
    return _build_worlds("core", ANCHOR_YEARS, pathways, years)


def key_worlds(
    pathways: Optional[Sequence[str]] = None,
    years: Optional[Sequence[int]] = None,
) -> Tuple[FutureWorld, ...]:
    """返回关键年份世界，默认共 12 个。"""
    return _build_worlds("key", KEY_YEARS, pathways, years)


# 未来快照


class FutureFootprintSnapshot:
    """供最终情景引擎复用的未来年份轻量快照。"""

    def __init__(
        self,
        mrio,
        L: np.ndarray,
        Y: np.ndarray,
        x: np.ndarray,
        regions: List[str],
        n_products: int,
        products: List[str],
        co2_emissions: np.ndarray,
        employment: Dict[str, object],
        wage: Dict[str, object],
        hours: Dict[str, object],
    ) -> None:
        self.mrio = mrio
        self.L = L
        self.Y = Y
        self.x = x
        self.regions = regions
        self.n_products = n_products
        self.products = products
        self.co2_emissions = co2_emissions
        self.employment = employment
        self.wage = wage
        self.hours = hours


# 最终未来分析引擎


class FinalFutureEngine:
    """基于最终四目标情景引擎构造未来世界并执行精确计算。"""

    def __init__(self, footprint_engine) -> None:
        """初始化基础精确场景引擎和未来参数状态。"""
        self.footprint_engine = footprint_engine
        self.base_scenario_engine = FinalScenarioEngine(footprint_engine)
        self.base_scenario_engine.compute_baseline()
        self.years = YEAR_GRID
        self.anchor_years = ANCHOR_YEARS
        self.key_years = KEY_YEARS
        self.pathways = PATHWAY_ORDER
        self.results: Dict[str, pd.DataFrame] = {}
        self.parameter_table: Optional[pd.DataFrame] = None
        self.masks = self._build_future_sector_masks()

    def _build_future_sector_masks(self) -> Dict[str, np.ndarray]:
        """构造未来投影所需的高碳、绿色和服务部门掩码。"""
        scenario = self.base_scenario_engine
        high_carbon_mask = scenario.sector_masks["high_carbon"].copy()
        green_mask = scenario.sector_masks["green_investment"] | scenario.sector_masks["low_carbon"]
        green_mask &= ~high_carbon_mask
        service_mask = scenario._keyword_mask(
            [
                "service",
                "services",
                "business",
                "real estate",
                "hotel",
                "restaurant",
                "education",
                "health",
                "public",
                "administration",
            ]
        )
        service_mask &= ~(high_carbon_mask | green_mask)
        if not np.any(service_mask):
            service_mask = scenario.sector_masks["low_carbon"].copy()
            service_mask &= ~(high_carbon_mask | green_mask)
        return {
            "high_carbon": high_carbon_mask,
            "green": green_mask,
            "service": service_mask,
        }

    @staticmethod
    def _interpolate_year_value(anchor_map: Dict[int, float], year: int) -> float:
        """按锚点年份对中间年份做线性插值。"""
        years = np.array(sorted(anchor_map.keys()), dtype=float)
        values = np.array([anchor_map[int(value)] for value in years], dtype=float)
        return float(np.interp(year, years, values))

    def _get_progress(self, pathway: str, year: int, progress_key: str) -> float:
        """获取结构、脱碳或公正进度值。"""
        anchor_map = {
            anchor_year: anchor_values[progress_key]
            for anchor_year, anchor_values in PROGRESS_ANCHORS[pathway].items()
        }
        return self._interpolate_year_value(anchor_map, year)

    @staticmethod
    def _scale_from_terminal(progress: float, terminal_value: float) -> float:
        """从 2022 基准值平滑推进到 2060 终态值。"""
        return float(1.0 + progress * (terminal_value - 1.0))

    def _build_parameter_row(self, pathway: str, year: int) -> Dict[str, Any]:
        """生成单个路径和年份组合的参数行。"""
        q_struct = self._get_progress(pathway, year, "q_struct")
        q_decarb = self._get_progress(pathway, year, "q_decarb")
        q_justice = self._get_progress(pathway, year, "q_justice")
        terminal = TERMINAL_TARGETS[pathway]
        return {
            "Pathway": pathway,
            "Pathway_Label": PATHWAY_LABELS[pathway],
            "Year": int(year),
            "q_struct": q_struct,
            "q_decarb": q_decarb,
            "q_justice": q_justice,
            "fd_total_growth": self._interpolate_year_value(FD_TOTAL_GROWTH_ANCHORS[pathway], year),
            "fd_highcarbon_multiplier": self._scale_from_terminal(q_struct, terminal["fd_highcarbon_multiplier"]),
            "fd_green_multiplier": self._scale_from_terminal(q_struct, terminal["fd_green_multiplier"]),
            "fd_service_multiplier": self._scale_from_terminal(q_struct, terminal["fd_service_multiplier"]),
            "co2_highcarbon_multiplier": self._scale_from_terminal(q_decarb, terminal["co2_highcarbon_multiplier"]),
            "co2_other_multiplier": self._scale_from_terminal(q_decarb, terminal["co2_other_multiplier"]),
            "emp_low_green_multiplier": self._scale_from_terminal(
                q_struct + 0.20 * q_justice,
                terminal["emp_low_green_multiplier"],
            ),
            "emp_medium_green_multiplier": self._scale_from_terminal(
                q_struct + 0.10 * q_justice,
                terminal["emp_medium_green_multiplier"],
            ),
            "emp_high_green_multiplier": self._scale_from_terminal(q_struct, terminal["emp_high_green_multiplier"]),
            "emp_brown_multiplier": self._scale_from_terminal(q_decarb, terminal["emp_brown_multiplier"]),
            "wage_low_multiplier": self._scale_from_terminal(q_justice, terminal["wage_low_multiplier"]),
            "wage_medium_multiplier": self._scale_from_terminal(
                0.50 * q_struct + 0.50 * q_justice,
                terminal["wage_medium_multiplier"],
            ),
            "wage_high_multiplier": self._scale_from_terminal(q_struct, terminal["wage_high_multiplier"]),
            "jt_buffer": float(q_justice * terminal["jt_buffer"]),
        }

    def generate_pathway_parameters(self) -> pd.DataFrame:
        """生成四条路径和九个年份的 36 行参数表。"""
        rows = [
            self._build_parameter_row(world.pathway, world.year)
            for world in baseline_worlds()
        ]
        frame = pd.DataFrame(rows)
        self.parameter_table = frame
        self.results["pathway_parameters"] = frame
        return frame

    def _get_parameter_row(self, pathway: str, year: int) -> pd.Series:
        """获取单条路径和年份参数。"""
        if self.parameter_table is None:
            self.generate_pathway_parameters()
        assert self.parameter_table is not None
        mask = (self.parameter_table["Pathway"] == pathway) & (self.parameter_table["Year"] == year)
        matched = self.parameter_table.loc[mask]
        if matched.empty:
            raise KeyError(f"未找到路径 {pathway} 年份 {year} 的参数。")
        return matched.iloc[0]

    def _build_sector_multipliers(self, row: pd.Series) -> np.ndarray:
        """构造应用于最终需求的部门乘子。"""
        multipliers = np.ones(self.base_scenario_engine.Y.shape[0], dtype=float)
        multipliers[self.masks["high_carbon"]] = float(row["fd_highcarbon_multiplier"])
        multipliers[self.masks["green"]] = float(row["fd_green_multiplier"])
        multipliers[self.masks["service"]] = float(row["fd_service_multiplier"])
        return multipliers

    def _apply_future_demand(self, row: pd.Series) -> np.ndarray:
        """更新未来最终需求矩阵。"""
        scenario = self.base_scenario_engine
        demand = scenario.Y.copy()
        positive_mask = np.ones(demand.shape[0], dtype=bool)
        scenario._scale_positive_final_demand(demand, positive_mask, float(row["fd_total_growth"]))
        ratio = self._build_sector_multipliers(row) / float(row["fd_total_growth"])
        scenario._scale_positive_final_demand(demand, positive_mask, ratio)
        return demand

    def _apply_future_carbon(self, row: pd.Series) -> np.ndarray:
        """更新未来碳排放系数。"""
        high_mask = self.masks["high_carbon"]
        carbon = self.base_scenario_engine.co2_emissions.copy()
        carbon[high_mask] *= float(row["co2_highcarbon_multiplier"])
        carbon[~high_mask] *= float(row["co2_other_multiplier"])
        return carbon

    @staticmethod
    def _scale_skill_rows(matrix: np.ndarray, row_indices: List[int], factor: float) -> None:
        """对给定技能行整体缩放。"""
        if not row_indices:
            return
        matrix[row_indices, :] *= float(factor)

    def _apply_future_employment(self, row: pd.Series) -> np.ndarray:
        """更新未来就业系数矩阵。"""
        scenario = self.base_scenario_engine
        high_mask = self.masks["high_carbon"]
        green_mask = self.masks["green"]
        justice_buffer = float(row["jt_buffer"])
        employment = scenario.employment["data"].copy()
        low_rows = scenario.employment_skill_rows["Low"]
        medium_rows = scenario.employment_skill_rows["Medium"]
        high_rows = scenario.employment_skill_rows["High"]

        low_green = float(row["emp_low_green_multiplier"]) * (1.0 + 0.20 * justice_buffer)
        medium_green = float(row["emp_medium_green_multiplier"]) * (1.0 + 0.10 * justice_buffer)
        high_green = float(row["emp_high_green_multiplier"]) * (1.0 + 0.03 * justice_buffer)
        brown_base = float(row["emp_brown_multiplier"])
        low_brown = brown_base + 0.30 * justice_buffer * (1.0 - brown_base)
        medium_brown = brown_base + 0.15 * justice_buffer * (1.0 - brown_base)

        scenario._scale_block(employment, low_rows, green_mask, low_green)
        scenario._scale_block(employment, medium_rows, green_mask, medium_green)
        scenario._scale_block(employment, high_rows, green_mask, high_green)
        scenario._scale_block(employment, low_rows, high_mask, low_brown)
        scenario._scale_block(employment, medium_rows, high_mask, medium_brown)
        scenario._scale_block(employment, high_rows, high_mask, brown_base)
        return scenario._ensure_non_negative(employment)

    def _apply_future_wage(self, row: pd.Series) -> np.ndarray:
        """更新未来工资系数矩阵。"""
        scenario = self.base_scenario_engine
        justice_buffer = float(row["jt_buffer"])
        wage = scenario.wage["data"].copy()
        low_factor = float(row["wage_low_multiplier"]) * (1.0 + 0.15 * justice_buffer)
        medium_factor = float(row["wage_medium_multiplier"]) * (1.0 + 0.08 * justice_buffer)
        high_factor = float(row["wage_high_multiplier"]) * (1.0 + 0.03 * justice_buffer)
        self._scale_skill_rows(wage, scenario.wage_skill_rows["Low"], low_factor)
        self._scale_skill_rows(wage, scenario.wage_skill_rows["Medium"], medium_factor)
        self._scale_skill_rows(wage, scenario.wage_skill_rows["High"], high_factor)
        return scenario._ensure_non_negative(wage)

    def _build_future_snapshot(self, row: pd.Series) -> FutureFootprintSnapshot:
        """根据单条参数行构造含次级工时数据的未来快照。"""
        demand = self._apply_future_demand(row)
        carbon = self._apply_future_carbon(row)
        employment = self._apply_future_employment(row)
        wage = self._apply_future_wage(row)
        base_employment = np.asarray(self.footprint_engine.employment["data"], dtype=float)
        base_hours = np.asarray(self.footprint_engine.hours["data"], dtype=float)
        hours_per_worker = np.divide(
            base_hours,
            base_employment,
            out=np.zeros_like(base_hours),
            where=base_employment > 0.0,
        )
        hours = hours_per_worker * employment
        # 线性恒等式保证两种按列汇总顺序完全一致。
        output = self.base_scenario_engine.L @ demand.sum(axis=1)
        return FutureFootprintSnapshot(
            mrio=getattr(self.footprint_engine, "mrio", None),
            L=np.asarray(self.footprint_engine.L, dtype=float),
            Y=np.asarray(demand, dtype=float),
            x=np.asarray(output, dtype=float),
            regions=list(getattr(self.footprint_engine, "regions", [])),
            n_products=int(getattr(self.footprint_engine, "n_products", demand.shape[0])),
            products=list(getattr(self.footprint_engine, "products", [])),
            co2_emissions=np.asarray(carbon, dtype=float),
            employment={
                "data": np.asarray(employment, dtype=float),
                "categories": list(self.footprint_engine.employment["categories"]),
            },
            wage={
                "data": np.asarray(wage, dtype=float),
                "categories": list(self.footprint_engine.wage["categories"]),
            },
            hours={
                "data": np.asarray(hours, dtype=float),
                "categories": list(self.footprint_engine.hours["categories"]),
            },
        )

    def _create_future_scenario_engine(self, row: pd.Series):
        """为单个未来世界创建最终精确场景引擎。"""
        scenario = FinalScenarioEngine(self._build_future_snapshot(row))
        scenario.compute_baseline()
        return scenario

    @staticmethod
    def _attach_execution_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """显式附加精确状态、surrogate 状态和目标边界。"""
        record["Future_Engine_Version"] = ENGINE_VERSION
        record["Exact_or_Surrogate"] = EXACT_STATUS
        record["Surrogate_Status"] = SURROGATE_STATUS
        record["Surrogate_Used"] = False
        record["Objective_Count"] = len(OBJECTIVE_COLUMNS)
        record["Objective_Set"] = "|".join(OBJECTIVE_COLUMNS)
        record["Labour_Hours_Optimized"] = LABOUR_HOURS_OPTIMIZED
        return record

    @classmethod
    def _annotate_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """为结果表逐列写入统一执行状态。"""
        annotated = frame.copy()
        annotated["Future_Engine_Version"] = ENGINE_VERSION
        annotated["Exact_or_Surrogate"] = EXACT_STATUS
        annotated["Surrogate_Status"] = SURROGATE_STATUS
        annotated["Surrogate_Used"] = False
        annotated["Objective_Count"] = len(OBJECTIVE_COLUMNS)
        annotated["Objective_Set"] = "|".join(OBJECTIVE_COLUMNS)
        annotated["Labour_Hours_Optimized"] = LABOUR_HOURS_OPTIMIZED
        return annotated

    def project_future_baseline(
        self,
        pathways: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """执行完整年份网格的未来 baseline，默认返回 36 行。"""
        rows = []
        for world in baseline_worlds(pathways, years):
            params = self._get_parameter_row(world.pathway, world.year)
            snapshot = self._build_future_snapshot(params)
            outcomes = self.base_scenario_engine.compute_outcomes(
                Y=snapshot.Y,
                co2_emissions=snapshot.co2_emissions,
                employment_data=snapshot.employment["data"],
                wage_data=snapshot.wage["data"],
            )
            record = self.base_scenario_engine._build_result_record(
                f"FB_{world.pathway}_{world.year}",
                outcomes,
                metadata={
                    "World_Scope": world.scope,
                    "Pathway": world.pathway,
                    "Pathway_Label": world.pathway_label,
                    "Year": world.year,
                },
            )
            for column in params.index:
                record.setdefault(column, params[column])
            rows.append(self._attach_execution_metadata(record))
        frame = pd.DataFrame(rows)
        self.results["baseline_projection"] = frame
        return frame

    def run_future_core_scenarios(
        self,
        pathways: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """在五个锚点年份执行核心情景，默认覆盖 20 个世界。"""
        scenario_rows: List[pd.DataFrame] = []
        inequality_rows: List[pd.DataFrame] = []
        for world in core_worlds(pathways, years):
            params = self._get_parameter_row(world.pathway, world.year)
            future_scenario = self._create_future_scenario_engine(params)
            core_frame = future_scenario.run_core_scenarios().copy()
            core_frame["World_Scope"] = world.scope
            core_frame["Pathway"] = world.pathway
            core_frame["Pathway_Label"] = world.pathway_label
            core_frame["Year"] = world.year
            for column in params.index:
                if column not in core_frame.columns:
                    core_frame[column] = params[column]
            scenario_rows.append(self._annotate_frame(core_frame))

            inequality_frame = future_scenario.scenarios["triple_inequality"].copy()
            inequality_frame["World_Scope"] = world.scope
            inequality_frame["Pathway"] = world.pathway
            inequality_frame["Pathway_Label"] = world.pathway_label
            inequality_frame["Year"] = world.year
            inequality_rows.append(self._annotate_frame(inequality_frame))

        scenario_frame = pd.concat(scenario_rows, ignore_index=True) if scenario_rows else pd.DataFrame()
        inequality_frame = pd.concat(inequality_rows, ignore_index=True) if inequality_rows else pd.DataFrame()
        self.results["core_scenarios"] = scenario_frame
        self.results["triple_inequality"] = inequality_frame
        return scenario_frame, inequality_frame

    def iter_key_world_engines(
        self,
        pathways: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
    ) -> Iterator[Tuple[FutureWorld, Any]]:
        """逐个生成 12 个关键世界及其最终精确场景引擎。"""
        for world in key_worlds(pathways, years):
            params = self._get_parameter_row(world.pathway, world.year)
            yield world, self._create_future_scenario_engine(params)

    def run_key_worlds(
        self,
        runner: Callable[[Any, FutureWorld], Any],
        pathways: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
    ) -> List[Any]:
        """对关键世界执行调用方提供的精确分析函数。"""
        return [
            runner(scenario, world)
            for world, scenario in self.iter_key_world_engines(pathways, years)
        ]


FutureAnalysisEngine = FinalFutureEngine

__all__ = [
    "ANCHOR_YEARS",
    "ENGINE_VERSION",
    "EXACT_STATUS",
    "FinalFutureEngine",
    "FinalScenarioEngine",
    "FutureAnalysisEngine",
    "FutureFootprintSnapshot",
    "FutureWorld",
    "KEY_YEARS",
    "LABOUR_HOURS_OPTIMIZED",
    "OBJECTIVE_COLUMNS",
    "PATHWAY_ORDER",
    "SURROGATE_STATUS",
    "YEAR_GRID",
    "baseline_worlds",
    "core_worlds",
    "key_worlds",
]
