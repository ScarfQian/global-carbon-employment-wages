"""Production-side employment and hours locking in high-carbon sectors."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SKILL_ORDER = ("Low", "Medium", "High")


class InequalityMetrics:
    """Production-side skill employment locking diagnostics."""

    def __init__(self, footprint_engine):
        """初始化不平等指标计算器。"""
        self.engine = footprint_engine
        self.mrio = footprint_engine.mrio
        self.L = np.asarray(footprint_engine.L, dtype=float)
        self.Y = np.asarray(footprint_engine.Y, dtype=float)
        self.x = np.asarray(footprint_engine.x, dtype=float).reshape(-1)
        self.regions = list(footprint_engine.regions)
        self.n_products = int(footprint_engine.n_products)
        self.co2_emissions = np.asarray(footprint_engine.co2_emissions, dtype=float).reshape(-1)
        self.employment = footprint_engine.employment
        self.wage = footprint_engine.wage

        self.skill_order = ['Low', 'Medium', 'High']
        self.results = {}
        self._employment_skill_matrix = None
        self._wage_skill_matrix = None

    def compute_production_side_inequality(self) -> Dict:
        """计算生产端不平等指标。"""
        print("\n" + "=" * 70)
        print("【2. 生产端不平等指标】")
        print("=" * 70)

        carbon_by_sector, employment_by_skill_and_sector, _ = self._compute_skill_sector_arrays()

        positive_carbon = carbon_by_sector[carbon_by_sector > 0]
        if positive_carbon.size > 0:
            threshold_high = float(np.percentile(positive_carbon, 80))
            high_carbon_mask = carbon_by_sector >= threshold_high
        else:
            high_carbon_mask = np.zeros_like(carbon_by_sector, dtype=bool)

        employment_lock = {}
        avg_carbon_intensity = {}
        for idx, skill in enumerate(self.skill_order):
            skill_employment = employment_by_skill_and_sector[idx, :]
            total_emp = float(skill_employment.sum())
            high_carbon_emp = float(skill_employment[high_carbon_mask].sum())
            employment_lock[skill] = {
                'Total_Employment': total_emp,
                'High_Carbon_Employment': high_carbon_emp,
                'Lock_Ratio': high_carbon_emp / total_emp if total_emp > 0 else 0.0,
            }
            avg_carbon_intensity[skill] = float(np.sum(skill_employment * carbon_by_sector) / total_emp) if total_emp > 0 else 0.0

        print("\n  [就业锁定度（高碳部门就业占比）]")
        for skill in self.skill_order:
            lock = employment_lock[skill]
            print(f"    {skill}技能: {lock['Lock_Ratio']:.2%} ({lock['High_Carbon_Employment']:.0f}/{lock['Total_Employment']:.0f})")

        print("\n  [平均碳强度（kg CO2/人）]")
        for skill in self.skill_order:
            print(f"    {skill}技能: {avg_carbon_intensity[skill]:.2e}")

        high_lock = employment_lock['High']['Lock_Ratio']
        low_lock = employment_lock['Low']['Lock_Ratio']
        lock_gap = float(low_lock / high_lock) if high_lock > 0 else 0.0

        print("\n  [锁定度差距]")
        print(f"    低技能/高技能锁定度比: {lock_gap:.2f}x")

        self.results['production_inequality'] = {
            'Employment_Lock': employment_lock,
            'Avg_Carbon_Intensity': avg_carbon_intensity,
            'Lock_Gap': lock_gap,
        }
        return self.results['production_inequality']

    def _compute_skill_sector_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回部门碳、技能就业、技能工资三类基础矩阵。"""
        employment_skill_matrix, wage_skill_matrix = self._get_skill_matrices()
        carbon_by_sector = np.clip(self.co2_emissions * self.x, 0.0, None)
        employment_by_skill_and_sector = employment_skill_matrix * self.x.reshape(1, -1)
        wage_by_skill_and_sector = wage_skill_matrix * self.x.reshape(1, -1)
        return carbon_by_sector, employment_by_skill_and_sector, wage_by_skill_and_sector

    def _get_skill_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """懒加载 Low/Medium/High 聚合后的就业与工资系数矩阵。"""
        if self._employment_skill_matrix is None:
            self._employment_skill_matrix = self._aggregate_skill_matrix(
                self.employment['data'],
                self.employment['categories'],
            )
        if self._wage_skill_matrix is None:
            self._wage_skill_matrix = self._aggregate_skill_matrix(
                self.wage['data'],
                self.wage['categories'],
            )
        return self._employment_skill_matrix, self._wage_skill_matrix

    def _aggregate_skill_matrix(self, raw_matrix: np.ndarray, categories: List[str]) -> np.ndarray:
        """将原始技能系数聚合为 Low / Medium / High 三行。"""
        aggregated = np.zeros((len(self.skill_order), raw_matrix.shape[1]), dtype=float)
        for row_idx, category in enumerate(categories):
            lower = str(category).lower()
            if 'low' in lower:
                target_idx = 0
            elif 'medium' in lower:
                target_idx = 1
            elif 'high' in lower:
                target_idx = 2
            else:
                continue
            aggregated[target_idx, :] += np.asarray(raw_matrix[row_idx, :], dtype=float)
        return aggregated


def export_employment_lock_table(payload: Dict, output_dir: str | Path) -> Path:
    """Export skill employment locking from the computed production account."""
    output_path = Path(output_dir) / '260324-employment_lock_ratio_by_skill.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    employment_lock = payload['Employment_Lock']
    lock_gap = float(payload['Lock_Gap'])
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Skill', 'Total_Employment', 'High_Carbon_Employment', 'Employment_Lock_Ratio', 'Low_to_High_Employment_Lock_Gap'])
        for skill in SKILL_ORDER:
            item = employment_lock[skill]
            writer.writerow([skill, item['Total_Employment'], item['High_Carbon_Employment'], item['Lock_Ratio'], lock_gap])
    return output_path


def export_hours_lock_table(hours_skill_gender: pd.DataFrame, output_dir: str | Path) -> Path:
    """Export skill hours locking from the computed skill and gender table."""
    output_path = Path(output_dir) / '260324-high_carbon_hours_share_by_skill.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skill_totals: dict[str, dict[str, float]] = {skill: {'Total_Hours': 0.0, 'High_Carbon_Hours': 0.0} for skill in SKILL_ORDER}
    for row in hours_skill_gender.to_dict('records'):
        skill = row['Skill']
        if skill not in skill_totals:
            continue
        skill_totals[skill]['Total_Hours'] += float(row['Total_Hours'])
        skill_totals[skill]['High_Carbon_Hours'] += float(row['High_Carbon_Hours'])
    total_high_carbon_hours = sum((item['High_Carbon_Hours'] for item in skill_totals.values()))
    low_ratio = skill_totals['Low']['High_Carbon_Hours'] / skill_totals['Low']['Total_Hours']
    high_ratio = skill_totals['High']['High_Carbon_Hours'] / skill_totals['High']['Total_Hours']
    hours_lock_gap = low_ratio / high_ratio if high_ratio > 0.0 else 0.0
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Skill', 'Total_Hours', 'High_Carbon_Hours', 'Hours_Lock_Ratio', 'High_Carbon_Hours_Share', 'Low_to_High_Hours_Lock_Gap'])
        for skill in SKILL_ORDER:
            total_hours = skill_totals[skill]['Total_Hours']
            high_carbon_hours = skill_totals[skill]['High_Carbon_Hours']
            lock_ratio = high_carbon_hours / total_hours if total_hours > 0.0 else 0.0
            high_carbon_share = high_carbon_hours / total_high_carbon_hours if total_high_carbon_hours > 0.0 else 0.0
            writer.writerow([skill, total_hours, high_carbon_hours, lock_ratio, high_carbon_share, hours_lock_gap])
    return output_path
