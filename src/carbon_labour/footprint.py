#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 EXIOBASE 的碳、就业、工资和工时足迹核算。"""

import pymrio
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FootprintEngine:
    """足迹计算引擎类"""

    def __init__(self, data_path: str):
        """
        初始化足迹计算引擎

        Args:
            data_path: EXIOBASE数据路径（如 '/path/to/IOT_2022_pxp'）
        """
        self.data_path = data_path
        self.mrio = None
        self.n_regions = None
        self.n_products = None
        self.n_fc = None
        self.regions = None
        self.products = None

        # 核心矩阵
        self.L = None  # Leontief逆矩阵
        self.Y = None  # 最终需求矩阵

        # 扩展数据
        self.co2_emissions = None  # CO2排放向量
        self.employment = None     # 就业人数矩阵
        self.hours = None          # 工作时间矩阵
        self.wage = None           # 工资矩阵

        self.results = {}

    def load_data(self) -> bool:
        """
        加载EXIOBASE数据

        Returns:
            bool: 加载是否成功
        """
        print("=" * 70)
        print("步骤1: 加载EXIOBASE数据")
        print("=" * 70)

        try:
            self.mrio = pymrio.parse_exiobase3(self.data_path)
            print(f"  ✓ 数据加载成功")

            self.mrio.calc_all()
            print(f"  ✓ Leontief逆矩阵计算完成")

            self._extract_dimension_info()

            self._extract_core_matrices()

            self._extract_extension_data()

            self._print_summary()
            return True

        except Exception as e:
            print(f"  ✗ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_dimension_info(self):
        """提取维度信息"""
        self.regions = self.mrio.Y.index.get_level_values(0).unique().tolist()
        self.n_regions = len(self.regions)

        self.products = self.mrio.Y.index.get_level_values(1).unique().tolist()
        self.n_products = len(self.products)

        self.n_fc = self.mrio.Y.shape[1]

        print(f"  - 区域数: {self.n_regions}")
        print(f"  - 产品数: {self.n_products}")
        print(f"  - 最终需求类别: {self.n_fc}")
        print(f"  - 总维度: {self.n_regions} × {self.n_products} = {self.n_regions * self.n_products}")

    def _extract_core_matrices(self):
        """提取核心矩阵"""
        self.L = self.mrio.L.values  # (9800, 9800)
        self.Y = self.mrio.Y.values  # (9800, 343)

        print(f"  - L矩阵: {self.L.shape}")
        print(f"  - Y矩阵: {self.Y.shape}")

    def _extract_extension_data(self):
        """提取扩展数据（CO2、就业人数、工作时间、工资）"""
        print("\n  提取扩展数据:")

        self.x = self.mrio.x.values.flatten()  # (9800,)

        # S contains satellite coefficients; F contains direct flows.
        co2_idx = [i for i, name in enumerate(self.mrio.air_emissions.S.index)
                   if 'CO2 - combustion' in name and 'bio' not in name]
        if co2_idx:
            s_co2 = self.mrio.air_emissions.S.iloc[co2_idx].values.sum(axis=0)
            self.co2_emissions = s_co2
            self.co2_total = (s_co2 * self.x).sum()
            print(f"    ✓ CO2排放: {len(co2_idx)}类 → 排放系数向量 {self.co2_emissions.shape}")
            print(f"      - 总排放: {self.co2_total:.2e} kg = {self.co2_total/1e9:.2f} 百万吨")

        if hasattr(self.mrio.employment, 'S'):
            emp_s = self.mrio.employment.S
            emp_categories = emp_s.index.tolist()

            emp_people_idx = [i for i, cat in enumerate(emp_categories) if 'hours' not in cat.lower()]
            emp_hours_idx = [i for i, cat in enumerate(emp_categories) if 'hours' in cat.lower()]

            emp_people_data = emp_s.iloc[emp_people_idx].values
            emp_hours_data = emp_s.iloc[emp_hours_idx].values

            self.employment = {
                'data': emp_people_data,
                'categories': [emp_categories[i] for i in emp_people_idx],
            }
            self.hours = {
                'data': emp_hours_data,
                'categories': [emp_categories[i] for i in emp_hours_idx],
            }
            print(f"    ✓ 就业（人数）: {len(emp_people_idx)}类 - 使用系数矩阵S")
            print(f"    ✓ 工作时间（小时）: {len(emp_hours_idx)}类 - 使用系数矩阵S")
        else:
            emp_df = self.mrio.employment.F
            emp_categories = emp_df.index.tolist()
            emp_people_idx = [i for i, cat in enumerate(emp_categories) if 'hours' not in cat.lower()]
            emp_hours_idx = [i for i, cat in enumerate(emp_categories) if 'hours' in cat.lower()]

            emp_people_f = emp_df.iloc[emp_people_idx].values
            emp_hours_f = emp_df.iloc[emp_hours_idx].values
            emp_people_data = np.where(np.isfinite(emp_people_f / self.x), emp_people_f / self.x, 0)
            emp_hours_data = np.where(np.isfinite(emp_hours_f / self.x), emp_hours_f / self.x, 0)

            self.employment = {
                'data': emp_people_data,
                'categories': [emp_categories[i] for i in emp_people_idx],
            }
            self.hours = {
                'data': emp_hours_data,
                'categories': [emp_categories[i] for i in emp_hours_idx],
            }
            print(f"    ✓ 就业（人数）: {len(emp_people_idx)}类 - 从F转换为系数")
            print(f"    ✓ 工作时间（小时）: {len(emp_hours_idx)}类 - 从F转换为系数")

        for cat in self.employment['categories']:
            print(f"      - {cat}")
        for cat in self.hours['categories']:
            print(f"      - {cat}")

        if hasattr(self.mrio.factor_inputs, 'S'):
            wage_s = self.mrio.factor_inputs.S
            wage_categories = wage_s.index.tolist()
            wage_data = wage_s.values
            self.wage = {
                'data': wage_data,
                'categories': wage_categories,
            }
            print(f"    ✓ 工资: {len(wage_categories)}类 - 使用系数矩阵S")
        else:
            wage_df = self.mrio.factor_inputs.F
            wage_categories = wage_df.index.tolist()
            wage_f = wage_df.values
            wage_data = np.where(np.isfinite(wage_f / self.x), wage_f / self.x, 0)
            self.wage = {
                'data': wage_data,
                'categories': wage_categories,
            }
            print(f"    ✓ 工资: {len(wage_categories)}类 - 从F转换为系数")

        for cat in self.wage['categories']:
            print(f"      - {cat}")

    def _print_summary(self):
        """打印数据摘要"""
        emp_coef = self.employment['data']
        total_employment = (emp_coef * self.x).sum()

        hours_coef = self.hours['data']
        total_hours = (hours_coef * self.x).sum()

        wage_coef = self.wage['data']
        total_wage = (wage_coef * self.x).sum()

        print("\n" + "=" * 70)
        print("数据摘要")
        print("=" * 70)
        print(f"  总产出: {self.x.sum():.2e} 百万欧元")
        print(f"  总碳排放: {self.co2_total:.2e} kg = {self.co2_total/1e9:.2f} 百万吨")
        print(f"  总就业: {total_employment:.2f} thousand persons = {total_employment/1e3:.2f} million persons")
        print(f"  总工作时间: {total_hours:.2e} M.hr")
        print(f"  总工资: {total_wage:.2f} M.EUR = {total_wage/1e6:.2f} trillion EUR")


    def _get_final_demand_region_labels(self) -> pd.Index:
        """
        提取最终需求列对应的地区标签。

        Returns:
            pd.Index: 与 Y 列一一对应的地区标签
        """
        y_columns = self.mrio.Y.columns
        if isinstance(y_columns, pd.MultiIndex):
            return y_columns.get_level_values(0)
        return pd.Index(y_columns, name='region')


    def _get_producer_region_labels(self) -> pd.Index:
        """
        提取生产侧行对应的地区标签。

        Returns:
            pd.Index: 与生产行一一对应的地区标签
        """
        y_index = self.mrio.Y.index
        if isinstance(y_index, pd.MultiIndex):
            return y_index.get_level_values(0)
        return pd.Index(y_index, name='region')

    def compute_consumption_side_country_triplets(self) -> pd.DataFrame:
        """
        按最终需求国家计算消费侧隐含碳-就业-工时-工资账户。

        Returns:
            DataFrame: 国家消费侧隐含账户结果
        """
        print("\n" + "=" * 70)
        print("步骤2b: 计算消费侧国家三元组（隐含碳/就业/工时/工资）")
        print("=" * 70)

        emp_total_intensity = np.asarray(self.employment['data'], dtype=float).sum(axis=0)
        hours_total_intensity = np.asarray(self.hours['data'], dtype=float).sum(axis=0) if self.hours else np.zeros_like(emp_total_intensity)
        wage_total_intensity = np.asarray(self.wage['data'], dtype=float).sum(axis=0)

        carbon_full = np.asarray(self.co2_emissions, dtype=float) @ self.L @ self.Y
        employment_full = emp_total_intensity @ self.L @ self.Y
        hours_full = hours_total_intensity @ self.L @ self.Y
        wage_full = wage_total_intensity @ self.L @ self.Y
        fd_regions = self._get_final_demand_region_labels()

        df = pd.DataFrame(
            {
                'Embodied_Carbon': pd.Series(carbon_full).groupby(fd_regions).sum(),
                'Embodied_Employment': pd.Series(employment_full).groupby(fd_regions).sum(),
                'Embodied_Hours': pd.Series(hours_full).groupby(fd_regions).sum(),
                'Embodied_Wage': pd.Series(wage_full).groupby(fd_regions).sum(),
            }
        ).sort_values('Embodied_Carbon', ascending=False)
        df.index.name = 'Region'
        df = df.reset_index()

        for column in ['Embodied_Carbon', 'Embodied_Employment', 'Embodied_Hours', 'Embodied_Wage']:
            total = float(df[column].sum())
            share_col = column.replace('Embodied_', '') + '_Share'
            df[share_col] = df[column] / total if total > 0.0 else 0.0

        print("  ✓ 消费侧国家三元组计算完成")
        print(df.head(10).to_string(index=False))

        self.results['consumption_side_country_triplets'] = df
        return df

    def compute_production_side_country_triplets(self) -> pd.DataFrame:
        """
        按生产地国家计算生产侧直接碳-就业-工时-工资账户。

        Returns:
            DataFrame: 国家生产侧直接账户结果
        """
        print("\n" + "=" * 70)
        print("步骤2c: 计算生产侧国家三元组（直接碳/就业/工时/工资）")
        print("=" * 70)

        carbon_direct = np.asarray(self.co2_emissions, dtype=float) * np.asarray(self.x, dtype=float)
        employment_direct = np.asarray(self.employment['data'], dtype=float).sum(axis=0) * np.asarray(self.x, dtype=float)
        hours_direct = (
            np.asarray(self.hours['data'], dtype=float).sum(axis=0) * np.asarray(self.x, dtype=float)
            if self.hours else np.zeros_like(employment_direct)
        )
        wage_direct = np.asarray(self.wage['data'], dtype=float).sum(axis=0) * np.asarray(self.x, dtype=float)
        producer_regions = self._get_producer_region_labels()

        df = pd.DataFrame(
            {
                'Direct_Carbon': pd.Series(carbon_direct).groupby(producer_regions).sum(),
                'Direct_Employment': pd.Series(employment_direct).groupby(producer_regions).sum(),
                'Direct_Hours': pd.Series(hours_direct).groupby(producer_regions).sum(),
                'Direct_Wage': pd.Series(wage_direct).groupby(producer_regions).sum(),
            }
        ).sort_values('Direct_Carbon', ascending=False)
        df.index.name = 'Region'
        df = df.reset_index()

        for column in ['Direct_Carbon', 'Direct_Employment', 'Direct_Hours', 'Direct_Wage']:
            total = float(df[column].sum())
            share_col = column.replace('Direct_', '') + '_Share'
            df[share_col] = df[column] / total if total > 0.0 else 0.0

        print("  ✓ 生产侧国家三元组计算完成")
        print(df.head(10).to_string(index=False))

        self.results['production_side_country_triplets'] = df
        return df


    def _parse_wage_skill_label(self, category: str) -> str:
        """
        解析工资类别中的技能标签，仅保留低/中/高技能三类。

        Args:
            category: 工资扩展分类标签

        Returns:
            str: 技能标签，无法识别时返回 Unknown
        """
        if 'Low' in category:
            return 'Low'
        if 'Medium' in category:
            return 'Medium'
        if 'High' in category:
            return 'High'
        return 'Unknown'

    def _build_employment_skill_intensity_matrix(self) -> Tuple[np.ndarray, list]:
        """
        将就业扩展聚合为 Low/Medium/High 三类技能的强度矩阵。

        Returns:
            Tuple[np.ndarray, list]: (技能×部门矩阵, 技能顺序)
        """
        skills = ['Low', 'Medium', 'High']
        skill_to_idx = {skill: idx for idx, skill in enumerate(skills)}
        emp_data = np.asarray(self.employment['data'], dtype=float)
        matrix = np.zeros((len(skills), emp_data.shape[1]), dtype=float)

        for row_idx, category in enumerate(self.employment['categories']):
            skill, _ = self._parse_skill_gender_label(category)
            if skill in skill_to_idx:
                matrix[skill_to_idx[skill], :] += emp_data[row_idx, :]

        return matrix, skills

    def _build_wage_skill_intensity_matrix(self) -> Tuple[np.ndarray, list]:
        """
        将工资扩展聚合为 Low/Medium/High 三类技能的强度矩阵。

        说明：为保证技能口径与就业一致，这里不纳入 Other 类别。

        Returns:
            Tuple[np.ndarray, list]: (技能×部门矩阵, 技能顺序)
        """
        skills = ['Low', 'Medium', 'High']
        skill_to_idx = {skill: idx for idx, skill in enumerate(skills)}
        wage_data = np.asarray(self.wage['data'], dtype=float)
        matrix = np.zeros((len(skills), wage_data.shape[1]), dtype=float)

        for row_idx, category in enumerate(self.wage['categories']):
            skill = self._parse_wage_skill_label(category)
            if skill in skill_to_idx:
                matrix[skill_to_idx[skill], :] += wage_data[row_idx, :]

        return matrix, skills

    def _assemble_country_skill_account_table(
        self,
        region_labels: pd.Index,
        skills: list,
        carbon_by_country: pd.Series,
        employment_by_skill: np.ndarray,
        hours_by_skill: np.ndarray,
        wage_by_skill: np.ndarray,
        carbon_total_col: str,
        carbon_share_col: str,
        employment_col: str,
        hours_col: str,
        wage_col: str,
    ) -> pd.DataFrame:
        """
        组织国家×技能账户表，并补充国家内与全球份额指标。

        Args:
            region_labels: 区域标签
            skills: 技能顺序
            carbon_by_country: 国家层碳总量
            employment_by_skill: 技能×区域维度的就业结果
            hours_by_skill: 技能×区域维度的工时结果
            wage_by_skill: 技能×区域维度的工资结果
            carbon_total_col: 国家层碳总量列名
            carbon_share_col: 国家层碳份额列名
            employment_col: 就业列名
            hours_col: 工时列名
            wage_col: 工资列名

        Returns:
            pd.DataFrame: 国家×技能账户表
        """
        rows = []
        region_labels = pd.Index(region_labels)
        carbon_total = float(carbon_by_country.sum())
        employment_total = float(np.asarray(employment_by_skill, dtype=float).sum())
        hours_total = float(np.asarray(hours_by_skill, dtype=float).sum())
        wage_total = float(np.asarray(wage_by_skill, dtype=float).sum())

        for region in region_labels.unique():
            region_mask = (region_labels == region)
            employment_country = np.asarray(employment_by_skill[:, region_mask], dtype=float).sum(axis=1)
            hours_country = np.asarray(hours_by_skill[:, region_mask], dtype=float).sum(axis=1)
            wage_country = np.asarray(wage_by_skill[:, region_mask], dtype=float).sum(axis=1)
            employment_country_total = float(employment_country.sum())
            hours_country_total = float(hours_country.sum())
            wage_country_total = float(wage_country.sum())
            carbon_value = float(carbon_by_country.loc[region]) if region in carbon_by_country.index else 0.0
            carbon_share = carbon_value / carbon_total if carbon_total > 0.0 else 0.0

            for skill_idx, skill in enumerate(skills):
                employment_value = float(employment_country[skill_idx])
                hours_value = float(hours_country[skill_idx])
                wage_value = float(wage_country[skill_idx])
                rows.append(
                    {
                        'Region': region,
                        'Skill': skill,
                        carbon_total_col: carbon_value,
                        carbon_share_col: carbon_share,
                        employment_col: employment_value,
                        f'{employment_col}_Share_Within_Country': (
                            employment_value / employment_country_total if employment_country_total > 0.0 else 0.0
                        ),
                        f'{employment_col}_Share_Global': (
                            employment_value / employment_total if employment_total > 0.0 else 0.0
                        ),
                        hours_col: hours_value,
                        f'{hours_col}_Share_Within_Country': (
                            hours_value / hours_country_total if hours_country_total > 0.0 else 0.0
                        ),
                        f'{hours_col}_Share_Global': (
                            hours_value / hours_total if hours_total > 0.0 else 0.0
                        ),
                        wage_col: wage_value,
                        f'{wage_col}_Share_Within_Country': (
                            wage_value / wage_country_total if wage_country_total > 0.0 else 0.0
                        ),
                        f'{wage_col}_Share_Global': wage_value / wage_total if wage_total > 0.0 else 0.0,
                    }
                )

        df = pd.DataFrame(rows)
        df['Skill'] = pd.Categorical(df['Skill'], categories=skills, ordered=True)
        df = df.sort_values([carbon_total_col, 'Region', 'Skill'], ascending=[False, True, True]).reset_index(drop=True)
        return df

    def compute_consumption_side_country_skill_triplets(self) -> pd.DataFrame:
        """
        计算消费侧国家×技能隐含就业/工时/工资账户，并附带国家层隐含碳总量作为对照。

        Returns:
            pd.DataFrame: 国家×技能消费侧账户表
        """
        print("\n" + "=" * 70)
        print("步骤2d: 计算消费侧国家×技能账户（隐含就业/工时/工资）")
        print("=" * 70)

        employment_skill_matrix, skills = self._build_employment_skill_intensity_matrix()
        hours_skill_matrix, _ = self._build_hours_skill_intensity_matrix()
        wage_skill_matrix, _ = self._build_wage_skill_intensity_matrix()

        employment_by_skill_fd = employment_skill_matrix @ self.L @ self.Y
        hours_by_skill_fd = hours_skill_matrix @ self.L @ self.Y
        wage_by_skill_fd = wage_skill_matrix @ self.L @ self.Y
        carbon_fd = np.asarray(self.co2_emissions, dtype=float) @ self.L @ self.Y
        fd_regions = self._get_final_demand_region_labels()
        carbon_by_country = pd.Series(carbon_fd).groupby(fd_regions).sum()

        df = self._assemble_country_skill_account_table(
            region_labels=fd_regions,
            skills=skills,
            carbon_by_country=carbon_by_country,
            employment_by_skill=employment_by_skill_fd,
            hours_by_skill=hours_by_skill_fd,
            wage_by_skill=wage_by_skill_fd,
            carbon_total_col='Embodied_Carbon_Country_Total',
            carbon_share_col='Embodied_Carbon_Country_Share',
            employment_col='Embodied_Employment',
            hours_col='Embodied_Hours',
            wage_col='Embodied_Wage',
        )

        print("  ✓ 消费侧国家×技能账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['consumption_side_country_skill_triplets'] = df
        return df

    def compute_production_side_country_skill_triplets(self) -> pd.DataFrame:
        """
        计算生产侧国家×技能直接就业/工时/工资账户，并附带国家层直接碳总量作为对照。

        Returns:
            pd.DataFrame: 国家×技能生产侧账户表
        """
        print("\n" + "=" * 70)
        print("步骤2e: 计算生产侧国家×技能账户（直接就业/工时/工资）")
        print("=" * 70)

        employment_skill_matrix, skills = self._build_employment_skill_intensity_matrix()
        hours_skill_matrix, _ = self._build_hours_skill_intensity_matrix()
        wage_skill_matrix, _ = self._build_wage_skill_intensity_matrix()

        x = np.asarray(self.x, dtype=float)
        employment_by_skill_sector = employment_skill_matrix * x.reshape(1, -1)
        hours_by_skill_sector = hours_skill_matrix * x.reshape(1, -1)
        wage_by_skill_sector = wage_skill_matrix * x.reshape(1, -1)
        carbon_sector = np.asarray(self.co2_emissions, dtype=float) * x
        producer_regions = self._get_producer_region_labels()
        carbon_by_country = pd.Series(carbon_sector).groupby(producer_regions).sum()

        df = self._assemble_country_skill_account_table(
            region_labels=producer_regions,
            skills=skills,
            carbon_by_country=carbon_by_country,
            employment_by_skill=employment_by_skill_sector,
            hours_by_skill=hours_by_skill_sector,
            wage_by_skill=wage_by_skill_sector,
            carbon_total_col='Direct_Carbon_Country_Total',
            carbon_share_col='Direct_Carbon_Country_Share',
            employment_col='Direct_Employment',
            hours_col='Direct_Hours',
            wage_col='Direct_Wage',
        )

        print("  ✓ 生产侧国家×技能账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['production_side_country_skill_triplets'] = df
        return df

    def _parse_skill_gender_label(self, category: str) -> Tuple[str, str]:
        """
        解析就业类别中的技能与性别标签。

        注意：`female` 包含 `male` 子串，因此必须先判断 `female`。

        Args:
            category: EXIOBASE 就业分类标签

        Returns:
            Tuple[str, str]: (技能, 性别)
        """
        category_lower = category.lower()

        if 'Low-skilled' in category:
            skill = 'Low'
        elif 'Medium-skilled' in category:
            skill = 'Medium'
        elif 'High-skilled' in category:
            skill = 'High'
        else:
            skill = 'Unknown'

        if 'female' in category_lower:
            gender = 'Female'
        elif 'male' in category_lower:
            gender = 'Male'
        else:
            gender = 'Unknown'

        return skill, gender

    def _validate_employment_skill_gender_coverage(self, skill_gender_map: Dict[str, Tuple[str, str]]):
        """
        校验就业人数分类是否完整覆盖三类技能与两类性别。

        Args:
            skill_gender_map: 类别到 (技能, 性别) 的映射

        Raises:
            ValueError: 若技能×性别组合不完整
        """
        observed_pairs = {
            pair for pair in skill_gender_map.values()
            if pair[0] != 'Unknown' and pair[1] != 'Unknown'
        }
        expected_pairs = {
            ('Low', 'Male'), ('Low', 'Female'),
            ('Medium', 'Male'), ('Medium', 'Female'),
            ('High', 'Male'), ('High', 'Female'),
        }

        if observed_pairs != expected_pairs:
            missing_pairs = sorted(expected_pairs - observed_pairs)
            extra_pairs = sorted(observed_pairs - expected_pairs)
            messages = ['就业人数分类未完整覆盖 3×2 技能-性别组合']
            if missing_pairs:
                messages.append('缺失: ' + ', '.join(f'{skill}_{gender}' for skill, gender in missing_pairs))
            if extra_pairs:
                messages.append('额外: ' + ', '.join(f'{skill}_{gender}' for skill, gender in extra_pairs))
            raise ValueError('；'.join(messages))

    def compute_carbon_footprint_by_country(self) -> pd.DataFrame:
        """
        按最终需求国家计算碳足迹。

        先计算各最终需求列的足迹，再按 Y 列索引中的国家汇总。

        Returns:
            DataFrame: 国家碳足迹结果
        """
        print("\n" + "=" * 70)
        print("步骤2: 计算碳足迹（按国家）")
        print("=" * 70)

        y_index = self.mrio.Y.index
        y_columns = self.mrio.Y.columns
        print(f"  - Y行索引层级: {y_index.names}")
        print(f"  - Y列索引层级: {y_columns.names}")

        carbon_footprint_full = self.co2_emissions @ self.L @ self.Y

        print("  ✓ 碳足迹计算完成")
        print(f"    - 维度: {carbon_footprint_full.shape}")
        print(f"    - 总碳足迹: {carbon_footprint_full.sum():.2e}")

        fd_regions = self._get_final_demand_region_labels()
        carbon_by_country = pd.Series(carbon_footprint_full).groupby(fd_regions).sum().sort_values(ascending=False)

        df = carbon_by_country.to_frame(name='Carbon_Footprint')
        df['Carbon_Footprint_Mt'] = df['Carbon_Footprint'] / 1e9

        print("\n  各国碳足迹（按最终需求国家汇总，Top 10）:")
        for i, (region, row) in enumerate(df.head(10).iterrows()):
            print(f"    {i+1}. {region}: {row['Carbon_Footprint_Mt']:.2f} Mt")

        self.results['carbon_by_country'] = df
        return df

    def compute_employment_footprint_by_skill_gender(self) -> pd.DataFrame:
        """
        按技能×性别计算就业足迹。

        Returns:
            DataFrame: 技能×性别就业足迹
        """
        print("\n" + "=" * 70)
        print("步骤3: 计算就业足迹（按技能×性别）")
        print("=" * 70)

        emp_data = self.employment['data']
        categories = self.employment['categories']

        skill_gender_map = {
            cat: self._parse_skill_gender_label(cat)
            for cat in categories
        }
        self._validate_employment_skill_gender_coverage(skill_gender_map)

        employment_by_skill_gender = {}

        for i, cat in enumerate(categories):
            skill, gender = skill_gender_map[cat]
            if skill == 'Unknown' or gender == 'Unknown':
                continue

            emp_intensity = emp_data[i, :]
            emp_footprint = emp_intensity @ self.L @ self.Y
            emp_total = emp_footprint.sum()
            key = f"{skill}_{gender}"

            if key not in employment_by_skill_gender:
                employment_by_skill_gender[key] = 0.0
            employment_by_skill_gender[key] += emp_total

            print(f"  - {cat}: {emp_total:.2f} 千人")

        df = pd.DataFrame.from_dict(
            employment_by_skill_gender,
            orient='index',
            columns=['Employment_Footprint']
        )
        df = df.sort_values('Employment_Footprint', ascending=False)

        print("\n  按技能×性别汇总:")
        for idx, row in df.iterrows():
            print(f"    {idx}: {row['Employment_Footprint']:.2f} 千人")

        self.results['employment_by_skill_gender'] = df
        return df

    def compute_wage_footprint_by_skill(self) -> pd.DataFrame:
        """
        按技能计算工资足迹

        Returns:
            DataFrame: 技能的工资足迹
        """
        print("\n" + "=" * 70)
        print("步骤4: 计算工资足迹（按技能）")
        print("=" * 70)

        wage_data = self.wage['data']  # (9, 9800)
        categories = self.wage['categories']

        wage_by_skill = {}

        for i, cat in enumerate(categories):
            wage_intensity = wage_data[i, :]  # (9800,)

            # 工资足迹: w @ L @ Y
            wage_footprint = wage_intensity @ self.L @ self.Y  # (343,)
            wage_total = wage_footprint.sum()

            if 'Low' in cat:
                skill = 'Low'
            elif 'Medium' in cat:
                skill = 'Medium'
            elif 'High' in cat:
                skill = 'High'
            else:
                skill = 'Other'

            if skill not in wage_by_skill:
                wage_by_skill[skill] = 0
            wage_by_skill[skill] += wage_total

            print(f"  - {cat}: {wage_total:.2e} 百万欧元")

        df = pd.DataFrame.from_dict(wage_by_skill, orient='index', columns=['Wage_Footprint'])
        df = df.sort_values('Wage_Footprint', ascending=False)

        print(f"\n  按技能汇总:")
        for idx, row in df.iterrows():
            print(f"    {idx}技能: {row['Wage_Footprint']:.2e} 百万欧元")

        self.results['wage_by_skill'] = df
        return df


    def _build_hours_skill_gender_intensity_matrix(self) -> Tuple[np.ndarray, list]:
        """
        构造技能×性别工作时间系数矩阵。

        Returns:
            Tuple[np.ndarray, list]: (6×部门矩阵, [(技能, 性别), ...])
        """
        categories = self.hours['categories']
        hours_data = np.asarray(self.hours['data'], dtype=float)
        label_map = {cat: self._parse_skill_gender_label(cat) for cat in categories}
        self._validate_employment_skill_gender_coverage(label_map)

        ordered_pairs = [
            ('Low', 'Male'), ('Low', 'Female'),
            ('Medium', 'Male'), ('Medium', 'Female'),
            ('High', 'Male'), ('High', 'Female'),
        ]
        pair_to_idx = {pair: idx for idx, pair in enumerate(ordered_pairs)}
        matrix = np.zeros((len(ordered_pairs), hours_data.shape[1]), dtype=float)

        for row_idx, category in enumerate(categories):
            pair = label_map[category]
            if pair in pair_to_idx:
                matrix[pair_to_idx[pair], :] += hours_data[row_idx, :]

        return matrix, ordered_pairs

    def _build_hours_skill_intensity_matrix(self) -> Tuple[np.ndarray, list]:
        """将工作时间扩展聚合为 Low/Medium/High 三类技能矩阵。"""
        pair_matrix, ordered_pairs = self._build_hours_skill_gender_intensity_matrix()
        skills = ['Low', 'Medium', 'High']
        matrix = np.zeros((len(skills), pair_matrix.shape[1]), dtype=float)
        for skill_idx, skill in enumerate(skills):
            indices = [idx for idx, pair in enumerate(ordered_pairs) if pair[0] == skill]
            matrix[skill_idx, :] = pair_matrix[indices, :].sum(axis=0)
        return matrix, skills

    def compute_hours_footprint_by_skill_gender(self) -> pd.DataFrame:
        """
        按技能×性别计算工作时间足迹及其占比。

        Returns:
            DataFrame: 技能×性别工作时间足迹与占比
        """
        print("\n" + "=" * 70)
        print("步骤4b: 计算工作时间足迹（按技能×性别）")
        print("=" * 70)

        hours_pair_matrix, pair_labels = self._build_hours_skill_gender_intensity_matrix()
        hours_by_pair = hours_pair_matrix @ self.L @ self.Y

        rows = []
        for pair_idx, (skill, gender) in enumerate(pair_labels):
            label = f'{skill}_{gender}'
            hours_total = float(hours_by_pair[pair_idx, :].sum())
            rows.append({'Skill': skill, 'Gender': gender, 'Skill_Gender': label, 'Hours_Footprint': hours_total})
            print(f"  - {label}: {hours_total:.2e} M.hr")

        df = pd.DataFrame(rows)
        total_hours = float(df['Hours_Footprint'].sum())
        df['Hours_Share'] = df['Hours_Footprint'] / total_hours if total_hours > 0.0 else 0.0
        df = df.sort_values('Hours_Footprint', ascending=False).reset_index(drop=True)
        self.results['hours_by_skill_gender'] = df
        self.results['hours_share_by_skill_gender'] = df[['Skill', 'Gender', 'Skill_Gender', 'Hours_Share']].copy()
        return df

    def compute_wage_per_hour_by_skill(self) -> pd.DataFrame:
        """
        计算按技能汇总的工资/工时强度。

        Returns:
            DataFrame: 各技能的工资总量、工时总量与每小时工资
        """
        print("\n" + "=" * 70)
        print("步骤4c: 计算每小时工资（按技能）")
        print("=" * 70)

        wage_skill_matrix, skills = self._build_wage_skill_intensity_matrix()
        hours_skill_matrix, _ = self._build_hours_skill_intensity_matrix()
        wage_direct = wage_skill_matrix * self.x.reshape(1, -1)
        hours_direct = hours_skill_matrix * self.x.reshape(1, -1)

        rows = []
        for skill_idx, skill in enumerate(skills):
            wage_total = float(wage_direct[skill_idx, :].sum())
            hours_total = float(hours_direct[skill_idx, :].sum())
            wage_per_hour = wage_total / hours_total if hours_total > 0.0 else 0.0
            rows.append(
                {
                    'Skill': skill,
                    'Wage_Total': wage_total,
                    'Hours_Total': hours_total,
                    'Wage_Per_Hour': wage_per_hour,
                }
            )
            print(f"  - {skill}: {wage_per_hour:.4f} EUR / hour")

        df = pd.DataFrame(rows).sort_values('Wage_Per_Hour', ascending=False).reset_index(drop=True)
        self.results['wage_per_hour_by_skill'] = df
        return df

    def compute_consumption_side_country_skill_gender_hours(self) -> pd.DataFrame:
        """
        计算消费侧国家×技能×性别隐含工作时间账户。

        Returns:
            DataFrame: 国家×技能×性别隐含工时账户
        """
        print("\n" + "=" * 70)
        print("步骤4d: 计算消费侧国家×技能×性别工时账户")
        print("=" * 70)

        hours_pair_matrix, pair_labels = self._build_hours_skill_gender_intensity_matrix()
        hours_by_pair_fd = hours_pair_matrix @ self.L @ self.Y
        carbon_fd = np.asarray(self.co2_emissions, dtype=float) @ self.L @ self.Y
        fd_regions = self._get_final_demand_region_labels()
        carbon_by_country = pd.Series(carbon_fd).groupby(fd_regions).sum()

        df = self._assemble_country_skill_gender_account_table(
            region_labels=fd_regions,
            pair_labels=pair_labels,
            carbon_by_country=carbon_by_country,
            value_matrix=hours_by_pair_fd,
            carbon_total_col='Embodied_Carbon_Country_Total',
            carbon_share_col='Embodied_Carbon_Country_Share',
            value_col='Embodied_Hours',
        )

        print("  ✓ 消费侧国家×技能×性别工时账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['consumption_side_country_skill_gender_hours'] = df
        return df

    def compute_production_side_country_skill_gender_hours(self) -> pd.DataFrame:
        """
        计算生产侧国家×技能×性别直接工作时间账户。

        Returns:
            DataFrame: 国家×技能×性别直接工时账户
        """
        print("\n" + "=" * 70)
        print("步骤4e: 计算生产侧国家×技能×性别工时账户")
        print("=" * 70)

        hours_pair_matrix, pair_labels = self._build_hours_skill_gender_intensity_matrix()
        x = np.asarray(self.x, dtype=float)
        hours_by_pair_sector = hours_pair_matrix * x.reshape(1, -1)
        carbon_sector = np.asarray(self.co2_emissions, dtype=float) * x
        producer_regions = self._get_producer_region_labels()
        carbon_by_country = pd.Series(carbon_sector).groupby(producer_regions).sum()

        df = self._assemble_country_skill_gender_account_table(
            region_labels=producer_regions,
            pair_labels=pair_labels,
            carbon_by_country=carbon_by_country,
            value_matrix=hours_by_pair_sector,
            carbon_total_col='Direct_Carbon_Country_Total',
            carbon_share_col='Direct_Carbon_Country_Share',
            value_col='Direct_Hours',
        )

        print("  ✓ 生产侧国家×技能×性别工时账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['production_side_country_skill_gender_hours'] = df
        return df

    def compute_carbon_hours_mismatch(self) -> pd.DataFrame:
        """
        计算碳份额与工时份额之间的错配关系（按技能）。

        Returns:
            DataFrame: 技能-碳-工时错配表
        """
        print("\n" + "=" * 70)
        print("步骤4f: 计算碳份额 vs 工时份额错配（按技能）")
        print("=" * 70)

        hours_skill_matrix, skills = self._build_hours_skill_intensity_matrix()
        hours_direct = hours_skill_matrix * self.x.reshape(1, -1)
        total_hours = float(hours_direct.sum())

        carbon_by_skill = {}
        hours_by_skill = {}
        for skill_idx, skill in enumerate(skills):
            skill_hours = float(hours_direct[skill_idx, :].sum())
            skill_carbon = float((hours_skill_matrix[skill_idx, :] * self.co2_emissions * self.x).sum())
            hours_by_skill[skill] = skill_hours
            carbon_by_skill[skill] = skill_carbon

        total_carbon = float(sum(carbon_by_skill.values()))
        rows = []
        for skill in skills:
            carbon_share = carbon_by_skill[skill] / total_carbon if total_carbon > 0.0 else 0.0
            hours_share = hours_by_skill[skill] / total_hours if total_hours > 0.0 else 0.0
            mismatch = carbon_share / hours_share if hours_share > 0.0 else 0.0
            rows.append(
                {
                    'Skill': skill,
                    'Carbon_Share': carbon_share,
                    'Hours_Share': hours_share,
                    'Mismatch_Index': mismatch,
                    'Mismatch_Type': 'Over-exposed' if mismatch > 1.0 else ('Under-exposed' if mismatch < 1.0 else 'Balanced'),
                }
            )
            print(f"  - {skill}: carbon={carbon_share:.2%}, hours={hours_share:.2%}, mismatch={mismatch:.3f}")

        df = pd.DataFrame(rows)
        self.results['carbon_hours_mismatch'] = df
        return df

    def compute_high_carbon_hours_by_skill_gender(self) -> pd.DataFrame:
        """
        计算高碳部门中的工时占比与工时锁定度（按技能×性别）。

        Returns:
            DataFrame: 技能×性别高碳工时分布表
        """
        print("\n" + "=" * 70)
        print("步骤4g: 计算高碳部门工时占比（按技能×性别）")
        print("=" * 70)

        hours_pair_matrix, pair_labels = self._build_hours_skill_gender_intensity_matrix()
        carbon_sector = np.asarray(self.co2_emissions, dtype=float) * np.asarray(self.x, dtype=float)
        positive_carbon = carbon_sector[carbon_sector > 0.0]
        if positive_carbon.size > 0:
            threshold_high = float(np.percentile(positive_carbon, 80))
            high_carbon_mask = carbon_sector >= threshold_high
        else:
            high_carbon_mask = np.zeros_like(carbon_sector, dtype=bool)

        total_hours_vector = hours_pair_matrix * self.x.reshape(1, -1)
        high_carbon_total = float(total_hours_vector[:, high_carbon_mask].sum())
        rows = []
        for pair_idx, (skill, gender) in enumerate(pair_labels):
            total_hours = float(total_hours_vector[pair_idx, :].sum())
            high_carbon_hours = float(total_hours_vector[pair_idx, high_carbon_mask].sum())
            rows.append(
                {
                    'Skill': skill,
                    'Gender': gender,
                    'Total_Hours': total_hours,
                    'High_Carbon_Hours': high_carbon_hours,
                    'Hours_Share': total_hours / float(total_hours_vector.sum()) if float(total_hours_vector.sum()) > 0.0 else 0.0,
                    'High_Carbon_Hours_Share': high_carbon_hours / high_carbon_total if high_carbon_total > 0.0 else 0.0,
                    'Hours_Lock_Ratio': high_carbon_hours / total_hours if total_hours > 0.0 else 0.0,
                }
            )

        df = pd.DataFrame(rows).sort_values('Hours_Lock_Ratio', ascending=False).reset_index(drop=True)
        self.results['high_carbon_hours_share_by_skill_gender'] = df
        return df

    def compute_hours_lock_gap(self) -> pd.DataFrame:
        """
        计算低技能与高技能之间的工时锁定差距。

        Returns:
            DataFrame: 工时锁定差距摘要表
        """
        print("\n" + "=" * 70)
        print("步骤4h: 计算工时锁定差距")
        print("=" * 70)

        if 'high_carbon_hours_share_by_skill_gender' not in self.results:
            self.compute_high_carbon_hours_by_skill_gender()
        table = self.results['high_carbon_hours_share_by_skill_gender']
        skill_summary = table.groupby('Skill').agg({'Total_Hours': 'sum', 'High_Carbon_Hours': 'sum'}).reset_index()
        skill_summary['Hours_Lock_Ratio'] = np.divide(
            skill_summary['High_Carbon_Hours'],
            skill_summary['Total_Hours'],
            out=np.zeros(skill_summary.shape[0], dtype=float),
            where=skill_summary['Total_Hours'] > 0.0,
        )
        low_lock = float(skill_summary.loc[skill_summary['Skill'] == 'Low', 'Hours_Lock_Ratio'].iloc[0])
        high_lock = float(skill_summary.loc[skill_summary['Skill'] == 'High', 'Hours_Lock_Ratio'].iloc[0])
        lock_gap = low_lock / high_lock if high_lock > 0.0 else 0.0
        df = pd.DataFrame(
            [
                {
                    'Low_Hours_Lock_Ratio': low_lock,
                    'High_Hours_Lock_Ratio': high_lock,
                    'Hours_Lock_Gap': lock_gap,
                }
            ]
        )
        print(f"  - Low / High hours lock gap: {lock_gap:.3f}x")
        self.results['hours_lock_gap'] = df
        return df


    def _build_employment_skill_gender_intensity_matrix(self) -> Tuple[np.ndarray, list]:
        """
        构造技能×性别就业系数矩阵。

        Returns:
            Tuple[np.ndarray, list]: (6×部门矩阵, [(技能, 性别), ...])
        """
        categories = self.employment['categories']
        emp_data = np.asarray(self.employment['data'], dtype=float)
        label_map = {cat: self._parse_skill_gender_label(cat) for cat in categories}
        self._validate_employment_skill_gender_coverage(label_map)

        ordered_pairs = [
            ('Low', 'Male'), ('Low', 'Female'),
            ('Medium', 'Male'), ('Medium', 'Female'),
            ('High', 'Male'), ('High', 'Female'),
        ]
        pair_to_idx = {pair: idx for idx, pair in enumerate(ordered_pairs)}
        matrix = np.zeros((len(ordered_pairs), emp_data.shape[1]), dtype=float)

        for row_idx, category in enumerate(categories):
            pair = label_map[category]
            if pair in pair_to_idx:
                matrix[pair_to_idx[pair], :] += emp_data[row_idx, :]

        return matrix, ordered_pairs

    def _build_estimated_wage_skill_gender_intensity_matrix(self) -> Tuple[np.ndarray, list]:
        """
        基于同技能组内的男女就业份额，估算工资的技能×性别系数矩阵。

        说明：EXIOBASE 工资扩展没有直接提供 gender 维度，因此这里采用
        “同技能组内按男女就业系数占比分摊工资系数”的代理方法。

        Returns:
            Tuple[np.ndarray, list]: (6×部门矩阵, [(技能, 性别), ...])
        """
        wage_skill_matrix, skills = self._build_wage_skill_intensity_matrix()
        emp_pair_matrix, ordered_pairs = self._build_employment_skill_gender_intensity_matrix()
        estimated = np.zeros_like(emp_pair_matrix)

        for skill_idx, skill in enumerate(skills):
            pair_indices = [idx for idx, pair in enumerate(ordered_pairs) if pair[0] == skill]
            skill_emp_total = emp_pair_matrix[pair_indices, :].sum(axis=0)
            global_pair_totals = emp_pair_matrix[pair_indices, :].sum(axis=1)
            global_total = float(global_pair_totals.sum())
            if global_total > 0.0:
                fallback_share = global_pair_totals / global_total
            else:
                fallback_share = np.full(len(pair_indices), 1.0 / len(pair_indices))

            for local_idx, pair_idx in enumerate(pair_indices):
                share_vector = np.divide(
                    emp_pair_matrix[pair_idx, :],
                    skill_emp_total,
                    out=np.full(skill_emp_total.shape, float(fallback_share[local_idx]), dtype=float),
                    where=skill_emp_total > 0.0,
                )
                estimated[pair_idx, :] = wage_skill_matrix[skill_idx, :] * share_vector

        return estimated, ordered_pairs

    def _assemble_country_skill_gender_account_table(
        self,
        region_labels: pd.Index,
        pair_labels: list,
        carbon_by_country: pd.Series,
        value_matrix: np.ndarray,
        carbon_total_col: str,
        carbon_share_col: str,
        value_col: str,
    ) -> pd.DataFrame:
        """
        组织国家×技能×性别账户表。

        Args:
            region_labels: 区域标签
            pair_labels: [(技能, 性别), ...]
            carbon_by_country: 国家层碳总量
            value_matrix: 技能×性别对应的账户矩阵
            carbon_total_col: 国家层碳总量列名
            carbon_share_col: 国家层碳份额列名
            value_col: 主账户列名

        Returns:
            pd.DataFrame: 国家×技能×性别账户表
        """
        rows = []
        region_labels = pd.Index(region_labels)
        value_total = float(np.asarray(value_matrix, dtype=float).sum())
        carbon_total = float(carbon_by_country.sum())

        for region in region_labels.unique():
            region_mask = (region_labels == region)
            value_country = np.asarray(value_matrix[:, region_mask], dtype=float).sum(axis=1)
            value_country_total = float(value_country.sum())
            carbon_value = float(carbon_by_country.loc[region]) if region in carbon_by_country.index else 0.0
            carbon_share = carbon_value / carbon_total if carbon_total > 0.0 else 0.0

            for pair_idx, (skill, gender) in enumerate(pair_labels):
                value_item = float(value_country[pair_idx])
                rows.append(
                    {
                        'Region': region,
                        'Skill': skill,
                        'Gender': gender,
                        carbon_total_col: carbon_value,
                        carbon_share_col: carbon_share,
                        value_col: value_item,
                        f'{value_col}_Share_Within_Country': (
                            value_item / value_country_total if value_country_total > 0.0 else 0.0
                        ),
                        f'{value_col}_Share_Global': value_item / value_total if value_total > 0.0 else 0.0,
                    }
                )

        df = pd.DataFrame(rows)
        df['Skill'] = pd.Categorical(df['Skill'], categories=['Low', 'Medium', 'High'], ordered=True)
        df['Gender'] = pd.Categorical(df['Gender'], categories=['Male', 'Female'], ordered=True)
        df = df.sort_values([carbon_total_col, 'Region', 'Skill', 'Gender'], ascending=[False, True, True, True]).reset_index(drop=True)
        return df

    def compute_wage_footprint_by_skill_gender_estimated(self) -> pd.DataFrame:
        """
        估算按技能×性别划分的工资足迹。

        Returns:
            DataFrame: 技能×性别工资足迹（估算）
        """
        print("\n" + "=" * 70)
        print("步骤4b: 计算工资足迹（按技能×性别，估算）")
        print("=" * 70)
        print("  方法：按同技能组内男女就业份额分摊工资系数")

        wage_pair_matrix, pair_labels = self._build_estimated_wage_skill_gender_intensity_matrix()
        wage_by_pair = wage_pair_matrix @ self.L @ self.Y

        rows = []
        for pair_idx, (skill, gender) in enumerate(pair_labels):
            label = f'{skill}_{gender}'
            wage_total = float(wage_by_pair[pair_idx, :].sum())
            rows.append({'Skill_Gender': label, 'Estimated_Wage_Footprint': wage_total})
            print(f"  - {label}: {wage_total:.2e} 百万欧元")

        df = pd.DataFrame(rows).sort_values('Estimated_Wage_Footprint', ascending=False).reset_index(drop=True)
        self.results['wage_by_skill_gender_estimated'] = df
        return df

    def compute_consumption_side_country_skill_gender_employment(self) -> pd.DataFrame:
        """
        计算消费侧国家×技能×性别隐含就业账户。

        Returns:
            DataFrame: 国家×技能×性别隐含就业账户
        """
        print("\n" + "=" * 70)
        print("步骤4c: 计算消费侧国家×技能×性别就业账户")
        print("=" * 70)

        emp_pair_matrix, pair_labels = self._build_employment_skill_gender_intensity_matrix()
        employment_by_pair_fd = emp_pair_matrix @ self.L @ self.Y
        carbon_fd = np.asarray(self.co2_emissions, dtype=float) @ self.L @ self.Y
        fd_regions = self._get_final_demand_region_labels()
        carbon_by_country = pd.Series(carbon_fd).groupby(fd_regions).sum()

        df = self._assemble_country_skill_gender_account_table(
            region_labels=fd_regions,
            pair_labels=pair_labels,
            carbon_by_country=carbon_by_country,
            value_matrix=employment_by_pair_fd,
            carbon_total_col='Embodied_Carbon_Country_Total',
            carbon_share_col='Embodied_Carbon_Country_Share',
            value_col='Embodied_Employment',
        )

        print("  ✓ 消费侧国家×技能×性别就业账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['consumption_side_country_skill_gender_employment'] = df
        return df

    def compute_production_side_country_skill_gender_employment(self) -> pd.DataFrame:
        """
        计算生产侧国家×技能×性别直接就业账户。

        Returns:
            DataFrame: 国家×技能×性别直接就业账户
        """
        print("\n" + "=" * 70)
        print("步骤4d: 计算生产侧国家×技能×性别就业账户")
        print("=" * 70)

        emp_pair_matrix, pair_labels = self._build_employment_skill_gender_intensity_matrix()
        x = np.asarray(self.x, dtype=float)
        employment_by_pair_sector = emp_pair_matrix * x.reshape(1, -1)
        carbon_sector = np.asarray(self.co2_emissions, dtype=float) * x
        producer_regions = self._get_producer_region_labels()
        carbon_by_country = pd.Series(carbon_sector).groupby(producer_regions).sum()

        df = self._assemble_country_skill_gender_account_table(
            region_labels=producer_regions,
            pair_labels=pair_labels,
            carbon_by_country=carbon_by_country,
            value_matrix=employment_by_pair_sector,
            carbon_total_col='Direct_Carbon_Country_Total',
            carbon_share_col='Direct_Carbon_Country_Share',
            value_col='Direct_Employment',
        )

        print("  ✓ 生产侧国家×技能×性别就业账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['production_side_country_skill_gender_employment'] = df
        return df

    def compute_consumption_side_country_skill_gender_wage_estimated(self) -> pd.DataFrame:
        """
        估算消费侧国家×技能×性别隐含工资账户。

        Returns:
            DataFrame: 国家×技能×性别隐含工资账户（估算）
        """
        print("\n" + "=" * 70)
        print("步骤4e: 计算消费侧国家×技能×性别工资账户（估算）")
        print("=" * 70)

        wage_pair_matrix, pair_labels = self._build_estimated_wage_skill_gender_intensity_matrix()
        wage_by_pair_fd = wage_pair_matrix @ self.L @ self.Y
        carbon_fd = np.asarray(self.co2_emissions, dtype=float) @ self.L @ self.Y
        fd_regions = self._get_final_demand_region_labels()
        carbon_by_country = pd.Series(carbon_fd).groupby(fd_regions).sum()

        df = self._assemble_country_skill_gender_account_table(
            region_labels=fd_regions,
            pair_labels=pair_labels,
            carbon_by_country=carbon_by_country,
            value_matrix=wage_by_pair_fd,
            carbon_total_col='Embodied_Carbon_Country_Total',
            carbon_share_col='Embodied_Carbon_Country_Share',
            value_col='Estimated_Embodied_Wage',
        )

        print("  ✓ 消费侧国家×技能×性别工资账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['consumption_side_country_skill_gender_wage_estimated'] = df
        return df

    def compute_production_side_country_skill_gender_wage_estimated(self) -> pd.DataFrame:
        """
        估算生产侧国家×技能×性别直接工资账户。

        Returns:
            DataFrame: 国家×技能×性别直接工资账户（估算）
        """
        print("\n" + "=" * 70)
        print("步骤4f: 计算生产侧国家×技能×性别工资账户（估算）")
        print("=" * 70)

        wage_pair_matrix, pair_labels = self._build_estimated_wage_skill_gender_intensity_matrix()
        x = np.asarray(self.x, dtype=float)
        wage_by_pair_sector = wage_pair_matrix * x.reshape(1, -1)
        carbon_sector = np.asarray(self.co2_emissions, dtype=float) * x
        producer_regions = self._get_producer_region_labels()
        carbon_by_country = pd.Series(carbon_sector).groupby(producer_regions).sum()

        df = self._assemble_country_skill_gender_account_table(
            region_labels=producer_regions,
            pair_labels=pair_labels,
            carbon_by_country=carbon_by_country,
            value_matrix=wage_by_pair_sector,
            carbon_total_col='Direct_Carbon_Country_Total',
            carbon_share_col='Direct_Carbon_Country_Share',
            value_col='Estimated_Direct_Wage',
        )

        print("  ✓ 生产侧国家×技能×性别工资账户计算完成")
        print(df.head(12).to_string(index=False))

        self.results['production_side_country_skill_gender_wage_estimated'] = df
        return df

    def compute_skill_carbon_mismatch_index_by_country(self) -> pd.DataFrame:
        """
        计算每个国家的技能-碳错配指数

        方法：按部门直接碳排放量排序，将部门分为高碳/中碳/低碳三类，
        然后分析各类别中的技能结构

        公式: Mismatch_{r,s} = (高碳部门就业占比) / (总就业占比)
        - Mismatch > 1: 该技能群体更多集中在高碳部门

        Returns:
            DataFrame: 国家×技能的错配指数
        """
        print("\n" + "=" * 70)
        print("步骤5: 计算技能-碳错配指数（按国家）")
        print("=" * 70)
        print("  方法：按部门直接碳排放量分类，计算高碳部门的技能集中度")

        emp_data = self.employment['data']  # (6, 9800) 就业系数
        categories = self.employment['categories']
        co2_coef = self.co2_emissions  # (9800,) kg CO2 / M.EUR
        x = self.x  # (9800,) 总产出 (M.EUR)

        skill_map = {}
        for i, cat in enumerate(categories):
            if 'Low-skilled' in cat:
                skill = 'Low'
            elif 'Medium-skilled' in cat:
                skill = 'Medium'
            elif 'High-skilled' in cat:
                skill = 'High'
            else:
                continue
            skill_map[i] = skill

        # 部门直接碳排放量：排放系数乘以总产出。
        carbon_intensity = co2_coef * x  # (9800,) kg CO2

        # 按正直接碳排放量划分高碳(前20%)、中碳(中间60%)、低碳(后20%)。
        threshold_high = np.percentile(carbon_intensity[carbon_intensity > 0], 80)
        threshold_low = np.percentile(carbon_intensity[carbon_intensity > 0], 20)

        print(f"\n  部门直接碳排放量阈值:")
        print(f"    - 高碳部门: > {threshold_high:.2f} kg CO2")
        print(f"    - 低碳部门: < {threshold_low:.2f} kg CO2")

        results = []

        for r, region in enumerate(self.regions):
            start_idx = r * self.n_products
            end_idx = start_idx + self.n_products

            x_region = x[start_idx:end_idx]  # (200,)
            co2_region = carbon_intensity[start_idx:end_idx]  # (200,)

            high_carbon_mask = co2_region > threshold_high
            low_carbon_mask = co2_region < threshold_low

            emp_by_skill = {'Low': 0, 'Medium': 0, 'High': 0}
            emp_high_carbon_by_skill = {'Low': 0, 'Medium': 0, 'High': 0}

            for i, cat in enumerate(categories):
                if i not in skill_map:
                    continue
                skill = skill_map[i]

                # 就业系数 × 产出 = 就业人数
                emp_coef = emp_data[i, start_idx:end_idx]  # (200,)
                emp_region = emp_coef * x_region  # (200,)

                emp_total = emp_region.sum()
                emp_by_skill[skill] += emp_total

                emp_high_carbon = (emp_region * high_carbon_mask).sum()
                emp_high_carbon_by_skill[skill] += emp_high_carbon

            total_emp = sum(emp_by_skill.values())
            total_high_carbon = sum(emp_high_carbon_by_skill.values())

            if total_emp == 0 or total_high_carbon == 0:
                continue

            # 计算错配指数：高碳部门就业占比 / 总就业占比
            for skill in ['Low', 'Medium', 'High']:
                emp_share = emp_by_skill[skill] / total_emp
                high_carbon_share = emp_high_carbon_by_skill[skill] / total_high_carbon

                if emp_share > 0:
                    mismatch = high_carbon_share / emp_share
                else:
                    mismatch = 0

                results.append({
                    'Region': region,
                    'Skill': skill,
                    'Total_Employment': emp_by_skill[skill],
                    'High_Carbon_Employment': emp_high_carbon_by_skill[skill],
                    'Employment_Share': emp_share,
                    'High_Carbon_Share': high_carbon_share,
                    'Mismatch_Index': mismatch
                })

        df = pd.DataFrame(results)

        print("\n  各国错配指数（低技能，Top 10）:")
        low_skill_df = df[df['Skill'] == 'Low'].sort_values('Mismatch_Index', ascending=False)
        for idx, row in low_skill_df.head(10).iterrows():
            print(f"    {row['Region']}: {row['Mismatch_Index']:.3f} (高碳集中度:{row['High_Carbon_Share']:.1%})")

        print("\n  各国错配指数（高技能，Top 10）:")
        high_skill_df = df[df['Skill'] == 'High'].sort_values('Mismatch_Index', ascending=False)
        for idx, row in high_skill_df.head(10).iterrows():
            print(f"    {row['Region']}: {row['Mismatch_Index']:.3f} (高碳集中度:{row['High_Carbon_Share']:.1%})")

        print("\n  全球汇总（按技能）:")
        global_summary = df.groupby('Skill').agg({
            'Total_Employment': 'sum',
            'High_Carbon_Employment': 'sum',
            'Employment_Share': 'mean',
            'High_Carbon_Share': 'mean',
            'Mismatch_Index': 'mean'
        }).round(3)
        print(global_summary)

        self.results['mismatch_by_country_skill'] = df
        return df

    def compute_skill_carbon_mismatch_index(self) -> pd.DataFrame:
        """
        计算技能-碳错配指数（全球汇总）

        公式: Mismatch_s = (碳排放占比) / (就业占比)
        - Mismatch > 1: 该技能群体承担了超出其就业比例的碳排放
        - Mismatch < 1: 该技能群体承担了低于其就业比例的碳排放

        Returns:
            DataFrame: 技能-碳错配指数
        """
        print("\n" + "=" * 70)
        print("步骤5b: 计算技能-碳错配指数（全球汇总）")
        print("=" * 70)

        # 计算各技能的碳排放（生产端）
        emp_data = self.employment['data']  # (6, 9800)
        categories = self.employment['categories']
        co2_coef = self.co2_emissions  # (9800,)

        employment_by_skill = {'Low': 0, 'Medium': 0, 'High': 0}

        for i, cat in enumerate(categories):
            emp_intensity = emp_data[i, :]  # (9800,)

            # 直接就业（生产端）
            emp_direct = emp_intensity * self.x  # (9800,)
            emp_total = emp_direct.sum()

            if 'Low-skilled' in cat:
                skill = 'Low'
            elif 'Medium-skilled' in cat:
                skill = 'Medium'
            elif 'High-skilled' in cat:
                skill = 'High'
            else:
                continue

            employment_by_skill[skill] += emp_total

        total_employment = sum(employment_by_skill.values())

        # 按部门内技能就业份额分配 CO2。
        co2_by_skill = {'Low': 0, 'Medium': 0, 'High': 0}
        emp_by_skill_sector = {'Low': np.zeros_like(self.x, dtype=float), 'Medium': np.zeros_like(self.x, dtype=float), 'High': np.zeros_like(self.x, dtype=float)}

        for i, cat in enumerate(categories):
            if 'Low-skilled' in cat:
                skill = 'Low'
            elif 'Medium-skilled' in cat:
                skill = 'Medium'
            elif 'High-skilled' in cat:
                skill = 'High'
            else:
                continue

            emp_coef = emp_data[i, :]  # (9800,)
            emp_by_skill_sector[skill] += emp_coef * self.x

        sector_employment_total = sum(emp_by_skill_sector.values())
        sector_carbon_total = co2_coef * self.x
        for skill in ['Low', 'Medium', 'High']:
            alpha = np.divide(
                emp_by_skill_sector[skill],
                sector_employment_total,
                out=np.zeros_like(sector_employment_total, dtype=float),
                where=sector_employment_total > 0,
            )
            co2_by_skill[skill] = float((sector_carbon_total * alpha).sum())

        total_co2 = sum(co2_by_skill.values())

        mismatch_index = {}

        for skill in ['Low', 'Medium', 'High']:
            carbon_share = co2_by_skill[skill] / total_co2 if total_co2 > 0 else 0

            employment_share = employment_by_skill[skill] / total_employment if total_employment > 0 else 0

            if employment_share > 0:
                mismatch = carbon_share / employment_share
            else:
                mismatch = 0

            mismatch_index[skill] = {
                'Carbon_Share': carbon_share,
                'Employment_Share': employment_share,
                'Mismatch_Index': mismatch
            }

            print(f"  {skill}技能:")
            print(f"    - 碳排放占比: {carbon_share:.2%}")
            print(f"    - 就业占比: {employment_share:.2%}")
            print(f"    - 错配指数: {mismatch:.3f}")

        df = pd.DataFrame.from_dict(mismatch_index, orient='index')

        df['Mismatch_Type'] = df['Mismatch_Index'].apply(
            lambda x: 'Over-exposed' if x > 1 else ('Under-exposed' if x < 1 else 'Balanced')
        )

        print(f"\n  错配分析:")
        for idx, row in df.iterrows():
            print(f"    {idx}技能: {row['Mismatch_Type']} (指数={row['Mismatch_Index']:.3f})")

        self.results['mismatch_index'] = df
        return df

    def compute_all(self) -> Dict:
        """
        执行完整计算流程

        Returns:
            Dict: 所有计算结果
        """
        print("\n" + "=" * 70)
        print("开始完整足迹计算")
        print("=" * 70)

        self.compute_carbon_footprint_by_country()

        self.compute_consumption_side_country_triplets()

        self.compute_production_side_country_triplets()

        self.compute_consumption_side_country_skill_triplets()

        self.compute_production_side_country_skill_triplets()

        self.compute_employment_footprint_by_skill_gender()

        self.compute_hours_footprint_by_skill_gender()

        self.compute_wage_footprint_by_skill()

        self.compute_wage_per_hour_by_skill()

        self.compute_consumption_side_country_skill_gender_employment()

        self.compute_production_side_country_skill_gender_employment()

        self.compute_consumption_side_country_skill_gender_hours()

        self.compute_production_side_country_skill_gender_hours()

        self.compute_carbon_hours_mismatch()

        self.compute_high_carbon_hours_by_skill_gender()

        self.compute_hours_lock_gap()

        self.compute_skill_carbon_mismatch_index_by_country()

        self.compute_skill_carbon_mismatch_index()

        # 工资×技能×性别估算不纳入默认计算。

        print("\n" + "=" * 70)
        print("计算完成！")
        print("=" * 70)

        return self.results

    def save_results(self, output_path: str):
        """
        保存结果到文件。

        Args:
            output_path: 输出目录路径
        """
        import os

        csv_dir = os.path.join(output_path, "csv")
        json_dir = os.path.join(output_path, "json")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        for name, df in self.results.items():
            file_path = os.path.join(csv_dir, f"{name}.csv")
            df.to_csv(file_path)
            print(f"  ✓ 保存: {file_path}")
