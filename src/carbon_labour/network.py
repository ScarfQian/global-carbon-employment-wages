"""Supply-chain networks and embodied carbon, employment, wage and hours transfers."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import pymrio

# =========================================================================
# 网络构建模块
# =========================================================================

class NetworkEngine:
    """供应链网络分析引擎"""

    def __init__(self, exiobase_path: str):
        """
        初始化网络引擎

        Args:
            exiobase_path: EXIOBASE数据路径
        """
        self.exiobase_path = Path(exiobase_path)
        self.mrio = None
        self.Y = None
        self.Z = None  # 中间投入矩阵
        self.L = None  # Leontief逆矩阵
        self.regions = None
        self.sectors = None
        self.network = None
        self.country_skill_network = None
        self.country_skill_results = {}
        self.co2_emissions = None
        self.employment = None
        self.employment_skill_gender = None
        self.hours = None
        self.hours_skill_gender = None
        self.wage = None
        self.wage_skill_gender_estimated = None

        print("=" * 70)
        print("供应链网络分析引擎")
        print("=" * 70)

    def load_data(self) -> None:
        """加载EXIOBASE数据并构建网络"""
        print("\n" + "=" * 70)
        print("步骤1: 加载EXIOBASE数据")
        print("=" * 70)

        try:
            self.mrio = pymrio.parse_exiobase3(str(self.exiobase_path))
            print("  ✓ 数据加载成功")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            return

        self.mrio.calc_all()
        self.L = self.mrio.L
        self.Y = self.mrio.Y
        self.Z = self.mrio.Z

        self.regions = list(self.mrio.get_regions())
        self.sectors = list(self.mrio.get_sectors())

        print(f"  ✓ Leontief逆矩阵计算完成")
        print(f"  - 区域数: {len(self.regions)}")
        print(f"  - 部门数: {len(self.sectors)}")
        print(f"  - L矩阵维度: {self.L.shape}")

        self._extract_extensions()

    def _extract_extensions(self) -> None:
        """提取环境和社会扩展数据。"""
        print("\n  提取扩展数据:")

        if hasattr(self.mrio, "air_emissions") and hasattr(self.mrio.air_emissions, "S"):
            air = self.mrio.air_emissions
            co2_idx = [
                idx for idx, name in enumerate(air.S.index)
                if "CO2 - combustion" in str(name) and "bio" not in str(name).lower()
            ]
            if co2_idx:
                s_co2 = air.S.iloc[co2_idx].values.sum(axis=0)
                self.co2_emissions = s_co2
                self.co2_total = float((s_co2 * self.mrio.x.values.flatten()).sum())
                print(f"    ✓ CO2排放: {len(co2_idx)}类 → {self.co2_emissions.shape}")
                print(f"      - 总排放: {self.co2_total:.2e} kg = {self.co2_total / 1e9:.2f} 百万吨")

        if hasattr(self.mrio, "employment") and hasattr(self.mrio.employment, "S"):
            employment_df = self.mrio.employment.S
            labels = [str(idx) for idx in employment_df.index]
            people_idx = [idx for idx, label in enumerate(labels) if 'hours' not in label.lower()]
            hours_idx = [idx for idx, label in enumerate(labels) if 'hours' in label.lower()]
            people_df = employment_df.iloc[people_idx]
            hours_df = employment_df.iloc[hours_idx]

            self.employment = self._aggregate_skill_extension(people_df)
            self.employment_skill_gender = self._aggregate_skill_gender_extension(people_df)
            self.hours = self._aggregate_skill_extension(hours_df)
            self.hours_skill_gender = self._aggregate_skill_gender_extension(hours_df)

            if self.employment:
                print(f"    ✓ 就业数据: {len(self.employment)} 类")
            if self.employment_skill_gender:
                print(f"    ✓ 就业技能×性别数据: {len(self.employment_skill_gender)} 类")
            if self.hours:
                print(f"    ✓ 工时数据: {len(self.hours)} 类")
            if self.hours_skill_gender:
                print(f"    ✓ 工时技能×性别数据: {len(self.hours_skill_gender)} 类")

        if hasattr(self.mrio, "factor_inputs") and hasattr(self.mrio.factor_inputs, "S"):
            self.wage = self._aggregate_skill_extension(self.mrio.factor_inputs.S)
            if self.wage:
                print(f"    ✓ 工资数据: {len(self.wage)} 类")

        if self.co2_emissions is None:
            print("    尝试加载扩展数据...")
            try:
                self.mrio.get_required_extensions()
                if hasattr(self.mrio, "air_emissions") and hasattr(self.mrio.air_emissions, "S"):
                    air = self.mrio.air_emissions
                    co2_idx = [
                        idx for idx, name in enumerate(air.S.index)
                        if "CO2" in str(name) and "combustion" in str(name).lower()
                    ]
                    if co2_idx:
                        s_co2 = air.S.iloc[co2_idx].values.sum(axis=0)
                        self.co2_emissions = s_co2
                        self.co2_total = float((s_co2 * self.mrio.x.values.flatten()).sum())
                        print(f"    ✓ CO2排放(延迟加载): {self.co2_emissions.shape}")
            except Exception as e:
                print(f"    ✗ 扩展数据加载失败: {e}")

    def _extract_skill_level(self, name: str) -> str:
        """从名称中提取技能等级。"""
        if "Low-skilled" in name:
            return "Low"
        if "Medium-skilled" in name:
            return "Medium"
        if "High-skilled" in name:
            return "High"
        return "Total"


    def _extract_skill_gender_level(self, name: str) -> Tuple[str, str]:
        """从名称中提取技能与性别等级。"""
        skill = self._extract_skill_level(name)
        name_lower = str(name).lower()
        if 'female' in name_lower:
            gender = 'Female'
        elif 'male' in name_lower:
            gender = 'Male'
        else:
            gender = 'Unknown'
        return skill, gender

    def _aggregate_skill_gender_extension(
        self,
        extension_df: pd.DataFrame,
        exclude_keywords: Tuple[str, ...] = (),
    ) -> Dict[str, np.ndarray]:
        """把扩展账户按 Low/Medium/High × Male/Female 六类聚合。"""
        results: Dict[str, np.ndarray] = {}
        labels = [str(idx) for idx in extension_df.index]
        ordered_pairs = [
            ('Low', 'Male'), ('Low', 'Female'),
            ('Medium', 'Male'), ('Medium', 'Female'),
            ('High', 'Male'), ('High', 'Female'),
        ]
        for skill, gender in ordered_pairs:
            row_indices = []
            for idx, label in enumerate(labels):
                label_lower = label.lower()
                if any(keyword in label_lower for keyword in exclude_keywords):
                    continue
                label_skill, label_gender = self._extract_skill_gender_level(label)
                if label_skill == skill and label_gender == gender:
                    row_indices.append(idx)
            key = f'{skill}_{gender}'
            if row_indices:
                results[key] = extension_df.iloc[row_indices].values.sum(axis=0)
        return results

    def _estimate_wage_skill_gender_extension(
        self,
        wage_skill: Dict[str, np.ndarray],
        employment_skill_gender: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """按同技能组内男女就业份额估算工资的技能×性别扩展。"""
        results: Dict[str, np.ndarray] = {}
        for skill in ('Low', 'Medium', 'High'):
            wage_vector = np.asarray(wage_skill.get(skill, np.array([])), dtype=float).reshape(-1)
            if wage_vector.size == 0:
                continue
            male_key = f'{skill}_Male'
            female_key = f'{skill}_Female'
            male_emp = np.asarray(employment_skill_gender.get(male_key, np.zeros_like(wage_vector)), dtype=float).reshape(-1)
            female_emp = np.asarray(employment_skill_gender.get(female_key, np.zeros_like(wage_vector)), dtype=float).reshape(-1)
            emp_total = male_emp + female_emp
            global_total = float(emp_total.sum())
            if global_total > 0.0:
                male_fallback = float(male_emp.sum() / global_total)
                female_fallback = float(female_emp.sum() / global_total)
            else:
                male_fallback = 0.5
                female_fallback = 0.5

            male_share = np.divide(
                male_emp,
                emp_total,
                out=np.full(emp_total.shape, male_fallback, dtype=float),
                where=emp_total > 0.0,
            )
            female_share = np.divide(
                female_emp,
                emp_total,
                out=np.full(emp_total.shape, female_fallback, dtype=float),
                where=emp_total > 0.0,
            )
            results[male_key] = wage_vector * male_share
            results[female_key] = wage_vector * female_share
        return results

    def _build_metric_summary_by_skill_gender(
        self,
        transfer_map: Dict[str, np.ndarray],
        metric_name: str,
    ) -> pd.DataFrame:
        """把技能×性别转移矩阵整理为净转移摘要表。"""
        rows = []
        for key, matrix in transfer_map.items():
            if '_' in key:
                skill, gender = key.split('_', 1)
            else:
                skill, gender = key, 'Unknown'
            summary = self._build_transfer_summary(matrix, metric_name)
            summary.insert(1, 'Skill', skill)
            summary.insert(2, 'Gender', gender)
            rows.append(summary)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def _build_bilateral_transfer_table_by_skill_gender(
        self,
        transfer_map: Dict[str, np.ndarray],
        value_col: str,
    ) -> pd.DataFrame:
        """把技能×性别的国家-国家转移矩阵展开为双边明细表。"""
        rows = []
        for key, matrix in transfer_map.items():
            if '_' in key:
                skill, gender = key.split('_', 1)
            else:
                skill, gender = key, 'Unknown'
            for i, from_region in enumerate(self.regions):
                for j, to_region in enumerate(self.regions):
                    if i == j:
                        continue
                    value = float(matrix[i, j])
                    if value <= 0.0:
                        continue
                    rows.append(
                        {
                            'From_Region': from_region,
                            'To_Region': to_region,
                            'Skill': skill,
                            'Gender': gender,
                            value_col: value,
                        }
                    )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(value_col, ascending=False).reset_index(drop=True)

    def _build_bilateral_transfer_table_by_skill(
        self,
        transfer_map: Dict[str, np.ndarray],
        value_col: str,
    ) -> pd.DataFrame:
        """把按技能分组的国家-国家转移矩阵展开为双边明细表。"""
        rows = []
        for skill, matrix in transfer_map.items():
            for i, from_region in enumerate(self.regions):
                for j, to_region in enumerate(self.regions):
                    if i == j:
                        continue
                    value = float(matrix[i, j])
                    if value <= 0.0:
                        continue
                    rows.append(
                        {
                            "From_Region": from_region,
                            "To_Region": to_region,
                            "Skill": skill,
                            value_col: value,
                        }
                    )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(value_col, ascending=False).reset_index(drop=True)

    def _aggregate_skill_extension(
        self,
        extension_df: pd.DataFrame,
        exclude_keywords: Tuple[str, ...] = (),
    ) -> Dict[str, np.ndarray]:
        """把扩展账户按 Low/Medium/High 三类聚合。"""
        results: Dict[str, np.ndarray] = {}
        labels = [str(idx) for idx in extension_df.index]
        for skill in ("Low", "Medium", "High"):
            row_indices = []
            for idx, label in enumerate(labels):
                label_lower = label.lower()
                if any(keyword in label_lower for keyword in exclude_keywords):
                    continue
                if self._extract_skill_level(label) == skill:
                    row_indices.append(idx)
            if row_indices:
                results[skill] = extension_df.iloc[row_indices].values.sum(axis=0)
        return results

    def _get_final_demand_region_labels(self) -> List[str]:
        """提取最终需求列对应的地区标签。"""
        if self.mrio is not None and hasattr(self.mrio, "Y"):
            columns = self.mrio.Y.columns
            if isinstance(columns, pd.MultiIndex):
                return [str(v) for v in columns.get_level_values(0).tolist()]
            return [str(v) for v in columns.tolist()]

        if self.Y is None:
            return []
        y_matrix = np.asarray(self.Y, dtype=float)
        if not self.regions:
            return [f"FD_{idx}" for idx in range(y_matrix.shape[1])]
        if len(self.regions) == 1:
            return [self.regions[0]] * y_matrix.shape[1]
        if y_matrix.shape[1] == len(self.regions):
            return [str(region) for region in self.regions]
        if y_matrix.shape[1] % len(self.regions) == 0:
            repeat_count = y_matrix.shape[1] // len(self.regions)
            return [str(region) for region in self.regions for _ in range(repeat_count)]
        return [str(self.regions[idx % len(self.regions)]) for idx in range(y_matrix.shape[1])]

    @staticmethod
    def _normalize_series(values: np.ndarray) -> np.ndarray:
        """把向量缩放到 0-1 区间。"""
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            return array
        max_value = float(np.max(array))
        if max_value <= 0.0:
            return np.zeros_like(array)
        return array / max_value

    @staticmethod
    def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
        """安全计算相关系数。"""
        left_array = np.asarray(left, dtype=float).reshape(-1)
        right_array = np.asarray(right, dtype=float).reshape(-1)
        if left_array.size == 0 or right_array.size == 0 or left_array.size != right_array.size:
            return 0.0
        if np.allclose(left_array, left_array[0]) or np.allclose(right_array, right_array[0]):
            return 0.0
        value = np.corrcoef(left_array, right_array)[0, 1]
        if not np.isfinite(value):
            return 0.0
        return float(value)

    @staticmethod
    def _flow_to_distance(weight: float) -> float:
        """将流量型权重转换为路径算法使用的距离，流量越大距离越短。"""
        return 1.0 / max(float(weight), 1e-12)

    def _prepare_path_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """为最短路类中心性构造带 distance 属性的图副本。"""
        path_graph = graph.copy()
        for _, _, edge_data in path_graph.edges(data=True):
            edge_data['distance'] = self._flow_to_distance(edge_data.get('weight', 1.0))
        return path_graph

    def build_network(self, threshold: float = 0.01, method: str = 'trade_flow') -> nx.DiGraph:
        """
        构建供应链网络

        Args:
            threshold: 流量阈值（过滤弱连接）
            method: 构建方法 ('trade_flow', 'leontief', 'aggregated')

        Returns:
            NetworkX有向图
        """
        print("\n" + "=" * 70)
        print(f"步骤2: 构建供应链网络 (方法: {method})")
        print("=" * 70)

        if method == 'trade_flow':
            self.network = self._build_trade_flow_network(threshold)
        elif method == 'leontief':
            self.network = self._build_leontief_network(threshold)
        elif method == 'aggregated':
            self.network = self._build_aggregated_network()
        else:
            raise ValueError(f"未知方法: {method}")

        print(f"  ✓ 网络构建完成")
        print(f"  - 节点数: {self.network.number_of_nodes()}")
        print(f"  - 边数: {self.network.number_of_edges()}")

        return self.network

    def _build_trade_flow_network(self, threshold: float) -> nx.DiGraph:
        """基于贸易流构建网络"""
        G = nx.DiGraph()

        # 获取总产出
        x = self.mrio.x.values.flatten()

        # 构建流量矩阵（Z矩阵）
        Z_values = self.Z.values
        n_regions = len(self.regions)
        n_sectors = len(self.sectors)

        # 添加节点
        for i, region in enumerate(self.regions):
            for j, sector in enumerate(self.sectors):
                node_id = i * n_sectors + j
                node_name = f"{region}_{sector}"
                G.add_node(node_id, name=node_name, region=region, sector=sector)

        # 添加边（基于中间投入）
        total_flow = Z_values.sum()
        threshold_value = total_flow * threshold

        edges_added = 0
        for i in range(Z_values.shape[0]):
            for j in range(Z_values.shape[1]):
                if Z_values[i, j] > threshold_value:
                    G.add_edge(i, j, weight=float(Z_values[i, j]))
                    edges_added += 1

        print(f"  - 添加了 {edges_added} 条边 (阈值: {threshold_value:.2e})")

        return G

    def _build_leontief_network(self, threshold: float) -> nx.DiGraph:
        """基于Leontief逆矩阵构建网络"""
        G = nx.DiGraph()

        L_values = self.L.values
        n_regions = len(self.regions)
        n_sectors = len(self.sectors)

        # 添加节点
        for i, region in enumerate(self.regions):
            for j, sector in enumerate(self.sectors):
                node_id = i * n_sectors + j
                node_name = f"{region}_{sector}"
                G.add_node(node_id, name=node_name, region=region, sector=sector)

        # 添加边（基于Leontief系数）
        threshold_value = L_values.mean() * threshold
        edges_added = 0

        for i in range(L_values.shape[0]):
            for j in range(L_values.shape[1]):
                if i != j and L_values[i, j] > threshold_value:
                    G.add_edge(j, i, weight=float(L_values[i, j]))  # j→i 表示i需要j的投入
                    edges_added += 1

        print(f"  - 添加了 {edges_added} 条边 (阈值: {threshold_value:.4f})")

        return G

    def _build_aggregated_network(self) -> nx.DiGraph:
        """构建国家层面的聚合网络"""
        G = nx.DiGraph()

        # 按国家聚合贸易流
        n_regions = len(self.regions)
        n_sectors = len(self.sectors)
        Z_values = self.Z.values

        # 创建国家间流量矩阵
        country_flow = np.zeros((n_regions, n_regions))

        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    # 计算从国家i到国家j的总流量
                    start_i = i * n_sectors
                    end_i = (i + 1) * n_sectors
                    start_j = j * n_sectors
                    end_j = (j + 1) * n_sectors
                    flow = Z_values[start_i:end_i, start_j:end_j].sum()
                    country_flow[i, j] = flow

        # 添加节点
        for i, region in enumerate(self.regions):
            # 计算该国家的碳排放
            if self.co2_emissions is not None:
                start = i * n_sectors
                end = (i + 1) * n_sectors
                co2 = self.co2_emissions[start:end].sum()
            else:
                co2 = 0

            G.add_node(region, region=region, co2=float(co2))

        # 添加边
        threshold = country_flow[country_flow > 0].mean() * 0.1
        edges_added = 0

        for i in range(n_regions):
            for j in range(n_regions):
                if i != j and country_flow[i, j] > threshold:
                    G.add_edge(self.regions[i], self.regions[j],
                              weight=float(country_flow[i, j]))
                    edges_added += 1

        print(f"  - 节点数: {n_regions} (国家层面)")
        print(f"  - 添加了 {edges_added} 条边")

        return G

    def compute_centrality(self) -> Dict[str, pd.DataFrame]:
        """
        计算网络中心性指标

        Returns:
            Dict: 各类中心性指标结果
        """
        print("\n" + "=" * 70)
        print("步骤3: 计算网络中心性指标")
        print("=" * 70)

        if self.network is None:
            print("  ✗ 请先构建网络")
            return {}

        results = {}
        path_graph = self._prepare_path_graph(self.network)

        # 1. 度中心性
        print("  计算度中心性...")
        in_degree = nx.in_degree_centrality(self.network)
        out_degree = nx.out_degree_centrality(self.network)
        results['in_degree'] = in_degree
        results['out_degree'] = out_degree

        # 2. 加权度中心性
        print("  计算加权度中心性...")
        weighted_in = dict(self.network.in_degree(weight='weight'))
        weighted_out = dict(self.network.out_degree(weight='weight'))
        results['weighted_in_degree'] = weighted_in
        results['weighted_out_degree'] = weighted_out

        # 3. 中介中心性
        print("  计算中介中心性...")
        betweenness = nx.betweenness_centrality(path_graph, weight='distance')
        results['betweenness'] = betweenness

        # 4. 接近中心性
        print("  计算接近中心性...")
        try:
            closeness = nx.closeness_centrality(path_graph, distance='distance')
            results['closeness'] = closeness
        except Exception:
            print("    (跳过: 图不连通或距离无效)")
            results['closeness'] = {}

        # 5. PageRank
        print("  计算PageRank...")
        pagerank = nx.pagerank(self.network, weight='weight')
        results['pagerank'] = pagerank

        # 创建汇总DataFrame
        df_results = []
        for node in self.network.nodes():
            node_data = self.network.nodes[node]
            row = {
                'Node': node,
                'Name': node_data.get('name', str(node)),
                'Region': node_data.get('region', ''),
                'Sector': node_data.get('sector', ''),
                'In_Degree': in_degree.get(node, 0),
                'Out_Degree': out_degree.get(node, 0),
                'Weighted_In': weighted_in.get(node, 0),
                'Weighted_Out': weighted_out.get(node, 0),
                'Betweenness': betweenness.get(node, 0),
                'PageRank': pagerank.get(node, 0),
            }
            if results['closeness']:
                row['Closeness'] = results['closeness'].get(node, 0)
            df_results.append(row)

        df = pd.DataFrame(df_results)
        results['summary'] = df

        print(f"  ✓ 中心性计算完成")
        print(f"  - 节点数: {len(df)}")

        return results
    def compute_country_centrality(self) -> pd.DataFrame:
        """
        计算国家层面的聚合中心性

        Returns:
            DataFrame: 国家中心性结果
        """
        print("\n" + "=" * 70)
        print("步骤4: 计算国家层面中心性")
        print("=" * 70)

        if self.network is None:
            print("  ✗ 请先构建网络")
            return pd.DataFrame()

        n_regions = len(self.regions)
        n_sectors = len(self.sectors)

        # 聚合节点级中心性到国家层面
        node_centrality = self.compute_centrality()
        df = node_centrality['summary']

        # 按国家聚合
        country_stats = df.groupby('Region').agg({
            'In_Degree': 'mean',
            'Out_Degree': 'mean',
            'Weighted_In': 'sum',
            'Weighted_Out': 'sum',
            'Betweenness': 'sum',
            'PageRank': 'sum'
        }).reset_index()

        # 添加碳排放数据
        if self.co2_emissions is not None:
            co2_by_country = []
            for i, region in enumerate(self.regions):
                start = i * n_sectors
                end = (i + 1) * n_sectors
                co2 = self.co2_emissions[start:end].sum()
                co2_by_country.append(co2)
            country_stats['CO2_Emissions'] = co2_by_country

        # 添加就业数据
        if self.employment is not None:
            for skill, data in self.employment.items():
                emp_by_country = []
                for i in range(n_regions):
                    start = i * n_sectors
                    end = (i + 1) * n_sectors
                    emp = data[start:end].sum()
                    emp_by_country.append(emp)
                country_stats[f'Employment_{skill}'] = emp_by_country

        # 计算综合影响力指数
        country_stats['Influence_Index'] = (
            country_stats['PageRank'] * 0.4 +
            country_stats['Betweenness'] * 0.3 +
            country_stats['Weighted_In'] / country_stats['Weighted_In'].max() * 0.3
        )

        # 排序
        country_stats = country_stats.sort_values('Influence_Index', ascending=False)

        print(f"  ✓ 国家中心性计算完成")
        print(f"\n  [Top 10 最具影响力国家]")
        print(country_stats[['Region', 'Influence_Index', 'PageRank', 'Betweenness']].head(10).to_string())

        return country_stats

    @staticmethod
    def _signed_normalize_series(values: np.ndarray) -> np.ndarray:
        """把带符号向量缩放到 -1 到 1 区间。"""
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0:
            return array
        max_abs = float(np.max(np.abs(array)))
        if max_abs <= 0.0:
            return np.zeros_like(array)
        return array / max_abs

    @staticmethod
    def _aggregate_extension_total(extension_map: Optional[Dict[str, np.ndarray]], size: int) -> np.ndarray:
        """把按技能分组的扩展账户聚合为总量向量。"""
        if not extension_map:
            return np.zeros(size, dtype=float)
        total = np.zeros(size, dtype=float)
        for values in extension_map.values():
            total += np.asarray(values, dtype=float).reshape(-1)
        return total

    def _build_transfer_summary(self, transfer_matrix: np.ndarray, metric_name: str) -> pd.DataFrame:
        """把国家-国家转移矩阵整理为净转移摘要表。"""
        total_export = transfer_matrix.sum(axis=1)
        total_import = transfer_matrix.sum(axis=0)
        net_transfer = total_export - total_import
        summary = pd.DataFrame(
            {
                "Region": self.regions,
                f"Net_{metric_name}_Transfer": net_transfer,
                f"Total_{metric_name}_Export": total_export,
                f"Total_{metric_name}_Import": total_import,
            }
        )
        return summary.sort_values(f"Net_{metric_name}_Transfer", ascending=False).reset_index(drop=True)

    def _build_top_transfer_paths(
        self,
        transfer_matrix: np.ndarray,
        value_column: str,
        top_k: int,
    ) -> pd.DataFrame:
        """把国家-国家正向转移矩阵整理为 Top 路径表。"""
        paths = []
        for i, from_region in enumerate(self.regions):
            for j, to_region in enumerate(self.regions):
                if i == j or transfer_matrix[i, j] <= 0.0:
                    continue
                paths.append(
                    {
                        "From": from_region,
                        "To": to_region,
                        value_column: float(transfer_matrix[i, j]),
                    }
                )
        if not paths:
            return pd.DataFrame(columns=["From", "To", value_column])
        return pd.DataFrame(paths).sort_values(value_column, ascending=False).head(top_k).reset_index(drop=True)

    def _build_transfer_asymmetry(
        self,
        carbon_summary: pd.DataFrame,
        employment_summary: pd.DataFrame,
        wage_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """整合碳、就业、工资净转移，构造传导错配摘要表。"""
        merged = carbon_summary.merge(employment_summary, on="Region", how="outer").merge(
            wage_summary,
            on="Region",
            how="outer",
        )
        merged = merged.fillna(0.0)
        carbon_score = self._signed_normalize_series(merged["Net_Carbon_Transfer"].values)
        employment_score = self._signed_normalize_series(merged["Net_Employment_Transfer"].values)
        wage_score = self._signed_normalize_series(merged["Net_Wage_Transfer"].values)
        merged["Carbon_Transfer_Score"] = carbon_score
        merged["Employment_Transfer_Score"] = employment_score
        merged["Wage_Transfer_Score"] = wage_score
        merged["Carbon_Employment_Gap"] = carbon_score - employment_score
        merged["Carbon_Wage_Gap"] = carbon_score - wage_score
        merged["Transfer_Asymmetry_Index"] = np.abs(
            carbon_score - 0.5 * (employment_score + wage_score)
        )
        return merged.sort_values("Transfer_Asymmetry_Index", ascending=False).reset_index(drop=True)

    def _build_transfer_asymmetry_by_skill(
        self,
        carbon_skill_summary: pd.DataFrame,
        employment_skill_summary: pd.DataFrame,
        wage_skill_summary: pd.DataFrame,
    ) -> pd.DataFrame:
        """整合国家-技能层面的碳、就业、工资净转移错配。"""
        merged = carbon_skill_summary.merge(
            employment_skill_summary,
            on=["Region", "Skill"],
            how="outer",
        ).merge(
            wage_skill_summary,
            on=["Region", "Skill"],
            how="outer",
        )
        merged = merged.fillna(0.0)
        carbon_score = self._signed_normalize_series(merged["Net_Carbon_Transfer"].values)
        employment_score = self._signed_normalize_series(merged["Net_Employment_Transfer"].values)
        wage_score = self._signed_normalize_series(merged["Net_Wage_Transfer"].values)
        merged["Carbon_Transfer_Score"] = carbon_score
        merged["Employment_Transfer_Score"] = employment_score
        merged["Wage_Transfer_Score"] = wage_score
        merged["Carbon_Employment_Gap"] = carbon_score - employment_score
        merged["Carbon_Wage_Gap"] = carbon_score - wage_score
        merged["Transfer_Asymmetry_Index"] = np.abs(
            carbon_score - 0.5 * (employment_score + wage_score)
        )
        return merged.sort_values("Transfer_Asymmetry_Index", ascending=False).reset_index(drop=True)

    def analyze_transmission(self, top_k: int = 20) -> Dict:
        """
        分析隐含碳/就业/工资/工时传导路径。
    
        Args:
            top_k: 返回前K条最重要路径
    
        Returns:
            Dict: 传导分析结果
        """
        print("\n" + "=" * 70)
        print("步骤5: 分析隐含碳/就业/工资/工时传导")
        print("=" * 70)
    
        if self.L is None or self.Z is None or self.co2_emissions is None:
            print("  ✗ 缺少 L / Z / CO2 数据")
            return {}
        if not self.regions or not self.sectors:
            print("  ✗ 缺少国家或部门维度信息")
            return {}
    
        results = {}
        l_values = self.L.values if hasattr(self.L, "values") else np.asarray(self.L, dtype=float)
        z_values = self.Z.values if hasattr(self.Z, "values") else np.asarray(self.Z, dtype=float)
        n_regions = len(self.regions)
        n_sectors = len(self.sectors)
        total_dim = n_regions * n_sectors
    
        carbon_footprint = np.asarray(self.co2_emissions, dtype=float).reshape(-1) @ l_values
        employment_total = self._aggregate_extension_total(self.employment, total_dim)
        wage_total = self._aggregate_extension_total(self.wage, total_dim)
        hours_total = self._aggregate_extension_total(self.hours, total_dim)
        employment_footprint = employment_total @ l_values
        wage_footprint = wage_total @ l_values
        hours_footprint = hours_total @ l_values
    
        total_employment_coeff = np.clip(employment_total, 0.0, None)
        skill_share_map = {}
        for skill, values in (self.employment or {}).items():
            skill_vector = np.clip(np.asarray(values, dtype=float).reshape(-1), 0.0, None)
            share = np.divide(
                skill_vector,
                total_employment_coeff,
                out=np.zeros_like(skill_vector),
                where=total_employment_coeff > 0.0,
            )
            skill_share_map[skill] = share
    
        skill_wage_footprint_map = {
            skill: np.asarray(wage_vector, dtype=float).reshape(-1) @ l_values
            for skill, wage_vector in (self.wage or {}).items()
        }
        skill_hours_footprint_map = {
            skill: np.asarray(hours_vector, dtype=float).reshape(-1) @ l_values
            for skill, hours_vector in (self.hours or {}).items()
        }
        employment_skill_gender_footprint_map = {
            key: np.asarray(values, dtype=float).reshape(-1) @ l_values
            for key, values in (self.employment_skill_gender or {}).items()
        }
        hours_skill_gender_footprint_map = {
            key: np.asarray(values, dtype=float).reshape(-1) @ l_values
            for key, values in (self.hours_skill_gender or {}).items()
        }
    
        print("  计算国家间隐含转移矩阵...")
        carbon_transfer = np.zeros((n_regions, n_regions), dtype=float)
        employment_transfer = np.zeros((n_regions, n_regions), dtype=float)
        wage_transfer = np.zeros((n_regions, n_regions), dtype=float)
        hours_transfer = np.zeros((n_regions, n_regions), dtype=float)
        skill_carbon_transfer = {
            skill: np.zeros((n_regions, n_regions), dtype=float)
            for skill in (self.employment or {}).keys()
        }
        skill_employment_transfer = {
            skill: np.zeros((n_regions, n_regions), dtype=float)
            for skill in (self.employment or {}).keys()
        }
        skill_wage_transfer = {
            skill: np.zeros((n_regions, n_regions), dtype=float)
            for skill in (self.wage or {}).keys()
        }
        skill_hours_transfer = {
            skill: np.zeros((n_regions, n_regions), dtype=float)
            for skill in (self.hours or {}).keys()
        }
        employment_skill_gender_transfer = {
            key: np.zeros((n_regions, n_regions), dtype=float)
            for key in (self.employment_skill_gender or {}).keys()
        }
        hours_skill_gender_transfer = {
            key: np.zeros((n_regions, n_regions), dtype=float)
            for key in (self.hours_skill_gender or {}).keys()
        }
    
        for i in range(n_regions):
            start_i = i * n_sectors
            end_i = (i + 1) * n_sectors
            carbon_block = carbon_footprint[start_i:end_i].reshape(-1, 1)
            employment_block = employment_footprint[start_i:end_i].reshape(-1, 1)
            wage_block = wage_footprint[start_i:end_i].reshape(-1, 1)
            hours_block = hours_footprint[start_i:end_i].reshape(-1, 1)
    
            skill_share_blocks = {
                skill: share_vector[start_i:end_i].reshape(-1, 1)
                for skill, share_vector in skill_share_map.items()
            }
            skill_wage_blocks = {
                skill: footprint[start_i:end_i].reshape(-1, 1)
                for skill, footprint in skill_wage_footprint_map.items()
            }
            skill_hours_blocks = {
                skill: footprint[start_i:end_i].reshape(-1, 1)
                for skill, footprint in skill_hours_footprint_map.items()
            }
            employment_skill_gender_blocks = {
                key: footprint[start_i:end_i].reshape(-1, 1)
                for key, footprint in employment_skill_gender_footprint_map.items()
            }
            hours_skill_gender_blocks = {
                key: footprint[start_i:end_i].reshape(-1, 1)
                for key, footprint in hours_skill_gender_footprint_map.items()
            }
    
            for j in range(n_regions):
                if i == j:
                    continue
                start_j = j * n_sectors
                end_j = (j + 1) * n_sectors
                z_block = z_values[start_i:end_i, start_j:end_j]
                carbon_transfer[i, j] = float((carbon_block * z_block).sum())
                employment_transfer[i, j] = float((employment_block * z_block).sum())
                wage_transfer[i, j] = float((wage_block * z_block).sum())
                hours_transfer[i, j] = float((hours_block * z_block).sum())
    
                for skill, share_block in skill_share_blocks.items():
                    skill_carbon_transfer[skill][i, j] = float((carbon_block * share_block * z_block).sum())
                    skill_employment_transfer[skill][i, j] = float((employment_block * share_block * z_block).sum())
                for skill, skill_wage_block in skill_wage_blocks.items():
                    skill_wage_transfer[skill][i, j] = float((skill_wage_block * z_block).sum())
                for skill, skill_hours_block in skill_hours_blocks.items():
                    skill_hours_transfer[skill][i, j] = float((skill_hours_block * z_block).sum())
                for key, pair_block in employment_skill_gender_blocks.items():
                    employment_skill_gender_transfer[key][i, j] = float((pair_block * z_block).sum())
                for key, pair_block in hours_skill_gender_blocks.items():
                    hours_skill_gender_transfer[key][i, j] = float((pair_block * z_block).sum())
    
        results["carbon_transfer_matrix"] = carbon_transfer
        results["employment_transfer_matrix"] = employment_transfer
        results["wage_transfer_matrix"] = wage_transfer
        results["hours_transfer_matrix"] = hours_transfer
    
        carbon_summary = self._build_transfer_summary(carbon_transfer, "Carbon")
        employment_summary = self._build_transfer_summary(employment_transfer, "Employment")
        wage_summary = self._build_transfer_summary(wage_transfer, "Wage")
        hours_summary = self._build_transfer_summary(hours_transfer, "Hours")
        asymmetry_summary = self._build_transfer_asymmetry(
            carbon_summary,
            employment_summary,
            wage_summary,
        )
    
        results["net_transfer"] = carbon_summary
        results["employment_transfer"] = employment_summary
        results["wage_transfer"] = wage_summary
        results["hours_transfer"] = hours_summary
        results["transfer_asymmetry"] = asymmetry_summary
    
        carbon_skill_rows = []
        for skill, matrix in skill_carbon_transfer.items():
            skill_summary = self._build_transfer_summary(matrix, "Carbon")
            skill_summary.insert(1, "Skill", skill)
            carbon_skill_rows.append(skill_summary)
        results["carbon_transfer_by_skill"] = (
            pd.concat(carbon_skill_rows, ignore_index=True) if carbon_skill_rows else pd.DataFrame()
        )
    
        employment_skill_rows = []
        for skill, matrix in skill_employment_transfer.items():
            skill_summary = self._build_transfer_summary(matrix, "Employment")
            skill_summary.insert(1, "Skill", skill)
            employment_skill_rows.append(skill_summary)
        results["employment_transfer_by_skill"] = (
            pd.concat(employment_skill_rows, ignore_index=True) if employment_skill_rows else pd.DataFrame()
        )
    
        wage_skill_rows = []
        for skill, matrix in skill_wage_transfer.items():
            skill_summary = self._build_transfer_summary(matrix, "Wage")
            skill_summary.insert(1, "Skill", skill)
            wage_skill_rows.append(skill_summary)
        results["wage_transfer_by_skill"] = (
            pd.concat(wage_skill_rows, ignore_index=True) if wage_skill_rows else pd.DataFrame()
        )
    
        hours_skill_rows = []
        for skill, matrix in skill_hours_transfer.items():
            skill_summary = self._build_transfer_summary(matrix, "Hours")
            skill_summary.insert(1, "Skill", skill)
            hours_skill_rows.append(skill_summary)
        results["hours_transfer_by_skill"] = (
            pd.concat(hours_skill_rows, ignore_index=True) if hours_skill_rows else pd.DataFrame()
        )
    
        results["transfer_asymmetry_by_skill"] = self._build_transfer_asymmetry_by_skill(
            results["carbon_transfer_by_skill"],
            results["employment_transfer_by_skill"],
            results["wage_transfer_by_skill"],
        )
    
        results["employment_transfer_by_skill_gender"] = self._build_metric_summary_by_skill_gender(
            employment_skill_gender_transfer,
            "Employment",
        )
        results["hours_transfer_by_skill_gender"] = self._build_metric_summary_by_skill_gender(
            hours_skill_gender_transfer,
            "Hours",
        )
        results["bilateral_carbon_transfer_by_skill"] = self._build_bilateral_transfer_table_by_skill(
            skill_carbon_transfer,
            "Carbon_Transfer",
        )
        results["bilateral_employment_transfer_by_skill"] = self._build_bilateral_transfer_table_by_skill(
        skill_employment_transfer,
        "Employment_Transfer",
        )
        results["bilateral_wage_transfer_by_skill"] = self._build_bilateral_transfer_table_by_skill(
        skill_wage_transfer,
        "Wage_Transfer",
        )
        results["bilateral_hours_transfer_by_skill"] = self._build_bilateral_transfer_table_by_skill(

            skill_hours_transfer,
            "Hours_Transfer",
        )
    
        print("\n  [Top 10 净碳出口国]")
        print(carbon_summary.head(10).to_string(index=False))
        print("\n  [Top 10 净就业输出国]")
        print(employment_summary.head(10).to_string(index=False))
        print("\n  [Top 10 净工资输出国]")
        print(wage_summary.head(10).to_string(index=False))
        print("\n  [Top 10 净工时输出国]")
        print(hours_summary.head(10).to_string(index=False))
        print("\n  [Top 10 转移错配国家]")
        print(
            asymmetry_summary[
                [
                    "Region",
                    "Net_Carbon_Transfer",
                    "Net_Employment_Transfer",
                    "Net_Wage_Transfer",
                    "Carbon_Employment_Gap",
                    "Carbon_Wage_Gap",
                    "Transfer_Asymmetry_Index",
                ]
            ].head(10).to_string(index=False)
        )
        if not results["carbon_transfer_by_skill"].empty:
            print("\n  [Top 10 技能-碳净输出组合]")
            print(
                results["carbon_transfer_by_skill"]
                .sort_values("Net_Carbon_Transfer", ascending=False)
                .head(10)
                .to_string(index=False)
            )
        if not results["employment_transfer_by_skill"].empty:
            print("\n  [Top 10 技能-就业净输出组合]")
            print(
                results["employment_transfer_by_skill"]
                .sort_values("Net_Employment_Transfer", ascending=False)
                .head(10)
                .to_string(index=False)
            )
        if not results["hours_transfer_by_skill"].empty:
            print("\n  [Top 10 技能-工时净输出组合]")
            print(
                results["hours_transfer_by_skill"]
                .sort_values("Net_Hours_Transfer", ascending=False)
                .head(10)
                .to_string(index=False)
            )
        if not results["hours_transfer_by_skill_gender"].empty:
            print("\n  [Top 10 技能×性别工时净输出组合]")
            print(
                results["hours_transfer_by_skill_gender"]
                .sort_values("Net_Hours_Transfer", ascending=False)
                .head(10)
                .to_string(index=False)
            )
    
        top_carbon_paths = self._build_top_transfer_paths(carbon_transfer, "Carbon_Transfer", top_k)
        top_employment_paths = self._build_top_transfer_paths(employment_transfer, "Employment_Transfer", top_k)
        top_wage_paths = self._build_top_transfer_paths(wage_transfer, "Wage_Transfer", top_k)
        top_hours_paths = self._build_top_transfer_paths(hours_transfer, "Hours_Transfer", top_k)
        results["top_paths"] = top_carbon_paths
        results["top_employment_paths"] = top_employment_paths
        results["top_wage_paths"] = top_wage_paths
        results["top_hours_paths"] = top_hours_paths
    
        print(f"\n  [Top {top_k} 隐含碳传导路径]")
        print(top_carbon_paths.to_string(index=False))
        print(f"\n  [Top {top_k} 隐含就业传导路径]")
        print(top_employment_paths.to_string(index=False))
        print(f"\n  [Top {top_k} 隐含工资传导路径]")
        print(top_wage_paths.to_string(index=False))
        print(f"\n  [Top {top_k} 隐含工时传导路径]")
        print(top_hours_paths.to_string(index=False))
    
        print("\n  计算传导网络中心性...")
        g_carbon = nx.DiGraph()
        positive_transfer = carbon_transfer[carbon_transfer > 0]
        threshold = positive_transfer.mean() if positive_transfer.size > 0 else 0.0
    
        for region in self.regions:
            g_carbon.add_node(region)
        for i, from_region in enumerate(self.regions):
            for j, to_region in enumerate(self.regions):
                if i == j or carbon_transfer[i, j] <= threshold:
                    continue
                g_carbon.add_edge(
                    from_region,
                    to_region,
                    weight=float(carbon_transfer[i, j]),
                )
    
        path_graph = self._prepare_path_graph(g_carbon)
        betweenness = nx.betweenness_centrality(path_graph, weight="distance")
        transmission_centrality = pd.DataFrame(
            {
                "Region": list(betweenness.keys()),
                "Transmission_Centrality": list(betweenness.values()),
            }
        ).sort_values("Transmission_Centrality", ascending=False)
    
        results["transmission_centrality"] = transmission_centrality
        results["carbon_network"] = g_carbon
        results["transfer_metrics"] = {
            "carbon_employment_net_coupling": self._safe_corr(
                carbon_summary["Net_Carbon_Transfer"].values,
                employment_summary["Net_Employment_Transfer"].values,
            ),
            "carbon_wage_net_coupling": self._safe_corr(
                carbon_summary["Net_Carbon_Transfer"].values,
                wage_summary["Net_Wage_Transfer"].values,
            ),
            "carbon_hours_net_coupling": self._safe_corr(
                carbon_summary["Net_Carbon_Transfer"].values,
                hours_summary["Net_Hours_Transfer"].values,
            ),
            "transfer_asymmetry_mean": float(asymmetry_summary["Transfer_Asymmetry_Index"].mean()),
            "skill_transfer_asymmetry_mean": float(results["transfer_asymmetry_by_skill"]["Transfer_Asymmetry_Index"].mean())
            if not results["transfer_asymmetry_by_skill"].empty else 0.0,
            "skill_gender_employment_transfer_count": int(len(results["employment_transfer_by_skill_gender"])),
            "skill_gender_hours_transfer_count": int(len(results["hours_transfer_by_skill_gender"])),
        }
    
        print("\n  [Top 10 碳传导中介节点]")
        print(transmission_centrality.head(10).to_string(index=False))
    
        return results
    
    def build_country_sector_transfer_tables(self) -> Dict[str, pd.DataFrame]:
        """
        构建“需求国 -> 生产国 -> 部门”层级的隐含转移表。

        Returns:
            Dict[str, pd.DataFrame]: 四类指标对应的国家-国家-部门转移表
        """
        print("\n" + "=" * 70)
        print("步骤5.2: 构建国家-国家-部门转移表")
        print("=" * 70)

        if self.L is None or self.Y is None or self.co2_emissions is None:
            print("  ✗ 缺少 L / Y / CO2 数据")
            return {}
        if not self.regions or not self.sectors:
            print("  ✗ 缺少国家或部门维度信息")
            return {}

        l_values = self.L.values if hasattr(self.L, "values") else np.asarray(self.L, dtype=float)
        y_matrix = self.Y.values if hasattr(self.Y, "values") else np.asarray(self.Y, dtype=float)
        output_response = l_values @ y_matrix
        fd_labels = np.asarray(self._get_final_demand_region_labels(), dtype=object)
        n_regions = len(self.regions)
        n_sectors = len(self.sectors)
        total_dim = n_regions * n_sectors

        employment_total = self._aggregate_extension_total(self.employment, total_dim)
        wage_total = self._aggregate_extension_total(self.wage, total_dim)
        hours_total = self._aggregate_extension_total(self.hours, total_dim)

        carbon_rows = []
        employment_rows = []
        wage_rows = []
        hours_rows = []

        for demand_country in self.regions:
            demand_columns = np.where(fd_labels == demand_country)[0]
            if demand_columns.size == 0:
                continue
            demand_output = output_response[:, demand_columns]

            for producer_idx, producer_country in enumerate(self.regions):
                producer_slice = slice(producer_idx * n_sectors, (producer_idx + 1) * n_sectors)
                carbon_block = self.co2_emissions[producer_slice].reshape(-1, 1) * demand_output[producer_slice, :]
                employment_block = employment_total[producer_slice].reshape(-1, 1) * demand_output[producer_slice, :]
                wage_block = wage_total[producer_slice].reshape(-1, 1) * demand_output[producer_slice, :]
                hours_block = hours_total[producer_slice].reshape(-1, 1) * demand_output[producer_slice, :]

                for sector_idx, sector in enumerate(self.sectors):
                    carbon_value = float(carbon_block[sector_idx, :].sum())
                    employment_value = float(employment_block[sector_idx, :].sum())
                    wage_value = float(wage_block[sector_idx, :].sum())
                    hours_value = float(hours_block[sector_idx, :].sum())
                    if carbon_value <= 0.0 and employment_value <= 0.0 and wage_value <= 0.0 and hours_value <= 0.0:
                        continue

                    base_row = {
                        'Demand_Country': demand_country,
                        'Producer_Country': producer_country,
                        'Sector': sector,
                    }
                    if carbon_value > 0.0:
                        carbon_rows.append({**base_row, 'Carbon_Transfer': carbon_value})
                    if employment_value > 0.0:
                        employment_rows.append({**base_row, 'Employment_Transfer': employment_value})
                    if wage_value > 0.0:
                        wage_rows.append({**base_row, 'Wage_Transfer': wage_value})
                    if hours_value > 0.0:
                        hours_rows.append({**base_row, 'Hours_Transfer': hours_value})

        sector_results = {
            'carbon': pd.DataFrame(carbon_rows).sort_values('Carbon_Transfer', ascending=False).reset_index(drop=True) if carbon_rows else pd.DataFrame(),
            'employment': pd.DataFrame(employment_rows).sort_values('Employment_Transfer', ascending=False).reset_index(drop=True) if employment_rows else pd.DataFrame(),
            'wage': pd.DataFrame(wage_rows).sort_values('Wage_Transfer', ascending=False).reset_index(drop=True) if wage_rows else pd.DataFrame(),
            'hours': pd.DataFrame(hours_rows).sort_values('Hours_Transfer', ascending=False).reset_index(drop=True) if hours_rows else pd.DataFrame(),
        }

        for metric_name, df in sector_results.items():
            if not df.empty:
                print(f"  ✓ {metric_name} country-sector transfer rows: {len(df)}")

        return sector_results

    def build_country_skill_dual_layer_network(self, top_k_per_demand_country: int = 12) -> Dict[str, object]:
        """
        构建“需求国 -> 生产国×技能”的国家-技能双层网络。

        Args:
            top_k_per_demand_country: 每个需求国保留的高脆弱性边数量

        Returns:
            Dict[str, object]: 双层网络图、边表、节点汇总与指标
        """
        print("\n" + "=" * 70)
        print("步骤5.3: 构建国家-技能双层网络")
        print("=" * 70)

        if self.L is None or self.Y is None or self.co2_emissions is None:
            print("  ✗ 缺少 L / Y / CO2 数据")
            return {}
        if not self.regions or not self.sectors:
            print("  ✗ 缺少国家或部门维度信息")
            return {}
        if not self.employment or not self.wage:
            print("  ✗ 缺少就业或工资技能扩展数据")
            return {}

        l_values = self.L.values if hasattr(self.L, "values") else np.asarray(self.L, dtype=float)
        y_matrix = self.Y.values if hasattr(self.Y, "values") else np.asarray(self.Y, dtype=float)
        output_response = l_values @ y_matrix
        fd_labels = np.asarray(self._get_final_demand_region_labels(), dtype=object)
        n_regions = len(self.regions)
        n_sectors = len(self.sectors)
        total_dim = n_regions * n_sectors

        candidate_edges = []
        for demand_country in self.regions:
            demand_columns = np.where(fd_labels == demand_country)[0]
            if demand_columns.size == 0:
                continue
            demand_output = output_response[:, demand_columns]
            demand_carbon_by_sector = self.co2_emissions.reshape(-1, 1) * demand_output

            for producer_idx, producer_country in enumerate(self.regions):
                producer_slice = slice(producer_idx * n_sectors, (producer_idx + 1) * n_sectors)
                carbon_transfer = float(demand_carbon_by_sector[producer_slice, :].sum())
                if carbon_transfer <= 0.0:
                    continue

                employment_by_skill = {}
                wage_by_skill = {}
                employment_total = 0.0
                for skill in ("Low", "Medium", "High"):
                    emp_vector = np.asarray(self.employment.get(skill, np.zeros(total_dim)), dtype=float)
                    wage_vector = np.asarray(self.wage.get(skill, np.zeros(total_dim)), dtype=float)
                    embodied_employment = float((emp_vector[producer_slice].reshape(-1, 1) * demand_output[producer_slice, :]).sum())
                    embodied_wage = float((wage_vector[producer_slice].reshape(-1, 1) * demand_output[producer_slice, :]).sum())
                    employment_by_skill[skill] = embodied_employment
                    wage_by_skill[skill] = embodied_wage
                    employment_total += embodied_employment

                for skill in ("Low", "Medium", "High"):
                    if employment_total > 0.0:
                        skill_carbon = carbon_transfer * employment_by_skill[skill] / employment_total
                    else:
                        skill_carbon = carbon_transfer / 3.0
                    candidate_edges.append(
                        {
                            "Demand_Country": demand_country,
                            "Producer_Country": producer_country,
                            "Skill": skill,
                            "Carbon_Transfer": float(skill_carbon),
                            "Employment_Embodied": float(employment_by_skill[skill]),
                            "Wage_Embodied": float(wage_by_skill[skill]),
                        }
                    )

        if not candidate_edges:
            print("  ✗ 未生成双层网络边")
            return {}

        edge_table = pd.DataFrame(candidate_edges)
        edge_table["Carbon_norm"] = self._normalize_series(edge_table["Carbon_Transfer"].values)
        edge_table["Employment_norm"] = self._normalize_series(edge_table["Employment_Embodied"].values)
        edge_table["Wage_norm"] = self._normalize_series(edge_table["Wage_Embodied"].values)
        edge_table["Low_Wage_Penalty"] = 1.0 - edge_table["Wage_norm"]
        edge_table["Vulnerability_Weight"] = (
            edge_table["Carbon_norm"] * 0.45
            + edge_table["Employment_norm"] * 0.35
            + edge_table["Low_Wage_Penalty"] * 0.20
        )
        edge_table = (
            edge_table.sort_values(["Demand_Country", "Vulnerability_Weight"], ascending=[True, False])
            .groupby("Demand_Country", as_index=False)
            .head(top_k_per_demand_country)
            .reset_index(drop=True)
        )

        graph = nx.DiGraph()
        for country in self.regions:
            graph.add_node(country, layer="country", node_type="country", region=country)
        for producer_country in self.regions:
            for skill in ("Low", "Medium", "High"):
                node_id = f"{producer_country}|{skill}"
                graph.add_node(
                    node_id,
                    layer="country_skill",
                    node_type="country_skill",
                    region=producer_country,
                    skill=skill,
                )
        for _, row in edge_table.iterrows():
            node_id = f"{row['Producer_Country']}|{row['Skill']}"
            graph.add_edge(
                row["Demand_Country"],
                node_id,
                weight=float(row["Vulnerability_Weight"]),
                carbon_transfer=float(row["Carbon_Transfer"]),
                employment_embodied=float(row["Employment_Embodied"]),
                wage_embodied=float(row["Wage_Embodied"]),
            )

        skill_node_table = edge_table.groupby(["Producer_Country", "Skill"], as_index=False).agg({
            "Carbon_Transfer": "sum",
            "Employment_Embodied": "sum",
            "Wage_Embodied": "sum",
            "Demand_Country": "nunique",
        }).rename(columns={
            "Carbon_Transfer": "Embodied_Carbon_In",
            "Employment_Embodied": "Embodied_Employment_In",
            "Wage_Embodied": "Embodied_Wage_In",
            "Demand_Country": "Demand_Country_Count",
        })
        skill_node_table["Carbon_norm"] = self._normalize_series(skill_node_table["Embodied_Carbon_In"].values)
        skill_node_table["Employment_norm"] = self._normalize_series(skill_node_table["Embodied_Employment_In"].values)
        skill_node_table["Wage_norm"] = self._normalize_series(skill_node_table["Embodied_Wage_In"].values)
        skill_node_table["Triple_Inequality_Index"] = (
            skill_node_table["Carbon_norm"] * 0.45
            + skill_node_table["Employment_norm"] * 0.35
            + (1.0 - skill_node_table["Wage_norm"]) * 0.20
        )
        skill_node_table = skill_node_table.sort_values("Triple_Inequality_Index", ascending=False).reset_index(drop=True)

        country_node_table = edge_table.groupby("Demand_Country", as_index=False).agg({
            "Carbon_Transfer": "sum",
            "Employment_Embodied": "sum",
            "Wage_Embodied": "sum",
            "Producer_Country": "nunique",
        }).rename(columns={
            "Carbon_Transfer": "Pulled_Carbon",
            "Employment_Embodied": "Pulled_Employment",
            "Wage_Embodied": "Pulled_Wage",
            "Producer_Country": "Producer_Country_Count",
        })

        metrics = {
            "country_nodes": len(self.regions),
            "skill_nodes": len(self.regions) * 3,
            "edges": int(len(edge_table)),
            "density": float(len(edge_table) / max(len(self.regions) * len(self.regions) * 3, 1)),
            "carbon_employment_coupling": self._safe_corr(
                skill_node_table["Embodied_Carbon_In"].values,
                skill_node_table["Embodied_Employment_In"].values,
            ),
            "carbon_wage_coupling": self._safe_corr(
                skill_node_table["Embodied_Carbon_In"].values,
                skill_node_table["Embodied_Wage_In"].values,
            ),
        }

        self.country_skill_network = graph
        self.country_skill_results = {
            "graph": graph,
            "edge_table": edge_table,
            "skill_node_table": skill_node_table,
            "country_node_table": country_node_table,
            "metrics": metrics,
        }

        print(f"  ✓ 双层网络节点: 国家 {metrics['country_nodes']} + 技能节点 {metrics['skill_nodes']}")
        print(f"  ✓ 双层网络边数: {metrics['edges']}")
        print("\n  [Top 10 国家-技能暴露节点]")
        print(skill_node_table.head(10).to_string())

        return self.country_skill_results

    def identify_key_sectors(self, top_k: int = 10, method: str = 'aggregated') -> pd.DataFrame:
        """
        识别关键部门（高碳+高就业+高中心性）
    
        Args:
            top_k: 返回前K个关键部门
            method: 'aggregated' 基于聚合网络, 'sector' 基于部门网络
    
        Returns:
            DataFrame: 关键部门分析结果
        """
        print("\n" + "=" * 70)
        print("步骤6: 识别关键部门")
        print("=" * 70)
    
        if method != 'aggregated':
            print("  提示: 如需部门级别网络分析，请使用 build_network(method='trade_flow')")
            return pd.DataFrame()
    
        if self.network is None:
            print("  ✗ 请先构建网络")
            return pd.DataFrame()
    
        n_regions = len(self.regions)
        n_sectors = len(self.sectors)
        output_vector = self.mrio.x.values.flatten() if self.mrio is not None else np.zeros(n_regions * n_sectors)
        z_values = self.Z.values if self.Z is not None else np.zeros((n_regions * n_sectors, n_regions * n_sectors))
    
        results = []
        for i, region in enumerate(self.regions):
            for j, sector in enumerate(self.sectors):
                idx = i * n_sectors + j
                direct_co2 = float(self.co2_emissions[idx] * output_vector[idx]) if self.co2_emissions is not None else 0.0
                row = {
                    'Node': idx,
                    'Region': region,
                    'Sector': sector,
                    'Direct_CO2': direct_co2,
                    'Flow_In': float(z_values[:, idx].sum()),
                    'Flow_Out': float(z_values[idx, :].sum()),
                }
                row['Flow_Total'] = row['Flow_In'] + row['Flow_Out']
    
                if self.employment is not None:
                    emp_total = 0.0
                    for skill, data in self.employment.items():
                        emp_value = float(data[idx] * output_vector[idx])
                        row[f'Emp_{skill}'] = emp_value
                        emp_total += emp_value
                    row['Emp_Total'] = emp_total
    
                results.append(row)
    
        df = pd.DataFrame(results)
    
        df['CO2_norm'] = df['Direct_CO2'] / df['Direct_CO2'].max() if df['Direct_CO2'].max() > 0 else 0.0
        if 'Emp_Total' in df.columns and df['Emp_Total'].max() > 0:
            df['Emp_norm'] = df['Emp_Total'] / df['Emp_Total'].max()
        else:
            df['Emp_norm'] = 0.0
        df['Flow_norm'] = df['Flow_Total'] / df['Flow_Total'].max() if df['Flow_Total'].max() > 0 else 0.0
        df['Criticality_Index'] = df['CO2_norm'] * 0.4 + df['Emp_norm'] * 0.3 + df['Flow_norm'] * 0.3
    
        df = df.sort_values('Criticality_Index', ascending=False).reset_index(drop=True)
    
        print(f"\n  [Top {top_k} 关键部门]")
        cols = ['Region', 'Sector', 'Criticality_Index', 'Direct_CO2', 'Flow_Total']
        if 'Emp_Total' in df.columns:
            cols.append('Emp_Total')
        print(df[cols].head(top_k).to_string())
    
        return df
    def compute_network_metrics(self) -> Dict:
        """
        计算网络整体指标
    
        Returns:
            Dict: 网络指标
        """
        print("\n" + "=" * 70)
        print("步骤7: 计算网络整体指标")
        print("=" * 70)
    
        if self.network is None:
            print("  ✗ 请先构建网络")
            return {}
    
        G = self.network
        metrics = {}
    
        # 基本指标
        metrics['nodes'] = G.number_of_nodes()
        metrics['edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
    
        # 连通性
        if nx.is_weakly_connected(G):
            metrics['weakly_connected'] = True
            metrics['num_weakly_connected_components'] = 1
        else:
            metrics['weakly_connected'] = False
            metrics['num_weakly_connected_components'] = nx.number_weakly_connected_components(G)
    
        # 度分布
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        metrics['avg_in_degree'] = np.mean(in_degrees)
        metrics['avg_out_degree'] = np.mean(out_degrees)
        metrics['std_in_degree'] = np.std(in_degrees)
        metrics['std_out_degree'] = np.std(out_degrees)
    
        # 加权度
        weighted_in = [d for n, d in G.in_degree(weight='weight')]
        weighted_out = [d for n, d in G.out_degree(weight='weight')]
        metrics['total_flow_in'] = sum(weighted_in)
        metrics['total_flow_out'] = sum(weighted_out)
    
        # 聚类系数（对于有向图）
        try:
            metrics['clustering_coefficient'] = nx.average_clustering(G.to_undirected())
        except:
            metrics['clustering_coefficient'] = None
    
        # 互惠性
        try:
            metrics['reciprocity'] = nx.reciprocity(G)
        except:
            metrics['reciprocity'] = None
    
        # 打印结果
        print(f"  节点数: {metrics['nodes']}")
        print(f"  边数: {metrics['edges']}")
        print(f"  密度: {metrics['density']:.4f}")
        print(f"  平均入度: {metrics['avg_in_degree']:.2f}")
        print(f"  平均出度: {metrics['avg_out_degree']:.2f}")
        print(f"  总流入: {metrics['total_flow_in']:.2e}")
        print(f"  总流出: {metrics['total_flow_out']:.2e}")
        if metrics['clustering_coefficient']:
            print(f"  聚类系数: {metrics['clustering_coefficient']:.4f}")
        if metrics['reciprocity']:
            print(f"  互惠性: {metrics['reciprocity']:.4f}")
    
        return metrics
    

    def save_results(self, output_dir: str) -> Dict:
        """
        保存分析结果。

        Args:
            output_dir: 输出目录
        """
        print("\n" + "=" * 70)
        print("步骤8: 保存结果")
        print("=" * 70)

        output_path = Path(output_dir)
        csv_path = output_path / "csv"
        json_path = output_path / "json"
        csv_path.mkdir(parents=True, exist_ok=True)
        json_path.mkdir(parents=True, exist_ok=True)

        centrality = self.compute_centrality()
        if 'summary' in centrality:
            centrality['summary'].to_csv(csv_path / 'network_centrality.csv', index=False)
            print(f"  ✓ 保存: {csv_path / 'network_centrality.csv'}")

        country_cent = self.compute_country_centrality()
        if not country_cent.empty:
            country_cent.to_csv(csv_path / 'network_country_centrality.csv', index=False)
            print(f"  ✓ 保存: {csv_path / 'network_country_centrality.csv'}")

        transmission = self.analyze_transmission()
        country_sector = self.build_country_sector_transfer_tables()

        csv_frames = {
            'network_carbon_transfer.csv': transmission.get('net_transfer'),
            'network_employment_transfer.csv': transmission.get('employment_transfer'),
            'network_wage_transfer.csv': transmission.get('wage_transfer'),
            'network_hours_transfer.csv': transmission.get('hours_transfer'),
            'network_carbon_transfer_by_skill.csv': transmission.get('carbon_transfer_by_skill'),
            'network_employment_transfer_by_skill.csv': transmission.get('employment_transfer_by_skill'),
            'network_wage_transfer_by_skill.csv': transmission.get('wage_transfer_by_skill'),
            'network_hours_transfer_by_skill.csv': transmission.get('hours_transfer_by_skill'),
            'network_transfer_asymmetry.csv': transmission.get('transfer_asymmetry'),
            'network_transfer_asymmetry_by_skill.csv': transmission.get('transfer_asymmetry_by_skill'),
            'network_employment_transfer_by_skill_gender.csv': transmission.get('employment_transfer_by_skill_gender'),
            'network_hours_transfer_by_skill_gender.csv': transmission.get('hours_transfer_by_skill_gender'),
            'network_bilateral_carbon_transfer_by_skill.csv': transmission.get('bilateral_carbon_transfer_by_skill'),
            'network_bilateral_employment_transfer_by_skill.csv': transmission.get('bilateral_employment_transfer_by_skill'),
            'network_bilateral_wage_transfer_by_skill.csv': transmission.get('bilateral_wage_transfer_by_skill'),
            'network_bilateral_hours_transfer_by_skill.csv': transmission.get('bilateral_hours_transfer_by_skill'),
            'network_top_paths.csv': transmission.get('top_paths'),
            'network_top_employment_paths.csv': transmission.get('top_employment_paths'),
            'network_top_wage_paths.csv': transmission.get('top_wage_paths'),
            'network_top_hours_paths.csv': transmission.get('top_hours_paths'),
            'network_transmission_centrality.csv': transmission.get('transmission_centrality'),
        }
        for file_name, df in csv_frames.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(csv_path / file_name, index=False)
                print(f"  ✓ 保存: {csv_path / file_name}")

        sector_frames = {
            'network_bilateral_carbon_transfer_by_country_sector.csv': country_sector.get('carbon'),
            'network_bilateral_employment_transfer_by_country_sector.csv': country_sector.get('employment'),
            'network_bilateral_wage_transfer_by_country_sector.csv': country_sector.get('wage'),
            'network_bilateral_hours_transfer_by_country_sector.csv': country_sector.get('hours'),
        }
        for file_name, df in sector_frames.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(csv_path / file_name, index=False)
                print(f"  ✓ 保存: {csv_path / file_name}")

        dual_layer = self.build_country_skill_dual_layer_network()
        if dual_layer:
            dual_layer['edge_table'].to_csv(csv_path / 'network_country_skill_edges.csv', index=False)
            print(f"  ✓ 保存: {csv_path / 'network_country_skill_edges.csv'}")
            dual_layer['skill_node_table'].to_csv(csv_path / 'network_country_skill_summary.csv', index=False)
            print(f"  ✓ 保存: {csv_path / 'network_country_skill_summary.csv'}")
            dual_layer['country_node_table'].to_csv(csv_path / 'network_country_layer_summary.csv', index=False)
            print(f"  ✓ 保存: {csv_path / 'network_country_layer_summary.csv'}")

            country_skill_metrics_path = json_path / 'network_country_skill_metrics.json'
            with open(country_skill_metrics_path, 'w') as f:
                json.dump(dual_layer['metrics'], f, indent=2, default=float)
            print(f"  ✓ 保存: {country_skill_metrics_path}")
            pd.json_normalize(dual_layer['metrics'], sep='_').to_csv(
                csv_path / 'network_country_skill_metrics.csv',
                index=False,
            )
            print(f"  ✓ 保存: {csv_path / 'network_country_skill_metrics.csv'}")

        key_sectors = self.identify_key_sectors()
        if not key_sectors.empty:
            key_sectors.to_csv(csv_path / 'network_key_sectors.csv', index=False)
            print(f"  ✓ 保存: {csv_path / 'network_key_sectors.csv'}")

        metrics = self.compute_network_metrics()
        if 'transfer_metrics' in transmission:
            metrics['transfer_metrics'] = transmission['transfer_metrics']
        metrics_json_path = json_path / 'network_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=float)
        print(f"  ✓ 保存: {metrics_json_path}")
        pd.json_normalize(metrics, sep='_').to_csv(csv_path / 'network_metrics.csv', index=False)
        print(f"  ✓ 保存: {csv_path / 'network_metrics.csv'}")

        print("\n  ✓ 所有结果保存完成")
        return {"transmission": transmission, "country_sector": country_sector}


# =========================================================================
# 主函数
# =========================================================================


