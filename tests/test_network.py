"""Synthetic tests for bilateral transfers and their network interpretation."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd

from carbon_labour import network as MODULE

NetworkEngine = MODULE.NetworkEngine


def test_flow_centrality_uses_inverse_distance_for_path_metrics() -> None:
    """流量越大应越容易形成主通路，而不是在最短路里被当作更长距离。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.network = nx.DiGraph()
    engine.network.add_node("A", name="A", region="R1", sector="S1")
    engine.network.add_node("B", name="B", region="R1", sector="S2")
    engine.network.add_node("C", name="C", region="R1", sector="S3")
    engine.network.add_edge("A", "B", weight=100.0)
    engine.network.add_edge("A", "C", weight=1.0)
    engine.network.add_edge("C", "B", weight=1.0)

    summary = engine.compute_centrality()["summary"].set_index("Node")

    assert summary.loc["C", "Betweenness"] == 0.0



def test_identify_key_sectors_uses_totals_instead_of_raw_coefficients() -> None:
    """关键部门识别应基于总量而非系数，否则会被极端小部门误导。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.regions = ["R1"]
    engine.sectors = ["DirtyLarge", "TinyExtreme"]
    engine.network = nx.DiGraph()
    engine.network.add_node("R1")
    engine.co2_emissions = np.array([10.0, 1000.0])
    engine.employment = {
        "Low": np.array([5.0, 100.0]),
        "Medium": np.array([0.0, 0.0]),
    }
    engine.mrio = SimpleNamespace(x=pd.Series([1000.0, 1.0]))
    engine.Z = pd.DataFrame([[0.0, 50.0], [10.0, 0.0]])

    result = engine.identify_key_sectors(top_k=2)

    assert result.iloc[0]["Sector"] == "DirtyLarge"
    assert result.iloc[0]["Direct_CO2"] > result.iloc[1]["Direct_CO2"]



def test_country_skill_dual_layer_network_combines_carbon_employment_and_wage() -> None:
    """国家-技能双层网络应同时跟踪隐含碳、就业承载和工资承载。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.regions = ["R1", "R2"]
    engine.sectors = ["Dirty", "Clean"]
    engine.L = np.eye(4)
    engine.Y = np.array(
        [
            [2.0, 14.0],
            [1.0, 6.0],
            [8.0, 1.0],
            [5.0, 1.0],
        ]
    )
    engine.co2_emissions = np.array([5.0, 1.0, 2.0, 0.2])
    engine.employment = {
        "Low": np.array([4.0, 2.0, 1.0, 0.5]),
        "Medium": np.array([1.0, 1.5, 1.0, 0.8]),
        "High": np.array([0.2, 0.6, 0.8, 1.2]),
    }
    engine.wage = {
        "Low": np.array([1.0, 1.2, 1.5, 1.6]),
        "Medium": np.array([2.0, 2.5, 2.8, 3.0]),
        "High": np.array([5.0, 5.5, 6.0, 6.5]),
    }

    result = engine.build_country_skill_dual_layer_network(top_k_per_demand_country=4)

    assert {"graph", "edge_table", "skill_node_table", "country_node_table", "metrics"}.issubset(result.keys())

    edge_table = result["edge_table"]
    skill_node_table = result["skill_node_table"]
    metrics = result["metrics"]

    assert {
        "Demand_Country",
        "Producer_Country",
        "Skill",
        "Carbon_Transfer",
        "Employment_Embodied",
        "Wage_Embodied",
        "Vulnerability_Weight",
    }.issubset(edge_table.columns)
    assert {
        "Producer_Country",
        "Skill",
        "Embodied_Carbon_In",
        "Embodied_Employment_In",
        "Embodied_Wage_In",
        "Triple_Inequality_Index",
    }.issubset(skill_node_table.columns)
    assert metrics["country_nodes"] == 2
    assert metrics["skill_nodes"] == 6

    top_node = skill_node_table.sort_values("Triple_Inequality_Index", ascending=False).iloc[0]
    assert top_node["Producer_Country"] == "R1"
    assert top_node["Skill"] == "Low"



def test_analyze_transmission_outputs_carbon_employment_wage_hours_and_asymmetry() -> None:
    """传导分析应同时输出碳、就业、工资、工时净转移及其错配摘要。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.regions = ["R1", "R2"]
    engine.sectors = ["Dirty", "Clean"]
    engine.L = np.eye(4)
    engine.Z = pd.DataFrame(
        [
            [0.0, 0.0, 8.0, 2.0],
            [0.0, 0.0, 4.0, 1.0],
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 0.2, 0.0, 0.0],
        ]
    )
    engine.co2_emissions = np.array([5.0, 1.0, 2.0, 0.2])
    engine.employment = {
        "Low": np.array([4.0, 2.0, 1.0, 0.5]),
        "Medium": np.array([1.0, 1.5, 1.0, 0.8]),
        "High": np.array([0.2, 0.6, 0.8, 1.2]),
    }
    engine.hours = {
        "Low": np.array([8.0, 6.0, 2.0, 1.0]),
        "Medium": np.array([3.0, 4.0, 3.0, 2.0]),
        "High": np.array([1.0, 2.0, 2.0, 3.0]),
    }
    engine.wage = {
        "Low": np.array([1.0, 1.2, 1.5, 1.6]),
        "Medium": np.array([2.0, 2.5, 2.8, 3.0]),
        "High": np.array([5.0, 5.5, 6.0, 6.5]),
    }

    result = engine.analyze_transmission(top_k=4)

    assert {
        "net_transfer",
        "employment_transfer",
        "wage_transfer",
        "hours_transfer",
        "carbon_transfer_by_skill",
        "employment_transfer_by_skill",
        "wage_transfer_by_skill",
        "transfer_asymmetry",
        "transfer_asymmetry_by_skill",
        "top_paths",
        "top_employment_paths",
        "top_wage_paths",
        "top_hours_paths",
    }.issubset(result.keys())

    carbon_df = result["net_transfer"]
    employment_df = result["employment_transfer"]
    wage_df = result["wage_transfer"]
    hours_df = result["hours_transfer"]
    carbon_skill_df = result["carbon_transfer_by_skill"]
    employment_skill_df = result["employment_transfer_by_skill"]
    wage_skill_df = result["wage_transfer_by_skill"]
    asymmetry_df = result["transfer_asymmetry"]
    asymmetry_skill_df = result["transfer_asymmetry_by_skill"]

    assert {"Region", "Net_Carbon_Transfer", "Total_Carbon_Export", "Total_Carbon_Import"}.issubset(carbon_df.columns)
    assert {"Region", "Net_Employment_Transfer", "Total_Employment_Export", "Total_Employment_Import"}.issubset(employment_df.columns)
    assert {"Region", "Net_Wage_Transfer", "Total_Wage_Export", "Total_Wage_Import"}.issubset(wage_df.columns)
    assert {"Region", "Net_Hours_Transfer", "Total_Hours_Export", "Total_Hours_Import"}.issubset(hours_df.columns)
    assert {"Region", "Skill", "Net_Carbon_Transfer"}.issubset(carbon_skill_df.columns)
    assert {"Region", "Skill", "Net_Employment_Transfer"}.issubset(employment_skill_df.columns)
    assert {"Region", "Skill", "Net_Wage_Transfer"}.issubset(wage_skill_df.columns)
    assert {
        "Region",
        "Carbon_Employment_Gap",
        "Carbon_Wage_Gap",
        "Transfer_Asymmetry_Index",
    }.issubset(asymmetry_df.columns)
    assert {
        "Region",
        "Skill",
        "Transfer_Asymmetry_Index",
    }.issubset(asymmetry_skill_df.columns)

    assert carbon_df.iloc[0]["Net_Carbon_Transfer"] > 0.0
    assert employment_df.iloc[0]["Net_Employment_Transfer"] > 0.0
    assert wage_df.iloc[0]["Net_Wage_Transfer"] > 0.0
    assert hours_df.iloc[0]["Net_Hours_Transfer"] > 0.0
    assert set(carbon_skill_df["Skill"]) == {"Low", "Medium", "High"}
    assert set(employment_skill_df["Skill"]) == {"Low", "Medium", "High"}
    assert set(wage_skill_df["Skill"]) == {"Low", "Medium", "High"}
    assert asymmetry_df["Transfer_Asymmetry_Index"].ge(0.0).all()
    assert asymmetry_skill_df["Transfer_Asymmetry_Index"].ge(0.0).all()
    assert not result["top_employment_paths"].empty
    assert not result["top_wage_paths"].empty
    assert not result["top_hours_paths"].empty



def test_hours_skill_gender_extension_tables_can_be_summarized() -> None:
    """工时技能×性别双边转移表应能整理为净转移摘要与双边明细。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.regions = ["R1", "R2"]
    transfer_map = {
        "Low_Male": np.array([[0.0, 5.0], [1.0, 0.0]]),
        "High_Female": np.array([[0.0, 2.0], [4.0, 0.0]]),
    }

    summary = engine._build_metric_summary_by_skill_gender(transfer_map, "Hours")
    bilateral = engine._build_bilateral_transfer_table_by_skill_gender(transfer_map, "Hours_Transfer")

    assert {"Region", "Skill", "Gender", "Net_Hours_Transfer"}.issubset(summary.columns)
    assert {"From_Region", "To_Region", "Skill", "Gender", "Hours_Transfer"}.issubset(bilateral.columns)
    assert set(summary["Gender"]) == {"Male", "Female"}
    assert not bilateral.empty



def test_analyze_transmission_outputs_skill_gender_summaries_and_skill_bilateral_tables() -> None:
    """传导分析应保留 skill×gender 摘要，同时输出 skill-level bilateral 三张表。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.regions = ["R1", "R2"]
    engine.sectors = ["Dirty", "Clean"]
    engine.L = np.eye(4)
    engine.Z = pd.DataFrame(
        [
            [0.0, 0.0, 8.0, 2.0],
            [0.0, 0.0, 4.0, 1.0],
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 0.2, 0.0, 0.0],
        ]
    )
    engine.co2_emissions = np.array([5.0, 1.0, 2.0, 0.2])
    engine.employment = {
        "Low": np.array([4.0, 2.0, 1.0, 0.5]),
        "Medium": np.array([1.0, 1.5, 1.0, 0.8]),
        "High": np.array([0.2, 0.6, 0.8, 1.2]),
    }
    engine.hours = {
        "Low": np.array([8.0, 6.0, 2.0, 1.0]),
        "Medium": np.array([3.0, 4.0, 3.0, 2.0]),
        "High": np.array([1.0, 2.0, 2.0, 3.0]),
    }
    engine.wage = {
        "Low": np.array([1.0, 1.2, 1.5, 1.6]),
        "Medium": np.array([2.0, 2.5, 2.8, 3.0]),
        "High": np.array([5.0, 5.5, 6.0, 6.5]),
    }
    engine.employment_skill_gender = {
        "Low_Male": np.array([2.5, 1.0, 0.6, 0.2]),
        "Low_Female": np.array([1.5, 1.0, 0.4, 0.3]),
        "Medium_Male": np.array([0.7, 0.8, 0.4, 0.3]),
        "Medium_Female": np.array([0.3, 0.7, 0.6, 0.5]),
        "High_Male": np.array([0.1, 0.3, 0.3, 0.5]),
        "High_Female": np.array([0.1, 0.3, 0.5, 0.7]),
    }
    engine.hours_skill_gender = {
        "Low_Male": np.array([5.0, 2.0, 1.2, 0.4]),
        "Low_Female": np.array([3.0, 2.0, 0.8, 0.6]),
        "Medium_Male": np.array([1.5, 2.0, 1.2, 0.7]),
        "Medium_Female": np.array([1.5, 2.0, 1.8, 1.3]),
        "High_Male": np.array([0.4, 0.8, 0.8, 1.2]),
        "High_Female": np.array([0.6, 1.2, 1.2, 1.8]),
    }

    result = engine.analyze_transmission(top_k=4)

    assert {
        "employment_transfer_by_skill_gender",
        "hours_transfer_by_skill_gender",
        "bilateral_carbon_transfer_by_skill",
        "bilateral_employment_transfer_by_skill",
        "bilateral_hours_transfer_by_skill",
    }.issubset(result.keys())

    emp_gender_df = result["employment_transfer_by_skill_gender"]
    hours_gender_df = result["hours_transfer_by_skill_gender"]
    carbon_bilateral_df = result["bilateral_carbon_transfer_by_skill"]
    emp_bilateral_df = result["bilateral_employment_transfer_by_skill"]
    hours_bilateral_df = result["bilateral_hours_transfer_by_skill"]

    assert {"Region", "Skill", "Gender", "Net_Employment_Transfer"}.issubset(emp_gender_df.columns)
    assert {"Region", "Skill", "Gender", "Net_Hours_Transfer"}.issubset(hours_gender_df.columns)
    assert {"From_Region", "To_Region", "Skill", "Carbon_Transfer"}.issubset(carbon_bilateral_df.columns)
    assert {"From_Region", "To_Region", "Skill", "Employment_Transfer"}.issubset(emp_bilateral_df.columns)
    assert {"From_Region", "To_Region", "Skill", "Hours_Transfer"}.issubset(hours_bilateral_df.columns)
    assert "Gender" not in carbon_bilateral_df.columns
    assert "Gender" not in emp_bilateral_df.columns
    assert "Gender" not in hours_bilateral_df.columns
    assert set(emp_gender_df["Gender"]) == {"Male", "Female"}
    assert set(hours_gender_df["Gender"]) == {"Male", "Female"}
    assert not carbon_bilateral_df.empty
    assert not emp_bilateral_df.empty
    assert not hours_bilateral_df.empty



def test_analyze_transmission_outputs_bilateral_wage_transfer_by_skill() -> None:
    """传导分析应输出按技能展开的双边工资转移表。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.regions = ["R1", "R2"]
    engine.sectors = ["Dirty", "Clean"]
    engine.L = np.eye(4)
    engine.Z = pd.DataFrame(
        [
            [0.0, 0.0, 8.0, 2.0],
            [0.0, 0.0, 4.0, 1.0],
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 0.2, 0.0, 0.0],
        ]
    )
    engine.co2_emissions = np.array([5.0, 1.0, 2.0, 0.2])
    engine.employment = {
        "Low": np.array([4.0, 2.0, 1.0, 0.5]),
        "Medium": np.array([1.0, 1.5, 1.0, 0.8]),
        "High": np.array([0.2, 0.6, 0.8, 1.2]),
    }
    engine.hours = {
        "Low": np.array([8.0, 6.0, 2.0, 1.0]),
        "Medium": np.array([3.0, 4.0, 3.0, 2.0]),
        "High": np.array([1.0, 2.0, 2.0, 3.0]),
    }
    engine.wage = {
        "Low": np.array([1.0, 1.2, 1.5, 1.6]),
        "Medium": np.array([2.0, 2.5, 2.8, 3.0]),
        "High": np.array([5.0, 5.5, 6.0, 6.5]),
    }

    result = engine.analyze_transmission(top_k=4)

    wage_bilateral_df = result["bilateral_wage_transfer_by_skill"]
    assert {"From_Region", "To_Region", "Skill", "Wage_Transfer"}.issubset(wage_bilateral_df.columns)
    assert "Gender" not in wage_bilateral_df.columns
    assert not wage_bilateral_df.empty



def test_build_country_sector_transfer_tables_outputs_four_metric_tables() -> None:
    """应能输出 demand-country -> producer-country -> sector 的四张转移表。"""
    engine = NetworkEngine("/tmp/not-used")
    engine.regions = ["R1", "R2"]
    engine.sectors = ["Dirty", "Clean"]
    engine.L = np.eye(4)
    engine.Y = np.array(
        [
            [2.0, 14.0],
            [1.0, 6.0],
            [8.0, 1.0],
            [5.0, 1.0],
        ]
    )
    engine.co2_emissions = np.array([5.0, 1.0, 2.0, 0.2])
    engine.employment = {
        "Low": np.array([4.0, 2.0, 1.0, 0.5]),
        "Medium": np.array([1.0, 1.5, 1.0, 0.8]),
        "High": np.array([0.2, 0.6, 0.8, 1.2]),
    }
    engine.hours = {
        "Low": np.array([8.0, 6.0, 2.0, 1.0]),
        "Medium": np.array([3.0, 4.0, 3.0, 2.0]),
        "High": np.array([1.0, 2.0, 2.0, 3.0]),
    }
    engine.wage = {
        "Low": np.array([1.0, 1.2, 1.5, 1.6]),
        "Medium": np.array([2.0, 2.5, 2.8, 3.0]),
        "High": np.array([5.0, 5.5, 6.0, 6.5]),
    }

    result = engine.build_country_sector_transfer_tables()

    assert {"carbon", "employment", "wage", "hours"}.issubset(result.keys())
    assert {"Demand_Country", "Producer_Country", "Sector", "Carbon_Transfer"}.issubset(result["carbon"].columns)
    assert {"Demand_Country", "Producer_Country", "Sector", "Employment_Transfer"}.issubset(result["employment"].columns)
    assert {"Demand_Country", "Producer_Country", "Sector", "Wage_Transfer"}.issubset(result["wage"].columns)
    assert {"Demand_Country", "Producer_Country", "Sector", "Hours_Transfer"}.issubset(result["hours"].columns)
    assert not result["carbon"].empty
    assert not result["employment"].empty
    assert not result["wage"].empty
    assert not result["hours"].empty

