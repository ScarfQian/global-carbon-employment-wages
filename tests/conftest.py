"""Small exact MRIO fixtures used by the model tests."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from carbon_labour import inequality, policy


@pytest.fixture(scope="session")
def inequality_module():
    return inequality


@pytest.fixture(scope="session")
def scenario_module():
    return policy


@pytest.fixture()
def fake_footprint():
    """构造 2 地区 x 2 部门的小型精确 MRIO 测试系统。"""
    regions = ["AA", "BB"]
    products = ["coal", "services"]
    row_index = pd.MultiIndex.from_product([regions, products])
    y_columns = pd.MultiIndex.from_tuples([("AA", "HH"), ("BB", "HH")])
    y_frame = pd.DataFrame(
        [[2.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]],
        index=row_index,
        columns=y_columns,
    )
    employment_categories = [
        "Employment people: Low-skilled male",
        "Employment people: Low-skilled female",
        "Employment people: Medium-skilled male",
        "Employment people: Medium-skilled female",
        "Employment people: High-skilled male",
        "Employment people: High-skilled female",
    ]
    hours_categories = [name.replace("people", "hours") for name in employment_categories]
    wage_categories = [
        "Taxes less subsidies on products purchased: Total",
        "Other net taxes on production",
        "Compensation of employees: Low-skilled",
        "Compensation of employees: Medium-skilled",
        "Compensation of employees: High-skilled",
        "Operating surplus: Consumption of fixed capital",
        "Operating surplus: Rents on land",
        "Operating surplus: Royalties on resources",
        "Operating surplus: Remaining net operating surplus",
    ]
    employment = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0, 3.0],
    ])
    hours = employment * 2.0
    wage = np.zeros((9, 4), dtype=float)
    wage[0:2, :] = 100.0  # broad residual，不得进入技能工资。
    wage[2, :] = 1.0
    wage[3, :] = 2.0
    wage[4, :] = 3.0
    wage[5:, :] = 100.0
    return SimpleNamespace(
        mrio=SimpleNamespace(Y=y_frame),
        L=np.eye(4),
        Y=y_frame.to_numpy(dtype=float),
        x=np.ones(4),
        regions=regions,
        products=products,
        n_products=2,
        co2_emissions=np.array([10.0, 1.0, 10.0, 1.0]),
        employment={"data": employment, "categories": employment_categories},
        hours={"data": hours, "categories": hours_categories},
        wage={"data": wage, "categories": wage_categories},
    )

