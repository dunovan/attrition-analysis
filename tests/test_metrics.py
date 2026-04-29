import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


# --- Shared fixtures ---

@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "overtime": ["Yes", "No", "Yes", "No"],
        "monthly_income": [3000, 5000, 4000, 8000],
        "job_satisfaction": [1, 3, 1, 3],
        "attrition": ["Yes", "No", "Yes", "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_basic(basic_df):
    assert attrition_rate(basic_df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["Yes", "Yes"],
    })
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["No", "No"],
    })
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_columns(basic_df):
    result = attrition_by_department(basic_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(basic_df):
    # Sales: 2 employees, 1 leaver → 50%  |  HR: 2 employees, 1 leaver → 50%
    result = attrition_by_department(basic_df)
    sales_row = result[result["department"] == "Sales"].iloc[0]
    assert sales_row["employees"] == 2
    assert sales_row["leavers"] == 1
    assert sales_row["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["HR", "HR", "Sales", "Sales"],
        "attrition": ["No", "No", "Yes", "Yes"],
    })
    result = attrition_by_department(df)
    # Sales (100%) should appear before HR (0%)
    assert list(result["department"]) == ["Sales", "HR"]
    assert list(result["attrition_rate"]) == [100.0, 0.0]


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(basic_df):
    result = attrition_by_overtime(basic_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values(basic_df):
    # Overtime=Yes: employees 1,3 both left → 100%
    # Overtime=No:  employees 2,4 neither left → 0%
    result = attrition_by_overtime(basic_df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_columns(basic_df):
    result = average_income_by_attrition(basic_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(basic_df):
    # Leavers (Yes):  3000 + 4000 / 2 = 3500
    # Stayers (No):   5000 + 8000 / 2 = 6500
    result = average_income_by_attrition(basic_df)
    yes_row = result[result["attrition"] == "Yes"].iloc[0]
    no_row = result[result["attrition"] == "No"].iloc[0]
    assert yes_row["avg_monthly_income"] == 3500.0
    assert no_row["avg_monthly_income"] == 6500.0


# --- satisfaction_summary ---

def test_satisfaction_summary_columns(basic_df):
    result = satisfaction_summary(basic_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_is_per_group_not_share_of_leavers():
    # satisfaction=1: 2 employees, 2 leavers → correct rate = 100%
    # satisfaction=3: 2 employees, 0 leavers → correct rate = 0%
    # Buggy formula (leavers / total_leavers) would give 100% and 0% here too,
    # so use an asymmetric case: sat=1 has 1/2, sat=2 has 1/1
    df = pd.DataFrame({
        "employee_id": [1, 2, 3],
        "job_satisfaction": [1, 1, 2],
        "attrition": ["Yes", "No", "Yes"],
    })
    result = satisfaction_summary(df)
    sat1 = result[result["job_satisfaction"] == 1].iloc[0]
    sat2 = result[result["job_satisfaction"] == 2].iloc[0]
    # Correct: sat=1 → 1/2 = 50%,  sat=2 → 1/1 = 100%
    # Buggy:   sat=1 → 1/2 = 50%,  sat=2 → 1/2 = 50%  (divides by total leavers=2)
    assert sat1["attrition_rate"] == 50.0
    assert sat2["attrition_rate"] == 100.0


def test_satisfaction_summary_sorted_by_satisfaction(basic_df):
    result = satisfaction_summary(basic_df)
    scores = list(result["job_satisfaction"])
    assert scores == sorted(scores)
