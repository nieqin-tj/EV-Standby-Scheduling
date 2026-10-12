"""Deterministic station allocation policies used for rule diagnostics."""
from __future__ import annotations

import numpy as np


POLICY_NAMES = (
    "equal_share_by_request",
    "fcfs_strict",
    "fcfs_minimum_one_module",
)


def _validate(demand_kw: np.ndarray, budget_kw: float, arrival_rank: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    demand = np.maximum(np.asarray(demand_kw, dtype=float), 0.0)
    rank = np.asarray(arrival_rank)
    if demand.ndim != 1 or rank.shape != demand.shape:
        raise ValueError("demand and arrival_rank must be aligned one-dimensional arrays")
    budget = min(max(float(budget_kw), 0.0), float(demand.sum()))
    order = np.lexsort((np.arange(len(rank)), rank))
    return demand, budget, order


def _water_fill(demand: np.ndarray, budget: float) -> np.ndarray:
    if demand.sum() <= budget + 1e-12:
        return demand.copy()
    low, high = 0.0, float(demand.max(initial=0.0))
    for _ in range(60):
        level = (low + high) / 2.0
        if np.minimum(demand, level).sum() < budget:
            low = level
        else:
            high = level
    return np.minimum(demand, (low + high) / 2.0)


def allocate_policy(
    demand_kw: np.ndarray,
    budget_kw: float,
    arrival_rank: np.ndarray,
    module_power_kw: float,
    policy: str,
) -> np.ndarray:
    """Allocate a known diagnostic cabinet budget under one deterministic rule.

    ``budget_kw`` may be the observed cabinet total only for retrospective rule
    identification. It is not a deployment-time prediction input.
    """
    demand, budget, order = _validate(demand_kw, budget_kw, arrival_rank)
    if policy == "equal_share_by_request":
        return _water_fill(demand, budget)

    allocated = np.zeros_like(demand)
    if policy == "fcfs_minimum_one_module":
        guarantee = np.minimum(demand, max(float(module_power_kw), 0.0))
        guarantee_total = float(guarantee.sum())
        if guarantee_total >= budget:
            return _water_fill(guarantee, budget)
        allocated = guarantee.copy()
        budget -= guarantee_total
    elif policy != "fcfs_strict":
        raise ValueError(f"unknown allocation policy: {policy}")

    for position in order:
        addition = min(float(demand[position] - allocated[position]), budget)
        if addition > 0:
            allocated[position] += addition
            budget -= addition
        if budget <= 1e-12:
            break
    return allocated
