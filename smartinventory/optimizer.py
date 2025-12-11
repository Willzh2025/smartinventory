"""
Optimization models for inventory planning.
Supports both Gurobi (preferred) and SciPy (fallback).
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

# Try to import Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

# SciPy fallback
try:
    from scipy.optimize import minimize, linprog
    from scipy.optimize import LinearConstraint, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("Neither Gurobi nor SciPy available. Optimization will fail.")


def solve_model_a_mip(
    sku_data: 'pd.DataFrame',
    budget: Optional[float] = None,
    capacity: Optional[float] = None
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, bool]]:
    """
    Model A: Cost Minimization MIP
    
    Decision variables:
    - Q_i >= 0: order quantity for SKU i
    - y_i in {0,1}: binary variable indicating if SKU i is ordered
    - s_i >= 0: shortage amount for SKU i
    
    Objective: Minimize total cost
    min sum_i [c_i * Q_i + F_i * y_i + h_i * (Q_i/2) + p_i * s_i]
    
    Constraints:
    1. s_i >= D_i - Q_i (shortage definition)
    2. Q_i <= M * y_i (fixed cost trigger)
    3. sum_i (c_i * Q_i + F_i * y_i) <= Budget (optional)
    4. sum_i (v_i * Q_i) <= Capacity (optional)
    
    Args:
        sku_data: DataFrame with columns: sku, unit_cost, holding_cost, 
                  stockout_penalty, volume, fixed_order_cost, demand_forecast
        budget: Optional budget constraint
        capacity: Optional storage capacity constraint
        
    Returns:
        Tuple of (order_quantities dict, shortages dict, total_cost, binding_constraints dict)
    """
    n_skus = len(sku_data)
    sku_list = sku_data['sku'].tolist()
    
    # Extract parameters
    c = sku_data['unit_cost'].values  # unit cost
    F = sku_data['fixed_order_cost'].values  # fixed order cost
    h = sku_data['holding_cost'].values  # holding cost per unit
    p = sku_data['stockout_penalty'].values  # stockout penalty
    v = sku_data['volume'].values  # volume per unit
    D = sku_data['demand_forecast'].values  # demand forecast
    
    # Big M value (large enough to not constrain Q_i)
    M = max(D) * 2 if len(D) > 0 else 10000
    
    if GUROBI_AVAILABLE:
        return _solve_model_a_gurobi(
            n_skus, sku_list, c, F, h, p, v, D, M, budget, capacity
        )
    elif SCIPY_AVAILABLE:
        return _solve_model_a_scipy(
            n_skus, sku_list, c, F, h, p, v, D, M, budget, capacity
        )
    else:
        raise RuntimeError("No optimization solver available. Install Gurobi or SciPy.")


def _solve_model_a_gurobi(
    n_skus: int,
    sku_list: List[str],
    c: np.ndarray,
    F: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    D: np.ndarray,
    M: float,
    budget: Optional[float],
    capacity: Optional[float]
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, bool]]:
    """Solve Model A using Gurobi."""
    model = gp.Model("InventoryOptimization")
    model.setParam('OutputFlag', 0)  # Suppress output
    
    # Decision variables
    Q = model.addVars(n_skus, lb=0.0, name="Q")  # Order quantities
    y = model.addVars(n_skus, vtype=GRB.BINARY, name="y")  # Binary order indicators
    s = model.addVars(n_skus, lb=0.0, name="s")  # Shortages
    
    # Objective: minimize total cost
    # Cost = purchasing + fixed + holding + shortage
    obj = gp.quicksum(
        c[i] * Q[i] + F[i] * y[i] + h[i] * (Q[i] / 2) + p[i] * s[i]
        for i in range(n_skus)
    )
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # 1. Shortage definition: s_i >= D_i - Q_i
    for i in range(n_skus):
        model.addConstr(s[i] >= D[i] - Q[i], name=f"shortage_{i}")
    
    # 2. Fixed cost trigger: Q_i <= M * y_i
    for i in range(n_skus):
        model.addConstr(Q[i] <= M * y[i], name=f"fixed_cost_{i}")
    
    # 3. Budget constraint (optional)
    if budget is not None:
        model.addConstr(
            gp.quicksum(c[i] * Q[i] + F[i] * y[i] for i in range(n_skus)) <= budget,
            name="budget"
        )
    
    # 4. Storage capacity (optional)
    if capacity is not None:
        model.addConstr(
            gp.quicksum(v[i] * Q[i] for i in range(n_skus)) <= capacity,
            name="capacity"
        )
    
    # Optimize
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")
    
    # Extract results
    order_quantities = {sku_list[i]: Q[i].X for i in range(n_skus)}
    shortages = {sku_list[i]: max(0, D[i] - Q[i].X) for i in range(n_skus)}
    total_cost = model.ObjVal
    
    # Check binding constraints
    binding_constraints = {
        'budget': False,
        'capacity': False
    }
    if budget is not None:
        budget_used = sum(c[i] * Q[i].X + F[i] * y[i].X for i in range(n_skus))
        binding_constraints['budget'] = abs(budget_used - budget) < 1e-6
    if capacity is not None:
        capacity_used = sum(v[i] * Q[i].X for i in range(n_skus))
        binding_constraints['capacity'] = abs(capacity_used - capacity) < 1e-6
    
    return order_quantities, shortages, total_cost, binding_constraints


def _solve_model_a_scipy(
    n_skus: int,
    sku_list: List[str],
    c: np.ndarray,
    F: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    D: np.ndarray,
    M: float,
    budget: Optional[float],
    capacity: Optional[float]
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, bool]]:
    """
    Solve Model A using SciPy (simplified - treats y_i as continuous in [0,1]).
    Note: This is an approximation since SciPy doesn't handle binary variables well.
    """
    # For SciPy, we'll use a continuous relaxation approach
    # Variables: [Q_0, ..., Q_{n-1}, y_0, ..., y_{n-1}, s_0, ..., s_{n-1}]
    n_vars = 3 * n_skus
    
    # Objective coefficients
    # Cost = c*Q + F*y + h*(Q/2) + p*s
    obj_coeff = np.zeros(n_vars)
    for i in range(n_skus):
        obj_coeff[i] = c[i] + h[i] / 2  # Q_i coefficient
        obj_coeff[n_skus + i] = F[i]  # y_i coefficient
        obj_coeff[2 * n_skus + i] = p[i]  # s_i coefficient
    
    # Bounds: Q >= 0, y in [0,1], s >= 0
    bounds = []
    for i in range(n_skus):
        bounds.append((0, None))  # Q_i
    for i in range(n_skus):
        bounds.append((0, 1))  # y_i (relaxed)
    for i in range(n_skus):
        bounds.append((0, None))  # s_i
    
    # Constraints
    constraints = []
    
    # 1. Shortage: s_i >= D_i - Q_i  =>  s_i + Q_i >= D_i
    for i in range(n_skus):
        A = np.zeros(n_vars)
        A[i] = 1  # Q_i
        A[2 * n_skus + i] = 1  # s_i
        constraints.append(LinearConstraint(A, lb=D[i], ub=np.inf))
    
    # 2. Fixed cost: Q_i <= M * y_i  =>  Q_i - M * y_i <= 0
    for i in range(n_skus):
        A = np.zeros(n_vars)
        A[i] = 1  # Q_i
        A[n_skus + i] = -M  # -M * y_i
        constraints.append(LinearConstraint(A, lb=-np.inf, ub=0))
    
    # 3. Budget constraint
    if budget is not None:
        A = np.zeros(n_vars)
        for i in range(n_skus):
            A[i] = c[i]  # c_i * Q_i
            A[n_skus + i] = F[i]  # F_i * y_i
        constraints.append(LinearConstraint(A, lb=-np.inf, ub=budget))
    
    # 4. Capacity constraint
    if capacity is not None:
        A = np.zeros(n_vars)
        for i in range(n_skus):
            A[i] = v[i]  # v_i * Q_i
        constraints.append(LinearConstraint(A, lb=-np.inf, ub=capacity))
    
    # Solve
    result = linprog(
        obj_coeff,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=bounds,
        constraints=constraints,
        method='highs'
    )
    
    if not result.success:
        raise RuntimeError(f"SciPy optimization failed: {result.message}")
    
    # Extract results
    x = result.x
    order_quantities = {sku_list[i]: max(0, x[i]) for i in range(n_skus)}
    shortages = {sku_list[i]: max(0, D[i] - x[i]) for i in range(n_skus)}
    total_cost = result.fun
    
    # Check binding constraints
    binding_constraints = {
        'budget': False,
        'capacity': False
    }
    if budget is not None:
        budget_used = sum(c[i] * x[i] + F[i] * x[n_skus + i] for i in range(n_skus))
        binding_constraints['budget'] = abs(budget_used - budget) < 1e-3
    if capacity is not None:
        capacity_used = sum(v[i] * x[i] for i in range(n_skus))
        binding_constraints['capacity'] = abs(capacity_used - capacity) < 1e-3
    
    return order_quantities, shortages, total_cost, binding_constraints


def solve_model_b_service_level(
    sku_data: 'pd.DataFrame',
    budget: Optional[float] = None,
    shortage_weight_multiplier: float = 10.0
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, bool]]:
    """
    Model B: Service-Level Feasibility Model
    
    Simplified model that minimizes weighted shortage under budget constraint.
    No binary variables - continuous optimization only.
    
    Decision variables:
    - Q_i >= 0: order quantity for SKU i
    - s_i >= 0: shortage amount for SKU i
    
    Objective: Minimize weighted shortage
    min sum_i [p_i * shortage_weight_multiplier * s_i]
    
    Constraints:
    1. s_i >= D_i - Q_i (shortage definition)
    2. sum_i (c_i * Q_i) <= Budget (optional)
    
    Args:
        sku_data: DataFrame with SKU parameters and demand forecasts
        budget: Optional budget constraint
        shortage_weight_multiplier: Multiplier for shortage penalty (default: 10.0)
        
    Returns:
        Tuple of (order_quantities dict, shortages dict, total_cost, binding_constraints dict)
    """
    n_skus = len(sku_data)
    sku_list = sku_data['sku'].tolist()
    
    # Extract parameters
    c = sku_data['unit_cost'].values
    h = sku_data['holding_cost'].values
    p = sku_data['stockout_penalty'].values * shortage_weight_multiplier
    D = sku_data['demand_forecast'].values
    
    if GUROBI_AVAILABLE:
        return _solve_model_b_gurobi(n_skus, sku_list, c, h, p, D, budget)
    elif SCIPY_AVAILABLE:
        return _solve_model_b_scipy(n_skus, sku_list, c, h, p, D, budget)
    else:
        raise RuntimeError("No optimization solver available. Install Gurobi or SciPy.")


def _solve_model_b_gurobi(
    n_skus: int,
    sku_list: List[str],
    c: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    D: np.ndarray,
    budget: Optional[float]
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, bool]]:
    """Solve Model B using Gurobi."""
    model = gp.Model("ServiceLevelOptimization")
    model.setParam('OutputFlag', 0)
    
    # Decision variables (no binary variables)
    Q = model.addVars(n_skus, lb=0.0, name="Q")
    s = model.addVars(n_skus, lb=0.0, name="s")
    
    # Objective: minimize weighted shortage
    obj = gp.quicksum(p[i] * s[i] for i in range(n_skus))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraints
    # 1. Shortage definition
    for i in range(n_skus):
        model.addConstr(s[i] >= D[i] - Q[i], name=f"shortage_{i}")
    
    # 2. Budget constraint (optional)
    if budget is not None:
        model.addConstr(
            gp.quicksum(c[i] * Q[i] for i in range(n_skus)) <= budget,
            name="budget"
        )
    
    # Optimize
    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")
    
    # Extract results
    order_quantities = {sku_list[i]: Q[i].X for i in range(n_skus)}
    shortages = {sku_list[i]: max(0, D[i] - Q[i].X) for i in range(n_skus)}
    
    # Calculate total cost (for reporting)
    total_cost = sum(
        c[i] * Q[i].X + h[i] * (Q[i].X / 2) + (p[i] / 10.0) * s[i].X
        for i in range(n_skus)
    )
    
    # Check binding constraints
    binding_constraints = {'budget': False}
    if budget is not None:
        budget_used = sum(c[i] * Q[i].X for i in range(n_skus))
        binding_constraints['budget'] = abs(budget_used - budget) < 1e-6
    
    return order_quantities, shortages, total_cost, binding_constraints


def _solve_model_b_scipy(
    n_skus: int,
    sku_list: List[str],
    c: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    D: np.ndarray,
    budget: Optional[float]
) -> Tuple[Dict[str, float], Dict[str, float], float, Dict[str, bool]]:
    """Solve Model B using SciPy."""
    # Variables: [Q_0, ..., Q_{n-1}, s_0, ..., s_{n-1}]
    n_vars = 2 * n_skus
    
    # Objective: minimize weighted shortage
    obj_coeff = np.zeros(n_vars)
    for i in range(n_skus):
        obj_coeff[n_skus + i] = p[i]  # s_i coefficient
    
    # Bounds: Q >= 0, s >= 0
    bounds = [(0, None)] * n_vars
    
    # Constraints
    constraints = []
    
    # 1. Shortage: s_i >= D_i - Q_i
    for i in range(n_skus):
        A = np.zeros(n_vars)
        A[i] = 1  # Q_i
        A[n_skus + i] = 1  # s_i
        constraints.append(LinearConstraint(A, lb=D[i], ub=np.inf))
    
    # 2. Budget constraint
    if budget is not None:
        A = np.zeros(n_vars)
        for i in range(n_skus):
            A[i] = c[i]  # c_i * Q_i
        constraints.append(LinearConstraint(A, lb=-np.inf, ub=budget))
    
    # Solve
    result = linprog(
        obj_coeff,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=bounds,
        constraints=constraints,
        method='highs'
    )
    
    if not result.success:
        raise RuntimeError(f"SciPy optimization failed: {result.message}")
    
    # Extract results
    x = result.x
    order_quantities = {sku_list[i]: max(0, x[i]) for i in range(n_skus)}
    shortages = {sku_list[i]: max(0, D[i] - x[i]) for i in range(n_skus)}
    
    # Calculate total cost
    total_cost = sum(
        c[i] * x[i] + h[i] * (x[i] / 2) + (p[i] / 10.0) * x[n_skus + i]
        for i in range(n_skus)
    )
    
    # Check binding constraints
    binding_constraints = {'budget': False}
    if budget is not None:
        budget_used = sum(c[i] * x[i] for i in range(n_skus))
        binding_constraints['budget'] = abs(budget_used - budget) < 1e-3
    
    return order_quantities, shortages, total_cost, binding_constraints

