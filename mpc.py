from time import perf_counter

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from plant import thermal_discretization
from rainflow_convex import rainflow_value_gradient


def _matrix(rows, width):
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        for j, value in row.items():
            if value:
                rr.append(i)
                cc.append(j)
                vv.append(value)
    return coo_matrix((vv, (rr, cc)), shape=(len(rows), width)).tocsr()


def _worst_expectation(costs, distances, radius, weights):
    count = len(costs)
    if count == 1 or radius == 0:
        return float(weights @ costs)
    rows = [{0: -distances[i, j], i + 1: -1} for i in range(count) for j in range(count)]
    result = linprog(np.r_[radius, weights], A_ub=_matrix(rows, count + 1),
                     b_ub=np.tile(-costs, count), bounds=[(0, None)] + [(None, None)] * count,
                     method="highs")
    return float(result.fun) if result.success else None


def solve_mpc(net_scenarios, ambient_forecast, buy_price, sell_price, initial_energy,
              state_estimate, ripple_estimate, soc_prefix, config, distance=None):
    started = perf_counter()
    net = np.atleast_2d(np.asarray(net_scenarios, dtype=float))
    scenarios, horizon = net.shape
    weights = np.asarray(config.get("scenario_weights", np.full(scenarios, 1 / scenarios)), dtype=float)
    if weights.shape != (scenarios,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("scenario_weights must be nonnegative, have length S, and sum to one")
    ambient = np.broadcast_to(np.asarray(ambient_forecast, dtype=float), net.shape)
    ripple = np.broadcast_to(np.asarray(ripple_estimate, dtype=float), net.shape)
    buy, sell = np.asarray(buy_price, dtype=float), np.asarray(sell_price, dtype=float)
    if buy.shape != (horizon,) or sell.shape != (horizon,) or np.any(sell < 0) or np.any(buy < sell):
        raise ValueError("Prices must have shape H and satisfy buy >= sell >= 0")
    capacity, rating = config.get("energy_capacity", 200.0), config.get("power_rating", 100.0)
    prefix = np.asarray(soc_prefix, dtype=float)
    if not len(prefix) or not np.isclose(prefix[-1], initial_energy / capacity, atol=1e-7):
        raise ValueError("soc_prefix must end at the current energy/capacity")
    health = config.get("health_constraints", True)
    capex = config.get("battery_capex", 50000.0)
    temperature_weight = config.get("temperature_weight", 0.02 if health else 0.0)
    radius = float(config.get("rho", 0.1))
    if radius < 0 or capex < 0 or temperature_weight < 0:
        raise ValueError("Radius and cost weights must be nonnegative")
    distances = (np.mean(np.abs(net[:, None, :] - net[None, :, :]), axis=2) / rating
                 if distance is None else np.asarray(distance, dtype=float))
    if distances.shape != (scenarios, scenarios) or np.any(distances < 0) or not np.allclose(np.diag(distances), 0):
        raise ValueError("distance must be a nonnegative S by S matrix with zero diagonal")
    dr_limit = np.broadcast_to(np.asarray(config.get("grid_limit", 1e4)), net.shape)
    dr_penalty = config.get("dr_penalty", 2.0)
    thermal_limit = config.get("thermal_limit", 105.0) - config.get("margin", 0.0)
    f, g = thermal_discretization(cooling=1.0, dt=900.0)
    indices, bounds = [], []
    for scene in range(scenarios):
        ix = {}
        for name in ["ch", "dis", "gimp", "gexp", "q", "excess", "dr_violation", "energy", "Tj", "Ts", "theta"]:
            size = horizon + 1 if name in ["energy", "Tj", "Ts"] else (1 if name == "theta" else horizon)
            ix[name] = np.arange(len(bounds), len(bounds) + size)
            bound = (None, None) if name in ["Tj", "Ts"] else (0, None)
            if name in ["ch", "dis"]:
                bound = (0, rating)
            if name == "energy":
                bound = (0.1 * capacity, 0.9 * capacity)
            if name in ["gimp", "gexp"]:
                bound = (0, config.get("max_grid_import" if name == "gimp" else "max_grid_export", 1e4))
            bounds.extend([bound] * size)
        indices.append(ix)
    if scenarios > 1:
        dual_lambda = len(bounds)
        dual_s = np.arange(dual_lambda + 1, dual_lambda + 1 + scenarios)
        bounds.extend([(0, None)] + [(None, None)] * scenarios)
    width = len(bounds)
    regularizer = np.zeros(width)
    equality, eq_rhs, inequality, ineq_rhs, cost_vectors = [], [], [], [], []
    def eq(row, rhs):
        equality.append(row)
        eq_rhs.append(rhs)
    def ub(row, rhs):
        inequality.append(row)
        ineq_rhs.append(rhs)
    for scene, ix in enumerate(indices):
        eq({ix["energy"][0]: 1}, initial_energy)
        eq({ix["energy"][-1]: 1}, 0.5 * capacity)
        eq({ix["Tj"][0]: 1}, float(state_estimate[0]))
        eq({ix["Ts"][0]: 1}, float(state_estimate[1]))
        costs = np.zeros(width)
        costs[ix["theta"][0]] = 1 if capex else 0
        for k in range(horizon):
            ch, dis, imp, exp, loss = (ix[key][k] for key in ["ch", "dis", "gimp", "gexp", "q"])
            eq({ix["energy"][k + 1]: 1, ix["energy"][k]: -1, ch: -0.25 * 0.98, dis: 0.25 / 0.98}, 0)
            eq({imp: 1, exp: -1, ch: -1, dis: 1, loss: -1}, net[scene, k])
            ub({ch: 1, dis: 1}, rating)
            knots = np.linspace(0, rating, 5)
            quadratic = 0.00018 * (1 + ripple[scene, k] ** 2)
            for left, right in zip(knots[:-1], knots[1:]):
                slope = 0.009 + quadratic * (left + right)
                ub({ch: slope, dis: slope, loss: -1}, -0.2 + quadratic * left * right)
            for row, name in enumerate(["Tj", "Ts"]):
                eq({ix[name][k + 1]: 1, ix["Tj"][k]: -f[row, 0], ix["Ts"][k]: -f[row, 1], loss: -g[row, 0]}, g[row, 1] * ambient[scene, k])
            ub({ix["Tj"][k + 1]: 1, ix["excess"][k]: -1}, 80)
            ub({imp: 1, ix["dr_violation"][k]: -1}, dr_limit[scene, k])
            if health:
                ub({ix["Tj"][k + 1]: 1}, thermal_limit)
                ub({ix["Ts"][k]: 1, loss: 6.0}, thermal_limit)
            costs[imp], costs[exp] = 0.25 * buy[k], -0.25 * sell[k]
            costs[ix["excess"][k]] = 0.25 * temperature_weight
            costs[ix["dr_violation"][k]] = 0.25 * dr_penalty
            regularizer[ch] = regularizer[dis] = 1e-7
        cost_vectors.append(costs)
        if scene:
            for name in ["ch", "dis"]:
                eq({ix[name][0]: 1, indices[0][name][0]: -1}, 0)
    objective = cost_vectors[0].copy() if scenarios == 1 else np.zeros(width)
    if scenarios > 1:
        objective[dual_lambda] = radius
        objective[dual_s] = weights
        for i in range(scenarios):
            for j in range(scenarios):
                row = {index: coefficient for index, coefficient in enumerate(cost_vectors[j]) if coefficient}
                row.update({dual_lambda: -distances[i, j], dual_s[i]: -1})
                ub(row, 0)
    objective += regularizer
    a_eq, b_eq = _matrix(equality, width), np.asarray(eq_rhs)
    options = {"primal_feasibility_tolerance": 1e-7, "dual_feasibility_tolerance": 1e-7}
    if "time_limit" in config:
        options["time_limit"] = config["time_limit"]
    max_cuts, tolerance = config.get("max_cuts", 12), config.get("rainflow_tolerance", 0.005)
    for iteration in range(1, max_cuts + 1):
        a_ub, b_ub = _matrix(inequality, width), np.asarray(ineq_rhs)
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                         bounds=bounds, method="highs", options=options)
        if not result.success:
            return {"power": None, "status": "solver_failed", "solver_status": int(result.status),
                    "message": result.message, "solve_seconds": perf_counter() - started,
                    "iterations": iteration, "rainflow_gap": None, "plan": None, "energy_trace": None}
        actual_wear, cuts, gaps = [], [], []
        for ix in indices:
            energy = result.x[ix["energy"]]
            value, gradient = rainflow_value_gradient(np.r_[prefix[:-1], energy / capacity])
            value, tail = capex * value, capex * gradient[len(prefix) - 1:] / capacity
            gap = max(0.0, value - result.x[ix["theta"][0]])
            gaps.append(gap)
            actual_wear.append(value)
            row = {index: slope for index, slope in zip(ix["energy"], tail) if slope}
            row[ix["theta"][0]] = -1
            cuts.append((row, float(tail @ energy - value)))
        gap = max(gaps)
        if gap <= tolerance or iteration == max_cuts:
            break
        for (row, rhs), scene_gap in zip(cuts, gaps):
            if scene_gap > tolerance:
                ub(row, rhs)
    x = result.x
    plan = {name: np.stack([x[ix[name]] for ix in indices]) for name in indices[0]}
    actual_costs = np.array([cost @ x - (x[ix["theta"][0]] if capex else 0) + wear
                             for cost, ix, wear in zip(cost_vectors, indices, actual_wear)])
    residuals = {"equality": float(np.max(np.abs(a_eq @ x - b_eq))),
                 "inequality": float(max(0, np.max(a_ub @ x - b_ub))),
                 "bounds": float(max([0] + [max(0 if lo is None else lo - x[i], 0 if hi is None else x[i] - hi) for i, (lo, hi) in enumerate(bounds)]))}
    simultaneous = float(np.minimum(plan["ch"], plan["dis"]).max())
    upper = _worst_expectation(actual_costs, distances, radius, weights)
    regularization_value = float(regularizer @ x)
    if upper is not None:
        upper += regularization_value
    status = "optimal" if gap <= tolerance else "cut_limit"
    if max(residuals.values()) > 2e-6 or simultaneous > 2e-6:
        status = "invalid_plan"
    return {"power": float(plan["dis"][0, 0] - plan["ch"][0, 0]), "status": status,
            "solver_status": int(result.status), "solve_seconds": perf_counter() - started,
            "iterations": iteration, "rainflow_gap": float(gap), "rainflow_gap_scenarios": gaps,
            "plan": plan, "energy_trace": plan["energy"], "cost_scenarios": actual_costs,
            "objective_lower": float(result.fun), "objective_upper": upper,
            "objective_gap": None if upper is None else float(upper - result.fun),
            "regularization_value": regularization_value,
            "constraint_residuals": residuals, "simultaneous_charge_discharge_kW": simultaneous,
            "nonanticipativity_error": float(max(np.ptp(plan["ch"][:, 0]), np.ptp(plan["dis"][:, 0])))}
