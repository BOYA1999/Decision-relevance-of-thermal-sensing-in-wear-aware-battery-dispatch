import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from vendor.rainflow import extract_cycles

PARAMETERS = {
    "status": "Illustrative synthetic assumptions; not calibrated physical lifetimes",
    "units": {"power": "kW", "time": "s", "temperature": "degC", "thermal_capacity": "kJ/K", "thermal_resistance": "K/kW"},
    "Rjc": 6.0, "Rsa": 20.0, "Cj": 0.3, "Cs": 60.0,
    "converter_rating_kW": 100.0, "battery_capacity_kWh": 200.0,
    "loss_intercept_kW": 0.2, "loss_abs_coefficient": 0.009,
    "loss_quadratic_coefficient_per_kW": 0.00018,
    "loss_equation": "q=0.2+0.009*abs(p)+0.00018*p**2*(1+ripple**2)",
    "ripple_definition": "Dimensionless synthetic RMS loss multiplier; no resolved switching waveform",
    "cooling_definition": "Positive multiplier on Rsa; larger means weaker cooling",
    "interval_seconds": 900.0, "evaluation_dt_seconds": 1.0, "trip_temperature_degC": 115.0,
    "trip_rule": "At first sampled Tj>=115, set remaining interval p=0; retain 0.2 kW standby loss. Caller latches trip until episode day ends.",
    "battery_wear_equation": "sum(count*DoD**1.7/6000); SOC and DoD are fractions",
    "converter_wear_equation": "sum(count*(range_degC/40)**5/1e6*exp((cycle_mean_degC-80)/40))",
    "wear_interpretation": "Dimensionless assumed lifetime-consumption indices, not empirically calibrated failure probabilities",
    "rainflow": {"version": "3.2.0", "license": "MIT", "source": "https://pypi.org/project/rainflow/3.2.0/", "residuals": "Unclosed endpoint ranges contribute half cycles"},
    "cost_equation": "battery_CAPEX*battery_wear+converter_CAPEX*converter_wear, each exactly once",
}


def power_loss(p, ripple=0.0):
    return 0.2 + 0.009 * np.abs(p) + 0.00018 * np.asarray(p) ** 2 * (1 + ripple ** 2)


@lru_cache(maxsize=32)
def thermal_discretization(cooling=1.0, dt=1.0):
    r, s, c, d = 6.0, 20.0 * cooling, 0.3, 60.0
    if cooling <= 0 or dt <= 0:
        raise ValueError("cooling and dt must be positive")
    a = np.array([[-1 / (r * c), 1 / (r * c)], [1 / (r * d), -(1 / r + 1 / s) / d]])
    b = np.array([[1 / c, 0], [0, 1 / (s * d)]])
    f = expm(a * dt)
    return f, np.linalg.solve(a, (f - np.eye(2)) @ b)


@lru_cache(maxsize=32)
def _modal_grid(cooling, dt, steps):
    f, _ = thermal_discretization(cooling, dt)
    values, vectors = np.linalg.eig(f)
    return values[None, :] ** np.arange(steps + 1)[:, None], vectors, np.linalg.inv(vectors)


def _trajectory(state, p, ambient, ripple, cooling, dt, steps):
    q = float(power_loss(p, ripple))
    steady = np.array([ambient + q * (6 + 20 * cooling), ambient + q * 20 * cooling])
    powers, vectors, inverse = _modal_grid(cooling, dt, steps)
    return (powers * (inverse @ (np.asarray(state) - steady))) @ vectors.T + steady


def simulate_interval(state, p, Tamb, ripple=0.0, cooling=1.0, dt=1.0, protection=False):
    steps = round(900 / dt)
    if not np.isclose(steps * dt, 900):
        raise ValueError("dt must divide the 900 second interval")
    trace = _trajectory(state, p, Tamb, ripple, cooling, dt, steps)
    hit = np.flatnonzero(trace[:, 0] >= 115) if protection else np.array([], dtype=int)
    active_steps = int(hit[0]) if len(hit) else steps
    tripped = bool(len(hit))
    if tripped and active_steps < steps:
        trace[active_steps:] = _trajectory(trace[active_steps], 0.0, Tamb, ripple, cooling, dt, steps - active_steps)
    power = np.zeros(steps)
    power[:active_steps] = p
    return {
        "state": trace[-1].copy(), "time_s": np.arange(steps + 1) * dt,
        "Tj": trace[:, 0], "Ts": trace[:, 1], "peak_Tj": float(trace[:, 0].max()),
        "tripped": tripped, "active_seconds": active_steps * dt,
        "delivered_energy_kWh": float(power.sum() * dt / 3600), "power_kW": power,
    }


def step_interval(state, p, Tamb, ripple=0.0, cooling=1.0, dt=1.0):
    return simulate_interval(state, p, Tamb, ripple, cooling, dt, protection=False)


def rainflow_cycles(series):
    values = np.asarray(series, dtype=float).ravel()
    if len(values) == 2:
        values = np.append(values, values[-1])
    return [(r, m, n, i, j) for r, m, n, i, j in extract_cycles(values) if r > 0]


def battery_wear(soc_fraction_trace):
    return float(sum(n * r ** 1.7 / 6000 for r, _, n, _, _ in rainflow_cycles(soc_fraction_trace)))


def converter_wear(junction_temperature_trace):
    return float(sum(n * (r / 40) ** 5 / 1e6 * np.exp((m - 80) / 40) for r, m, n, _, _ in rainflow_cycles(junction_temperature_trace)))


def wear_cost(battery_index, converter_index, battery_capex, converter_capex):
    return float(battery_capex * battery_index + converter_capex * converter_index)


if __name__ == "__main__":
    destination = Path(__file__).with_name("plant_parameters.json")
    destination.write_text(json.dumps(PARAMETERS, indent=2), encoding="utf-8")
    print(destination)
