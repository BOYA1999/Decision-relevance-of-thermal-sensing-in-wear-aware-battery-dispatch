import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from observer import ThermalObserver
from plant import battery_wear, converter_wear, power_loss, simulate_interval

ROOT = Path(__file__).resolve().parent
MODES = ['economic', 'rainflow', 'scalar', 'waveform', 'infrared', 'full_mean', 'full', 'oracle']


def load_data():
    frame = pd.read_csv(ROOT / 'data/household4_hourly_2016_2017.csv')
    frame['date'] = frame.utc_timestamp.str[:10]
    return frame[frame.strict_usable_day].copy()


def prepare():
    frame = load_data()
    train = frame[frame.date < '2016-09-01']
    load_scale = 100 / train.load_kWh.quantile(.95)
    pv_scale = 100 / train.PV_kWh.quantile(.99)
    days = {date: np.repeat((g.load_kWh * load_scale - g.PV_kWh * pv_scale).to_numpy(), 4)
            for date, g in frame.groupby('date', sort=True)}
    dates = [d for d in days if d < '2017-01-01']
    residual = np.array([days[b] - days[a] for a, b in zip(dates[:-1], dates[1:])])
    labels = np.array(dates[1:])
    fit = residual[labels < '2016-09-01']
    calibration = residual[labels >= '2016-09-01']
    km = KMeans(n_clusters=3, n_init=10, random_state=20260906).fit(fit)
    p = np.bincount(km.labels_, minlength=3) / len(fit)
    q = np.bincount(km.predict(calibration), minlength=3) / len(calibration)
    distance = np.abs(km.cluster_centers_[:, None] - km.cluster_centers_[None, :]).mean(axis=2) / 100
    transport = linprog(distance.ravel(), A_eq=np.r_[np.kron(np.eye(3), np.ones((1, 3))),
                         np.tile(np.eye(3), (1, 3))], b_eq=np.r_[p, q], bounds=(0, None), method='highs')
    assert transport.success
    contract = {
        'load_scale': float(load_scale), 'pv_scale': float(pv_scale),
        'train_dates': [d for d in dates if d < '2016-09-01'],
        'calibration_dates': [d for d in dates if d >= '2016-09-01'],
        'residual_fit_count': len(fit), 'residual_calibration_count': len(calibration),
        'centers': km.cluster_centers_.tolist(), 'weights': p.tolist(),
        'calibration_weights': q.tolist(), 'distance': distance.tolist(), 'rho': float(transport.fun),
        'test_dates': [f'2017-{m:02d}-01' for m in range(1, 13)],
        'seeds': [41, 73], 'regimes': ['ordinary', 'heatwave_dr'],
        'mpc': {'energy_capacity': 200., 'power_rating': 100., 'thermal_limit': 110.,
                'margin': 3., 'battery_capex': 50000., 'temperature_weight': .02,
                'rainflow_tolerance': .0001, 'max_cuts': 40, 'dr_penalty': 2.},
        'horizon_intervals': 16, 'converter_capex': 30000.,
        'profiles': {'ambient': '22+8*sin(2*pi*(doy-80)/365)+4*sin(2*pi*(hour-9)/24)',
                     'ripple': '0.25+0.45*(0.5+0.5*sin(2*pi*hour/6+doy/20))',
                     'heatwave_shift_C': 10, 'DR_hours_UTC': [17, 18.5], 'DR_import_reduction': .4,
                     'buy_USD_kWh': {'hours_0_to_6': .12, 'hours_17_to_21': .32, 'other': .20},
                     'sell_USD_kWh': .04, 'cooling_multiplier': 1., 'initial_SOC': .5,
                     'terminal_SOC': .5, 'efficiency': .98},
        'status': 'Development configuration; freeze a hashed copy before main results',
    }
    (ROOT / 'config.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    print(json.dumps({k: contract[k] for k in ['rho', 'weights', 'calibration_weights',
                                             'residual_fit_count', 'residual_calibration_count',
                                             'load_scale', 'pv_scale']}, indent=2))


def episode(date, mode, regime, seed, output, overrides=None):
    from mpc import solve_mpc
    threadpool_limits(1)
    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    frame = load_data()
    g = frame[frame.date == date]
    assert len(g) == 24
    previous_date = frame.loc[frame.date < date, 'date'].max()
    previous = frame[frame.date == previous_date]
    load = np.repeat(g.load_kWh.to_numpy() * cfg['load_scale'], 4)
    pv = np.repeat(g.PV_kWh.to_numpy() * cfg['pv_scale'], 4)
    persistence = np.repeat((previous.load_kWh * cfg['load_scale'] - previous.PV_kWh * cfg['pv_scale']).to_numpy(), 4)
    net = load - pv
    hours = np.arange(96) / 4
    doy = pd.Timestamp(date).dayofyear
    ambient = 22 + 8 * np.sin(2 * np.pi * (doy - 80) / 365) + 4 * np.sin(2 * np.pi * (hours - 9) / 24)
    if regime == 'heatwave_dr':
        ambient += 10
    ripple = .25 + .45 * (.5 + .5 * np.sin(2 * np.pi * hours / 6 + doy / 20))
    buy = np.where(hours < 6, .12, np.where((hours >= 17) & (hours < 21), .32, .20))
    dr = (hours >= 17) & (hours < 18.5) & (regime == 'heatwave_dr')
    actual_limit = np.where(dr, .6 * np.maximum(net + .2, 0), 1e4)
    parameters = dict(cfg['mpc'])
    parameters['rho'] = cfg['rho']
    parameters['scenario_weights'] = cfg['weights']
    cooling, common_bias = 1., 0.
    if overrides:
        cooling = overrides.get('cooling', cooling)
        common_bias = overrides.get('ir_bias', common_bias)
        parameters.update({k: v for k, v in overrides.items() if k not in ['cooling', 'ir_bias']})
    observer = ThermalObserver(mode if mode not in ['economic', 'rainflow'] else 'scalar', [ambient[0] + 5.2, ambient[0] + 4])
    if mode == 'economic':
        parameters.update(battery_capex=0., temperature_weight=0., health_constraints=False)
    elif mode == 'rainflow':
        parameters.update(temperature_weight=0., health_constraints=False)
    else:
        parameters['health_constraints'] = True
    capacity = parameters['energy_capacity']
    rng = np.random.default_rng(seed)
    state = np.array([ambient[0] + 5.2, ambient[0] + 4.])
    energy, soc = capacity / 2, [.5]
    junction = [state[0]]
    rows, solves = [], []
    tripped, last_p, last_ambient, last_ripple = False, 0., ambient[0], .4
    started = time.perf_counter()
    for t in range(96):
        estimate, rhat = observer.update(state, last_p, last_ambient, last_ripple, rng, common_bias=common_bias)
        h = min(cfg['horizon_intervals'], 96 - t)
        adjustment = (net[t] - persistence[t]) * np.exp(-np.arange(h) / 8)
        scenarios = persistence[None, t:t+h] + adjustment + np.asarray(cfg['centers'])[:, t:t+h]
        scenarios[:, 0] = net[t]
        parameters['grid_limit'] = np.where(dr[t:t+h], .6 * np.maximum(persistence[t:t+h] + adjustment + .2, 0), 1e4)
        parameters['grid_limit'][0] = actual_limit[t]
        if tripped:
            result = {'power': 0., 'status': 'latched_trip', 'solve_seconds': 0., 'iterations': 0,
                      'rainflow_gap': 0., 'constraint_residuals': {}}
        else:
            result = solve_mpc(scenarios, ambient[t:t+h], buy[t:t+h], np.full(h, .04), energy,
                               estimate, rhat, np.asarray(soc), parameters, np.asarray(cfg['distance']))
        valid = result['status'] in ['optimal', 'cut_limit', 'latched_trip']
        action = float(result.get('power', 0.)) if valid else 0.
        action = np.clip(action, -(capacity * .9 - energy) / (.25 * .98), (energy - capacity * .1) * .98 / .25)
        event = simulate_interval(state, action, ambient[t], ripple[t], cooling=cooling, protection=True)
        p = event['power_kW']
        q = power_loss(p, ripple[t])
        grid = net[t] - p + q
        cost = float((buy[t] * np.maximum(grid, 0) - .04 * np.maximum(-grid, 0)).sum() / 3600)
        shortfall = float(np.maximum(grid - actual_limit[t], 0).sum() / 3600)
        energy += float(np.sum(np.maximum(-p, 0) * .98 - np.maximum(p, 0) / .98) / 3600)
        row = {'interval': t, 'hour_utc': hours[t], 'load_kW': load[t], 'PV_kW': pv[t],
               'ambient_C': ambient[t], 'ripple': ripple[t], 'ripple_estimate': rhat,
               'Tj_start_C': state[0], 'Tj_estimate_C': estimate[0], 'Ts_start_C': state[1],
               'Ts_estimate_C': estimate[1], 'Tj_peak_C': event['peak_Tj'], 'action_kW': action,
               'executed_mean_kW': float(p.mean()), 'energy_kWh': energy, 'operating_cost_USD': cost,
               'DR_shortfall_kWh': shortfall, 'grid_energy_kWh': float(grid.sum() / 3600),
               'loss_energy_kWh': float(q.sum() / 3600), 'trip': int(event['tripped']),
               'thermal_exceedance_seconds': int(np.sum(event['Tj'][1:] > 110)),
               'solver_status': result['status'], 'solver_seconds': result['solve_seconds'],
               'rainflow_gap_USD': result.get('rainflow_gap', np.nan), 'iterations': result['iterations']}
        rows.append(row)
        solves.append({k: result.get(k) for k in ['status', 'solver_status', 'message',
                         'constraint_residuals', 'simultaneous_charge_discharge_kW',
                         'nonanticipativity_error', 'objective_lower', 'objective_upper', 'objective_gap']})
        tripped = tripped or event['tripped']
        state = event['state']
        junction.extend(event['Tj'][1:])
        soc.append(energy / capacity)
        last_p, last_ambient, last_ripple = float(p.mean()), ambient[t], ripple[t]
    table = pd.DataFrame(rows)
    bw, cw = battery_wear(soc), converter_wear(junction)
    operating = table.operating_cost_USD.sum()
    dr_cost = table.DR_shortfall_kWh.sum() * parameters['dr_penalty']
    actual_battery_capex = 250 * capacity
    wear_cost = actual_battery_capex * bw + cfg['converter_capex'] * cw
    terminal_deficit = capacity / 2 - energy
    terminal_correction = max(terminal_deficit, 0) / .98 * buy[-1] - max(-terminal_deficit, 0) * .98 * .04
    summary = {'date': date, 'mode': mode, 'regime': regime, 'seed': seed,
               'overrides': overrides or {}, 'operating_cost_USD': float(operating),
               'DR_cost_USD': float(dr_cost), 'battery_wear': bw, 'converter_wear': cw,
               'wear_cost_USD': float(wear_cost), 'terminal_correction_USD': float(terminal_correction),
               'total_cost_USD': float(operating + dr_cost + wear_cost),
               'energy_adjusted_cost_USD': float(operating + dr_cost + wear_cost + terminal_correction),
               'terminal_SOC': float(energy / capacity), 'terminal_SOC_error': float(energy / capacity - .5),
               'trip': int(tripped), 'thermal_exceedance_hours': float(table.thermal_exceedance_seconds.sum() / 3600),
               'DR_shortfall_kWh': float(table.DR_shortfall_kWh.sum()),
               'Tj_RMSE_C': float(np.sqrt(np.mean((table.Tj_estimate_C - table.Tj_start_C)**2))),
               'Tj_peak_C': float(max(junction)), 'FEC': float(np.abs(np.diff(soc)).sum() / 2),
               'solver_p95_seconds': float(table.solver_seconds.quantile(.95)),
               'max_rainflow_gap_USD': float(table.rainflow_gap_USD.max()),
               'solver_status_counts': table.solver_status.value_counts().to_dict(),
               'numerical_gate': 'PASS' if table.solver_status.isin(['optimal', 'latched_trip']).all() else 'REVISE',
               'terminal_service_met': bool(abs(energy / capacity - .5) < 1e-6),
               'elapsed_seconds': time.perf_counter() - started}
    out = Path(output) / f'{date}_{regime}_{mode}_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'intervals.csv', index=False)
    np.savez_compressed(out / 'physical_traces.npz', Tj=np.asarray(junction), SOC=np.asarray(soc))
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'solver_checks.json').write_text(json.dumps(solves, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'pilot', 'main'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
        return
    cfg = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
    output = ROOT / 'runs' / args.command
    dates = ['2016-01-15'] if args.command == 'pilot' else cfg['test_dates']
    seeds = [41] if args.command == 'pilot' else cfg['seeds']
    modes = ['economic', 'scalar', 'full'] if args.command == 'pilot' else MODES
    jobs = [(d, m, r, s, str(output)) for d in dates for m in modes for r in cfg['regimes'] for s in seeds]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(episode, *job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            summaries.append(result)
            pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
            print(json.dumps({k: result[k] for k in ['date', 'mode', 'regime', 'seed', 'trip',
                                                   'energy_adjusted_cost_USD', 'max_rainflow_gap_USD',
                                                   'solver_status_counts', 'elapsed_seconds']}), flush=True)
    failures = sum(x['numerical_gate'] != 'PASS' for x in summaries)
    print(f'EXECUTED {len(summaries)} episodes; numerical_gate_failures={failures}', flush=True)


if __name__ == '__main__':
    main()
