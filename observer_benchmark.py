import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from observer import MODES, ThermalObserver
from plant import simulate_interval

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
SEEDS = [41, 73, 109]
STEPS = 96


def metrics(frame):
    error = frame.Tj_estimate_degC.to_numpy() - frame.Tj_true_degC.to_numpy()
    return {'samples': len(error), 'Tj_RMSE_degC': float(np.sqrt(np.mean(error ** 2))),
            'Tj_MAE_degC': float(np.mean(np.abs(error))),
            'Tj_underestimate_p99_degC': float(np.quantile(np.maximum(-error, 0), 0.99, method='linear')),
            'peak_Tj_true_degC': float(frame.Tj_true_degC.max())}


def run_benchmark():
    OUT.mkdir(exist_ok=True)
    t = np.arange(STEPS)
    action = 100 * np.sin(2 * np.pi * t / 24)
    ambient = 25 + 8 * np.sin(2 * np.pi * t / 96)
    ripple = 0.25 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 24))
    rows = []
    for cooling in [1.0, 1.2]:
        states = [np.array([25.0, 25.0])]
        for step in range(STEPS - 1):
            states.append(simulate_interval(states[-1], action[step], ambient[step], ripple[step],
                                            cooling=cooling, protection=False)['state'])
        for bias in [0.0, -2.0]:
            for seed in SEEDS:
                observers = {mode: ThermalObserver(mode, states[0]) for mode in MODES}
                generators = {mode: np.random.default_rng(seed) for mode in MODES}
                for step, current in enumerate(states):
                    last_p = action[step - 1] if step else 0.0
                    last_ambient = ambient[step - 1] if step else ambient[0]
                    last_ripple = ripple[step - 1] if step else 0.4
                    for mode in MODES:
                        estimate, ripple_hat = observers[mode].update(
                            current, last_p, last_ambient, last_ripple, generators[mode], common_bias=bias)
                        rows.append({'cooling': cooling, 'image_bias_degC': bias, 'seed': seed,
                                     'mode': mode, 'step': step, 'elapsed_s': step * 900,
                                     'last_applied_p_kW': last_p, 'last_ambient_degC': last_ambient,
                                     'last_ripple_true': last_ripple, 'last_ripple_estimate': ripple_hat,
                                     'Tj_true_degC': current[0], 'Ts_true_degC': current[1],
                                     'Tj_estimate_degC': estimate[0], 'Ts_estimate_degC': estimate[1]})
    perstep = pd.DataFrame(rows)
    assert len(perstep) == 2 * 2 * len(SEEDS) * len(MODES) * STEPS
    assert np.isfinite(perstep.select_dtypes(include='number').to_numpy()).all()
    by_step = perstep.groupby(['cooling', 'step'])
    assert by_step[['Tj_true_degC', 'Ts_true_degC', 'last_applied_p_kW', 'last_ambient_degC', 'last_ripple_true']].nunique().eq(1).all().all()
    oracle = perstep.loc[perstep['mode'].eq('oracle')]
    assert np.array_equal(oracle.Tj_true_degC.to_numpy(), oracle.Tj_estimate_degC.to_numpy())
    for mode in ['scalar', 'waveform', 'oracle']:
        chosen = perstep.loc[perstep['mode'].eq(mode)].set_index(['cooling', 'seed', 'step', 'image_bias_degC'])
        assert np.array_equal(chosen.xs(0.0, level='image_bias_degC').Tj_estimate_degC,
                              chosen.xs(-2.0, level='image_bias_degC').Tj_estimate_degC)
    perstep.to_csv(OUT / 'observer_benchmark_perstep.csv', index=False, float_format='%.10f')
    tables = {}
    for name, keys in [('summary', ['cooling', 'image_bias_degC', 'mode']),
                       ('by_seed', ['cooling', 'image_bias_degC', 'mode', 'seed'])]:
        tables[name] = pd.DataFrame([{**dict(zip(keys, group)), **metrics(frame)}
                                    for group, frame in perstep.groupby(keys, sort=True)])
        tables[name].to_csv(OUT / f'observer_benchmark_{name}.csv', index=False, float_format='%.10f')
    manifest = {'claim_boundary': 'Illustrative synthetic sensor-model consistency under a shared deterministic excitation, not independent physical validation or feasible dispatch.',
                'interval_seconds': 900, 'observation_steps': STEPS, 'completed_intervals': STEPS - 1,
                'timing': 'Observe state at t=0,...,95. State at t>0 follows the completed interval t-1. No current or future action/ambient/ripple enters observer.update.',
                'power_kW': '100*sin(2*pi*t/24)', 'ambient_degC': '25+8*sin(2*pi*t/96)',
                'ripple': '0.25+0.45*(0.5+0.5*sin(2*pi*t/24))', 'initial_state_degC': [25, 25],
                'cooling': [1.0, 1.2], 'image_bias_degC': [0.0, -2.0], 'seeds': SEEDS,
                'modes': list(MODES), 'protection': False, 'SOC_model': 'Not used; identification excitation only.',
                'metric_definition': 'RMSE and MAE of current junction-temperature estimate; underestimate p99 is linear empirical 0.99 quantile of max(true-estimate,0). Summary pools 288 observations over three seeds per condition and mode; by_seed reports 96 each. No independence or population inference is claimed.',
                'quality_checks': {'shared_physical_trajectory_all_modes_seeds_biases': True,
                                   'oracle_exact': True, 'non_image_modes_invariant_to_image_bias': True,
                                   'all_numeric_outputs_finite': True},
                'source_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()
                                  for n in ['observer.py', 'plant.py', 'observer_benchmark.py']}}
    (OUT / 'observer_benchmark_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(tables['summary'].to_string(index=False))


if __name__ == '__main__':
    run_benchmark()
