import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from study import ROOT, episode


if __name__ == '__main__':
    cfg = json.loads((ROOT / 'config.json').read_text())
    dates = ['2017-01-01', '2017-04-01', '2017-07-01', '2017-10-01']
    modes = ['scalar', 'waveform', 'full_mean', 'full']
    settings = {
        'battery_100': {'energy_capacity': 100., 'battery_capex': 25000.},
        'battery_400': {'energy_capacity': 400., 'battery_capex': 100000.},
        'cooling_120': {'cooling': 1.2},
        'IR_bias_minus_2': {'ir_bias': -2.},
        'radius_zero': {'rho': 0.},
        'radius_double': {'rho': 2 * cfg['rho']},
        'margin_zero': {'margin': 0.},
        'margin_six': {'margin': 6.},
    }
    contract = {'status': 'Frozen before executing these robustness runs', 'dates': dates,
                'modes': modes, 'seeds': cfg['seeds'], 'regime': 'heatwave_dr', 'settings': settings,
                'precision_check': {'dates': dates, 'modes': modes, 'regimes': cfg['regimes'],
                                    'seed': 41, 'rainflow_tolerance': 1e-6, 'max_cuts': 80},
                'interpretation': 'Prespecified parameter cells; no external site validation'}
    (ROOT / 'robustness_contract.json').write_text(json.dumps(contract, indent=2))
    jobs = [(name, d, m, 'heatwave_dr', s, overrides) for name, overrides in settings.items()
            for d in dates for m in modes for s in cfg['seeds']]
    jobs += [('precision', d, m, r, 41, {'rainflow_tolerance': 1e-6, 'max_cuts': 80})
             for d in dates for m in modes for r in cfg['regimes']]
    jobs += [('oracle_cut_recovery', '2017-05-01', 'oracle', 'ordinary', s, {'max_cuts': 160})
             for s in cfg['seeds']]
    output = ROOT / 'runs/robustness'
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(episode, d, m, r, s, str(output / name), override): name
                   for name, d, m, r, s, override in jobs}
        for future in as_completed(pending):
            item = future.result()
            item['setting'] = pending[future]
            summaries.append(item)
            pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
            if len(summaries) % 16 == 0:
                print(f'{len(summaries)}/{len(jobs)} episodes; last={item["setting"]}; gate={item["numerical_gate"]}', flush=True)
    print('ROBUSTNESS_EXECUTED', len(summaries), 'GATE_FAILURES', sum(s['numerical_gate'] != 'PASS' for s in summaries), flush=True)
