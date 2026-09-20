import inspect
import json
from pathlib import Path

import numpy as np

from observer import INITIAL_COVARIANCE, MODES, PIXEL_ALPHA, PIXEL_H, ThermalObserver
from plant import power_loss, thermal_discretization

checks = {}
state0 = np.array([30.0, 29.0])
truth = np.array([46.0, 31.0])
outputs, draws = {}, {}
for mode in MODES:
    rng = np.random.default_rng(2031)
    observer = ThermalObserver(mode, state0)
    outputs[mode] = observer.update(truth, 0, 28, 0.9, rng)
    draws[mode] = rng.normal(size=5)
    assert observer.state.shape == (2,) and observer.covariance.shape == (2, 2)
assert all(np.array_equal(draws['scalar'], v) for v in draws.values())
assert np.array_equal(outputs['scalar'][0], outputs['waveform'][0])
assert np.array_equal(outputs['infrared'][0], outputs['full'][0])
assert all(v[1] == 0.4 for v in outputs.values())
checks['same_seed_noise_pairing_and_consumption'] = True
assert np.array_equal(outputs['oracle'][0], truth)
assert np.isclose(PIXEL_ALPHA.min(), 0.02) and np.isclose(PIXEL_ALPHA.max(), 0.30)
assert np.linalg.matrix_rank(PIXEL_H) == 2
checks['oracle_exact_and_spatial_observation_rank'] = True

noise = np.random.default_rng(2031).normal(size=18)
image = PIXEL_H @ truth + 0.5 * noise[1:17]
h = PIXEL_H.mean(axis=0, keepdims=True)
gain = INITIAL_COVARIANCE @ h.T / (h @ INITIAL_COVARIANCE @ h.T + 0.25 / 16)
expected = state0 + gain.ravel() * (image.mean() - (h @ state0)[0])
assert np.allclose(outputs['full_mean'][0], expected, atol=1e-12)
expected_covariance = INITIAL_COVARIANCE - gain @ h @ INITIAL_COVARIANCE
actual = ThermalObserver('full_mean', state0)
actual.update(truth, 0, 28, 0.9, np.random.default_rng(2031))
assert np.allclose(actual.covariance, expected_covariance, atol=1e-12)
checks['full_mean_uses_same_16_pixels_and_variance_divided_by_16'] = True

for mode in ['scalar', 'waveform']:
    low = ThermalObserver(mode, state0).update([20, 31], 0, 28, 0.0, np.random.default_rng(42))[0]
    high = ThermalObserver(mode, state0).update([120, 31], 0, 28, 9.0, np.random.default_rng(42))[0]
    assert np.array_equal(low, high)
for mode in MODES:
    observer = ThermalObserver(mode, state0)
    a = observer.update(truth, 0, 28, 0, np.random.default_rng(8))[0]
    b = ThermalObserver(mode, state0).update(truth, 100, 45, 0, np.random.default_rng(8), first=True)[0]
    assert np.array_equal(a, b)
assert set(inspect.signature(ThermalObserver.update).parameters) == {
    'self', 'true_state', 'last_p', 'last_Tamb', 'last_ripple', 'rng', 'common_bias', 'first'}
checks['scalar_hidden_junction_invariance_and_first_step_no_prediction'] = True
checks['interface_has_only_current_sensor_truth_and_past_drivers'] = True

for mode in ['scalar', 'infrared']:
    a = ThermalObserver(mode, state0).update(truth, 60, 28, 0.1, np.random.default_rng(42))[1]
    b = ThermalObserver(mode, state0).update(truth, 60, 28, 0.9, np.random.default_rng(42))[1]
    assert a == b == 0.4
for mode in ['waveform', 'full', 'full_mean']:
    a = ThermalObserver(mode, state0).update(truth, 60, 28, 0.1, np.random.default_rng(42))[1]
    b = ThermalObserver(mode, state0).update(truth, 60, 28, 0.9, np.random.default_rng(42))[1]
    assert np.isclose(b - a, 0.8)
checks['past_ripple_access_and_no_current_gate'] = True

for mode in ['scalar', 'infrared', 'full_mean']:
    a = ThermalObserver(mode, state0).update(truth, 0, 28, 0.4, np.random.default_rng(10), common_bias=0)[0]
    b = ThermalObserver(mode, state0).update(truth, 0, 28, 0.4, np.random.default_rng(10), common_bias=2)[0]
    assert np.array_equal(a, b) if mode == 'scalar' else np.linalg.norm(a - b) > 0
checks['common_bias_applies_only_to_the_synthetic_image'] = True

minimum_eigenvalue = {}
for mode in MODES:
    observer = ThermalObserver(mode, state0)
    current = state0.copy()
    rng = np.random.default_rng(12)
    smallest = np.inf
    for step in range(160):
        last_p = 0 if step == 0 else 80 * np.sin(step / 9)
        last_ambient = 30 + 4 * np.sin(step / 40)
        last_ripple = 0.4 + 0.2 * np.sin(step / 12)
        if step:
            f, g = thermal_discretization(cooling=1.4, dt=900)
            current = f @ current + g @ np.array([power_loss(last_p, last_ripple), last_ambient])
        estimate, ripple = observer.update(current, last_p, last_ambient, last_ripple, rng)
        assert np.isfinite(estimate).all() and np.isfinite(ripple)
        assert np.isfinite(observer.covariance).all()
        assert np.allclose(observer.covariance, observer.covariance.T)
        eigenvalue = np.linalg.eigvalsh(observer.covariance).min()
        assert eigenvalue > 0
        smallest = min(smallest, float(eigenvalue))
    minimum_eigenvalue[mode] = smallest
checks['finite_positive_covariance_under_cooling_mismatch_160_steps'] = True
result = {'status': 'PASS', 'checks': checks, 'minimum_covariance_eigenvalue_by_mode': minimum_eigenvalue,
          'claim_boundary': 'Software and synthetic-observation checks only; no physical sensor calibration or empirical accuracy validation.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
