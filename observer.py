import numpy as np

from plant import PARAMETERS, power_loss, thermal_discretization

MODES = ('scalar', 'waveform', 'infrared', 'full', 'full_mean', 'oracle')
PROCESS_COVARIANCE = np.diag([1.0, 1.0])
INITIAL_COVARIANCE = np.diag([4.0, 4.0])
NOMINAL_RIPPLE = 0.4
SENSOR_SIGMA_C = 0.5
RIPPLE_SIGMA = 0.015
grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
weights = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * 0.65 ** 2))
PIXEL_ALPHA = 0.02 + 0.28 * (weights - weights.min()) / (weights.max() - weights.min())
PIXEL_H = np.column_stack([PIXEL_ALPHA.ravel(), 1 - PIXEL_ALPHA.ravel()])
SCALAR_H = np.array([[0.0, 1.0]])


class ThermalObserver:
    def __init__(self, mode, state0):
        if mode not in MODES:
            raise ValueError(f'Unknown observer mode: {mode}')
        self.mode = mode
        self.state = np.asarray(state0, dtype=float).copy()
        self.covariance = INITIAL_COVARIANCE.copy()
        self.first = True

    def update(self, true_state, last_p, last_Tamb, last_ripple, rng,
               common_bias=0.0, first=None):
        noise = rng.normal(size=18)
        first = self.first if first is None else first
        self.first = False
        ripple = NOMINAL_RIPPLE
        if abs(last_p) >= 1.0 and self.mode in ('waveform', 'full', 'full_mean', 'oracle'):
            ripple = float(last_ripple + (0.0 if self.mode == 'oracle' else RIPPLE_SIGMA * noise[17]))
        if self.mode == 'oracle':
            self.state = np.asarray(true_state, dtype=float).copy()
            self.covariance = np.eye(2) * 1e-12
            return self.state.copy(), ripple
        if not first:
            f, g = thermal_discretization(cooling=1.0, dt=PARAMETERS['interval_seconds'])
            self.state = f @ self.state + g @ np.array([power_loss(last_p, ripple), last_Tamb])
            self.covariance = f @ self.covariance @ f.T + PROCESS_COVARIANCE
        variance = SENSOR_SIGMA_C ** 2
        if self.mode in ('infrared', 'full', 'full_mean'):
            h = PIXEL_H
            measured = h @ np.asarray(true_state) + SENSOR_SIGMA_C * noise[1:17] + common_bias
            if self.mode == 'full_mean':
                h = h.mean(axis=0, keepdims=True)
                measured = np.array([measured.mean()])
                variance /= len(PIXEL_H)
        else:
            h = SCALAR_H
            measured = np.array([true_state[1] + SENSOR_SIGMA_C * noise[0]])
        r = np.eye(len(h)) * variance
        innovation_covariance = h @ self.covariance @ h.T + r
        gain = np.linalg.solve(innovation_covariance, h @ self.covariance).T
        self.state = self.state + gain @ (measured - h @ self.state)
        residual = np.eye(2) - gain @ h
        self.covariance = residual @ self.covariance @ residual.T + gain @ r @ gain.T
        self.covariance = (self.covariance + self.covariance.T) / 2
        return self.state.copy(), ripple
