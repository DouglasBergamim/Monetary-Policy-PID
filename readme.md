# Economic Control Simulation – PID Tuning via Optimisation

This repository contains a small, self-contained Python toolkit to
simulate a simplified macro-economic model (gap of inflation driven by
interest-rate decisions) and automatically tune a discrete PID
controller.

## Contents

| Path | Description |
|------|-------------|
| `controlador/PID_controller.py` | Discrete PID (incremental form) with saturation. |
| `controlador/StatePlant.py`     | Generic state-space plant with optional Tustin discretisation. |
| `controlador/Simulator.py`      | Defines the continuous model, converts it to discrete time, tunes the PID gains with Nelder–Mead (`scipy.optimize.minimize`) and plots the closed-loop response. |
| `requirements.txt`             | All runtime dependencies. |

## Quick start

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the simulator:

```bash
python controlador/Simulator.py
```

You should see optimisation progress in the terminal followed by two
plots: the inflation gap response and the corresponding interest-rate
action.

## Model summary

The continuous-time state-space matrices are (simplified example):

```math
A_c = \begin{bmatrix}
  0 & 1 & 0 \\
  0 & 0 & 1 \\
 -\tfrac{\tau_r\,\gamma + 1}{\tau_r} & -\tfrac{\gamma^2 \tau_r/4 - \omega_1^2 \tau_r + \gamma}{\tau_r} & -\tfrac{\gamma^2/4 - \omega_1^2}{\tau_r}
\end{bmatrix},\quad
B_c = \begin{bmatrix}0 \\ 0 \\ \dfrac{J_r}{\tau_r m}\end{bmatrix},\quad
C = \begin{bmatrix}1 & 0 & 0\end{bmatrix}
```

where \(\gamma,\;\omega_1,\;\tau_r,\;m,\;J_r\) are physical/economic
parameters taken from the literature.
The model is discretised with a fixed sample period `DT` using first-order
Euler integration in `Simulator.py`.

## Controller & cost function

* Controller: discrete PID with output saturation (±5 pp gap in policy
  rate).
* Cost: ITAE (integral of time-weighted absolute error) + penalisation
  of control effort + variance of the final 20 % of the response.

```
J = ITAE + λ_u · ∫u² dt + λ_var · Var[y]
```

Weights `λ_u` and `λ_var` can be tuned in `Simulator.py`.

## Tuning algorithm

`scipy.optimize.minimize` with the Nelder–Mead simplex method searches
for the gains inside user-defined bounds (see `bounds` variable). The
routine stops when successive gain updates and cost improvements fall
below the absolute tolerances `xatol`/`fatol`.

## Customising

* **Change the reference** `REF` or initial state `x0` directly in
  `Simulator.py`.
* **Add constraints**: edit the `bounds` list or modify the `cost`
  function.
* **Different plant**: replace `A_c`, `B_c`, `C` with your own matrices
  (dimensions must stay consistent).

## License & contact

Released under the MIT license. Feel free to open issues or pull requests
for improvements.
