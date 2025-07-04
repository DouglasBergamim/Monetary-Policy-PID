import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from StatePlant import StateSpacePlant
from PID_controller import DiscretePIDController

# -------------------- Modelo contínuo ------------------------
γ, ω1 = 4.37, 0.0093
τr, m, Jr = 11.6279, 640.36, -1.9186

A_c = np.array([[0, 1, 0],
                [0, 0, 1],
                [-(τr*γ+1)/τr,
                 -(γ**2*τr/4 - ω1**2*τr + γ)/τr,
                 -(γ**2/4 - ω1**2)/τr]])
B_c = np.array([[0], [0], [Jr/(τr*m)]])
C   = np.array([[1, 0, 0]])

REF   = 0.0                     # referência (gap de inflação)
x0    = np.array([1.0, 0, 0])   # choque inicial de 1 pp
DT    = 0.01                    # passo de integração (anos)
U_SAT = 5.0                     # saturação ±5 p.p. no gap de juros

# ------------ limites admissíveis dos ganhos ------------------
bounds = [(0, 1_000),   # Kp
          (0,   200),   # Ki
          (0, 1_000)]   # Kd


# ------------ custo: ITAE  + λu·energia + λσ·variância --------
def cost(gains, tf=20.0, lam_u=1.0, lam_var=10.0):
    # rejeita valores fora dos bounds
    if any(not (lo <= k <= hi) for k, (lo, hi) in zip(gains, bounds)):
        return 1e6

    pid = DiscretePIDController(*gains, dt=DT,
                                u_min=-U_SAT, u_max=U_SAT)

    n  = int(tf/DT)
    t  = np.arange(n)*DT
    y_hist = np.zeros(n)
    u_hist = np.zeros(n)

    x = x0.copy()
    for k in range(n):
        y = (C @ x).item()
        u = pid.step(REF - y)
        x = x + DT*(A_c @ x + B_c.flatten()*u)    # Euler
        y_hist[k] = y
        u_hist[k] = u

    itae   = np.trapz(np.abs(REF - y_hist)*t, t)
    u_eng  = np.trapz(u_hist**2, t)
    var_y  = np.var(y_hist[int(0.8*n):])          # variância final (20 %)
    return itae + lam_u*u_eng + lam_var*var_y


# ------------ optimização Nelder–Mead -------------------------
def optimise_pid(x0=(200., 10., 200.)):
    res = minimize(cost, x0=x0, method='Nelder-Mead',
                   options={'maxiter': 800, 'xatol': 1e-3, 'fatol': 1e-3})
    return tuple(res.x)


# ------------ simulação para plot -----------------------------
def simulate(gains, tf=20.0):
    pid   = DiscretePIDController(*gains, dt=DT,
                                  u_min=-U_SAT, u_max=U_SAT)
    n  = int(tf/DT)
    t  = np.arange(n)*DT
    y_hist = np.zeros(n)
    u_hist = np.zeros(n)

    x = x0.copy()
    for k in range(n):
        y = (C @ x).item()
        u = pid.step(REF - y)
        x = x + DT*(A_c @ x + B_c.flatten()*u)
        y_hist[k] = y
        u_hist[k] = u
    return t, y_hist, u_hist


# ------------------------- gráficos ---------------------------
def plot(t, y, u):
    plt.subplot(1, 2, 1)
    plt.plot(t, y, label='π gap')
    plt.axhline(REF, ls='--', c='r', label='ref')
    plt.xlabel('tempo (anos)'); plt.ylabel('saída'); plt.grid(); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t, u, color='tab:orange')
    plt.xlabel('tempo (anos)'); plt.ylabel('gap juros'); plt.grid()

    plt.tight_layout(); plt.show()


# ---------------------------- main ----------------------------
if __name__ == "__main__":
    kp, ki, kd = optimise_pid()
    print(f"Gains → Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}")

    t, y_hist, u_hist = simulate((kp, ki, kd))
    plot(t, y_hist, u_hist)
