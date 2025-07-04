import pandas as pd
import numpy as np
from scipy.optimize import least_squares

# 1. Carregar arquivo
file_path = 'dados_finais_modelo.xlsx'
df = pd.read_excel(file_path)
# Colunas: 'Data', 'Gap_Produto', 'Gap_Inflacao'

dates = pd.to_datetime(df['Data'])
# Extract series
g_y = df['Gap_Produto'].values
g_pi = df['Gap_Inflacao'].values

# 2. Tempo
dt = 1.0 / 4 # dados são trimestrais, então dt = 1/4 ano
n = len(g_y)
t = np.arange(0, n*dt, dt)

# 3. Função do kernel chi_y(t)
def chi_kernel(t, gamma, w1, m):
    # chi_y(t) = (1/(w1*m)) * exp(-gamma*t/2) * sinh(w1*t)
    return (1.0 / (w1 * m)) * np.exp(-gamma * t / 2) * np.sinh(w1 * t)

# 4. Convolucao
def predict_gap_infl(gamma, w1, m):
    chi = chi_kernel(t, gamma, w1, m)
    # convolution discrete approximation: sum_{i=0}^k chi[i] * g_y[k-i] * dt
    pred = dt * np.convolve(g_y, chi)[:n]
    return pred

# 5. Residuals for least squares
def residuals(params):
    gamma, w1, m = params
    pred = predict_gap_infl(gamma, w1, m)
    return pred - g_pi

# 6. Chute inicial
initial = [1.0, 0.01, 500.0]
bounds = ([1e-6, 1e-6, 1e-6], [np.inf, np.inf, np.inf])

# 7. Ajuste
res = least_squares(residuals, x0=initial, bounds=bounds)
gamma_est, w1_est, m_est = res.x

# 8. Resultados
print("Estimated parameters:")
print(f"gamma = {gamma_est:.4f} per year")
print(f"omega_1 = {w1_est:.5e} per year")
print(f"m = {m_est:.2f} year^2")

# 9. Correlacao
pred = predict_gap_infl(gamma_est, w1_est, m_est)
corr = np.corrcoef(pred, g_pi)[0,1]
print(f"Correlation between predicted and actual inflation gap: {corr:.3f}")
