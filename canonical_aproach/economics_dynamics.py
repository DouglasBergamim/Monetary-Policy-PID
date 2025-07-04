import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel("dados_finais_modelo.xlsx")

# Estimar a equacao de phillips

df['Gap_Inflacao_Lag'] = df['Gap_Inflacao'].shift(1)

X_phillips = df[['Gap_Inflacao_Lag', 'Gap_Produto']].dropna()
y_phillips = df['Gap_Inflacao'].loc[X_phillips.index]

X_phillips_const = sm.add_constant(X_phillips)
model_phillips = sm.OLS(y_phillips, X_phillips_const).fit()
print("Equação de Phillips:\n", model_phillips.summary())

beta_pi = model_phillips.params['Gap_Inflacao_Lag']
alpha = model_phillips.params['Gap_Produto']

# Estimar IS

df['Gap_Produto_Lead'] = df['Gap_Produto'].shift(-1)  # y^g no período t+1

X_is = df[['Gap_Produto', 'Gap_Juros']].dropna()
y_is = df['Gap_Produto_Lead'].loc[X_is.index]

X_is_const = sm.add_constant(X_is)
model_is = sm.OLS(y_is, X_is_const).fit()
print("Equação IS:\n", model_is.summary())

beta_y = model_is.params['Gap_Produto']
gamma = -model_is.params['Gap_Juros']  # o sinal de γ é invertido pois juros mais altos reduzem produto

# Construir matrizes do sistema

A = np.array([[beta_pi, alpha],
              [0,       beta_y]])
B = np.array([[0],
              [gamma]])

print("Matriz A:\n", A)
print("Matriz B:\n", B)

# Simular

import matplotlib.pyplot as plt

# Ajuste os ganhos conforme seu objetivo de simulação
K = np.array([[1.5, 0.5]])

# Condições iniciais: ex.: choque de inflação inicial de +2 p.p.
x0 = np.array([[2.0],   # gap de inflação
               [0.0]])  # gap de produto

n_steps = 20
X = np.zeros((2, n_steps+1))
X[:, [0]] = x0

for t in range(n_steps):
    u_t = -K @ X[:, [t]]
    X[:, [t+1]] = A @ X[:, [t]] + B * u_t

# Plotar trajetórias
plt.figure(figsize=(10,6))
plt.plot(range(n_steps+1), X[0, :], label="Gap de inflação (πg)")
plt.plot(range(n_steps+1), X[1, :], label="Gap de produto (yg)")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Período")
plt.ylabel("Valor do gap")
plt.legend()
plt.title("Simulação da resposta do sistema com feedback de estado")
plt.grid()
plt.tight_layout()
plt.show()