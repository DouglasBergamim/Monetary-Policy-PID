import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parâmetros do sistema estimados
beta_pi, alpha, beta_y, gamma = 0.7, 0.3, 0.8, 0.4
A = np.array([[beta_pi, alpha],
              [0,       beta_y]])
B = np.array([[0],
              [gamma]])

# Função de simulação do sistema para dado K
def simulate(K, x0=[2.0, 0.0], n_steps=20):
    X = np.zeros((2, n_steps+1))
    X[:, 0] = x0
    for t in range(n_steps):
        u_t = -np.dot(K, X[:, t])
        X[:, t+1] = A @ X[:, t] + (B * u_t).flatten()
    return X

# Função de perda: soma dos quadrados dos gaps, ponderando gap do produto
def loss(K, lambda_y=0.5):
    X = simulate(K)
    pi_g, y_g = X[0, :], X[1, :]
    return np.sum(pi_g**2 + lambda_y * y_g**2)

# Otimização de K
K0 = np.array([1.5, 0.5])  # chute inicial
bounds = [(0, 5), (0, 5)]
result = minimize(loss, K0, bounds=bounds)
K_opt = result.x
print(f"K ótimo encontrado: {K_opt}")

# Simulações para K original e K ótimo
X_K0 = simulate(K0)
X_Kopt = simulate(K_opt)

# Plotar comparação
t = np.arange(X_K0.shape[1])
plt.figure(figsize=(12,6))

# Gap de inflação
plt.subplot(2,1,1)
plt.plot(t, X_K0[0], 'r--', label=f'Gap Inflação - K original {K0}')
plt.plot(t, X_Kopt[0], 'b-', label=f'Gap Inflação - K ótimo {K_opt.round(2)}')
plt.axhline(0, color='gray', linestyle='--')
plt.ylabel('Gap de inflação (p.p.)')
plt.legend()
plt.title('Comparação entre K original e K ótimo')
plt.grid(True, alpha=0.3)

# Gap de produto
plt.subplot(2,1,2)
plt.plot(t, X_K0[1], 'r--', label=f'Gap Produto - K original {K0}')
plt.plot(t, X_Kopt[1], 'b-', label=f'Gap Produto - K ótimo {K_opt.round(2)}')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Período')
plt.ylabel('Gap de produto (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparacao_Ks.png', dpi=300, bbox_inches='tight')
plt.show()

print("Gráfico salvo como comparacao_Ks.png")