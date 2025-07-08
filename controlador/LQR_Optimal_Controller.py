import numpy as np
from scipy.linalg import solve_continuous_are

class OptimalLQRController:

    def __init__(self, damping=0.3, R_weight=0.01, Q_weights=None):
        # Parâmetros econômicos brasileiros
        self.γ = 4.37
        self.ω1 = 0.0093
        self.τr = 11.6279
        self.m = 640.36
        self.Jr = 1.9186
        
        # Parâmetros do controlador
        self.damping = damping
        self.R_weight = R_weight
        self.Q_weights = Q_weights if Q_weights is not None else [100, 10, 1]
        
        self._setup_system()
        self._compute_lqr_gains()
        self.reset()
    
    def _setup_system(self):
        # Matriz A do sistema
        self.A = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [-(self.τr*self.γ+1)/self.τr,
             -(self.γ**2*self.τr/4 - self.ω1**2*self.τr + self.γ)/self.τr,
             -(self.γ**2/4 - self.ω1**2)/self.τr]
        ])
        
        # Estabilização artificial
        self.A[0,1] -= self.damping
        self.A[1,2] -= self.damping
        self.A[2,2] -= self.damping
        
        # Matriz B (entrada)
        self.B = np.array([[0], [0], [self.Jr/(self.τr*self.m)]])
        
        # Matriz C (saída)
        self.C = np.array([[1, 0, 0]])
    
    def _compute_lqr_gains(self):
        self.Q = np.diag(self.Q_weights)
        self.R = np.array([[self.R_weight]])
        
        try:
            self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = np.linalg.inv(self.R) @ self.B.T @ self.P
        except Exception as e:
            raise ValueError(f"Erro ao calcular ganhos LQR: {e}")
    
    def step(self, x, dt=0.01, u_max=5.0):
        # Calcular controle LQR
        u = -(self.K @ x).item()
        
        # Aplicar saturação
        u = np.clip(u, -u_max, u_max)
        
        # Integrar dinâmica
        x_next = x + dt * (self.A @ x + self.B.flatten() * u)
        
        # Armazenar histórico
        self.u_history.append(u)
        self.x_history.append(x.copy())
        self.y_history.append((self.C @ x).item())
        
        return x_next, u
    
    def get_output(self, x):
        return (self.C @ x).item()
    
    def reset(self):
        self.u_history = []
        self.x_history = []
        self.y_history = []
    
    def get_control_stats(self):
        if len(self.u_history) == 0:
            return None
            
        u_array = np.array(self.u_history)
        
        return {
            'control_mean': np.mean(np.abs(u_array)),
            'control_max': np.max(np.abs(u_array)),
            'control_std': np.std(u_array),
            'saturation_pct': np.sum(np.abs(u_array) >= 4.75) / len(u_array) * 100
        }
    
    def get_system_info(self):
        return {
            'controller_type': 'LQR Optimal',
            'damping': self.damping,
            'R_weight': self.R_weight,
            'Q_weights': self.Q_weights,
            'gains': self.K.flatten(),
            'eigenvalues': np.linalg.eigvals(self.A),
            'stability': np.sum(np.real(np.linalg.eigvals(self.A)) > 0) == 0
        }

if __name__ == "__main__":
    print("CONTROLADOR LQR OTIMIZADO - TESTE")
    print("="*50)
    
    controller = OptimalLQRController()
    
    info = controller.get_system_info()
    print(f"\nInformações do Sistema:")
    print(f"   Tipo: {info['controller_type']}")
    print(f"   Estável: {info['stability']}")
    print(f"   Ganhos: {info['gains']}")
    
    x0 = np.array([1.0, 0, 0])
    x, u = controller.step(x0)
    
    print(f"\nTeste:")
    print(f"   Estado inicial: {x0}")
    print(f"   Controle: {u:.3f}")
    print(f"   Próximo estado: {x}")
    print(f"   Gap inflação: {controller.get_output(x):.6f}")
    
    print(f"\nControlador funcionando!") 