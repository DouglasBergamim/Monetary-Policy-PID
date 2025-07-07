from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import os

class StateSpacePlant:
    """Discrete-time state-space plant model.

    Parameters
    Ad, Bd, C, D : np.ndarray
        Discrete or continuous state-space matrices.  
    dt : float | None, optional
        Sampling period (seconds).  When ``None`` the matrices are taken
        as already discrete.
    x0 : Sequence[float] | None, optional
        Initial state vector.  If ``None`` the state starts at the origin.
    """

    def __init__(
        self,
        Ad: np.ndarray,
        Bd: np.ndarray,
        C:  np.ndarray,
        D:  np.ndarray | None = None,
        *,
        dt: float | None = None,
        x0: Sequence[float] | None = None
    ) -> None:
        # ---------- Tustin (bilinear) conversion ----------
        if dt is not None:
            I   = np.eye(Ad.shape[0])
            fac = np.linalg.inv(I - 0.5 * dt * Ad)
            Ad  = fac @ (I + 0.5 * dt * Ad)
            Bd  = fac @ (dt * Bd)

        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.C  = np.asarray(C,  dtype=float)
        self.D  = np.zeros((self.C.shape[0], self.Bd.shape[1]), dtype=float) if D is None else np.asarray(D, dtype=float)

        self.x0 = np.zeros(self.Ad.shape[0], dtype=float) if x0 is None else np.asarray(x0, dtype=float)
        self._hist_u: list[np.ndarray] = []
        self._hist_y: list[np.ndarray] = []

        self.reset()

    def reset(self, x0: Sequence[float] | None = None) -> None:
        """Reset the plant state and clear history."""
        self.x = self.x0.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
        self._hist_u.clear()
        self._hist_y.clear()

    def step(self, u: float | Sequence[float]):
        """Propagate the plant by one sample.

        Parameters
        u : float | Sequence[float]
            Control input at the current sample.

        Returns
        y : np.ndarray | float
            Plant output corresponding to the applied input.
        """
        u_vec = np.atleast_1d(u).astype(float)
        y = (self.C @ self.x + self.D @ u_vec).squeeze()

        self._hist_u.append(u_vec.copy())
        self._hist_y.append(np.atleast_1d(y).copy())

        self.x = self.Ad @ self.x + self.Bd @ u_vec
        return y

    def history(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the logged history *(u, y)* tuples."""
        return np.vstack(self._hist_u), np.vstack(self._hist_y)
    
# just for testing purposes
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------
    # 1. Matrizes CONTÍNUAS (2ª ordem under-damped)
    ζ, ω = 0.2, 1.0              # amortecimento, frequência natural
    A_c = np.array([[0, 1],
                    [-ω**2, -2*ζ*ω]])
    B_c = np.array([[0],
                    [ω**2]])
    C   = np.array([[1, 0]])     # saída = posição
    # D = 0  (padrão da classe)

    # ---------------------------------------------------------------
    # 2. Cria planta – dt=0.02 s (50 Hz) → Tustin interno
    dt    = 0.02
    plant = StateSpacePlant(A_c, B_c, C, dt=dt)

    # ---------------------------------------------------------------
    # 3. Simulação: degrau unitário por 10 s
    t_final = 10.0
    N       = int(t_final / dt)
    t       = np.arange(N) * dt

    y_log = np.zeros(N)
    for k in range(N):
        y_k = plant.step(u=1.0)   # degrau na entrada
        y_log[k] = y_k

    # ---------------------------------------------------------------
    # 4. Plot
    plt.figure(figsize=(6,3))
    plt.plot(t, y_log, label='y(t)')
    plt.axhline(1.0, ls='--', color='r', label='ref=1')
    plt.title('Resposta 2ª ordem (ζ=0.2, ω=1 rad/s) – Tustin')
    plt.xlabel('tempo [s]'); plt.ylabel('saída'); plt.grid(); plt.legend()
    plt.tight_layout()
    
    # Salvar na pasta Resultados
    resultados_dir = "Resultados"
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)
    
    filename = "StatePlant_resposta_2ordem.png"
    filepath = os.path.join(resultados_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Gráfico salvo: {filename}")
    print(f"📁 Local: {resultados_dir}")
    
    plt.show()
