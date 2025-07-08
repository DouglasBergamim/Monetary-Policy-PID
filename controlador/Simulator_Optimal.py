import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from LQR_Optimal_Controller import OptimalLQRController

class MonetaryPolicySimulator:
    
    def __init__(self, controller=None, dt=0.01, u_max=5.0):
        self.dt = dt
        self.u_max = u_max
        self.reference = 0.0
        
        self.results_dir = os.path.join('..', 'Resultados')
        self._setup_results_directory()
        
        if controller is None:
            self.controller = OptimalLQRController()
        else:
            self.controller = controller
    
    def _setup_results_directory(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def simulate(self, x0, tf=20.0, scenarios=None):
        if scenarios is None:
            # Simulação única
            return self._simulate_single(x0, tf)
        else:
            # Múltiplos cenários
            return self._simulate_multiple(scenarios, tf)
    
    def _simulate_single(self, x0, tf):
        if isinstance(x0, (int, float)):
            x0 = np.array([float(x0), 0, 0])
        
        self.controller.reset()
        
        n = int(tf / self.dt)
        t = np.arange(n) * self.dt
        
        x_hist = np.zeros((n, 3))
        u_hist = np.zeros(n)
        y_hist = np.zeros(n)
        
        x = x0.copy()
        
        for k in range(n):
            y = self.controller.get_output(x)
            x, u = self.controller.step(x, self.dt, self.u_max)
            
            x_hist[k] = x
            u_hist[k] = u
            y_hist[k] = y
        
        stats = self._calculate_metrics(t, x_hist, u_hist, y_hist)
        
        return {
            'time': t,
            'states': x_hist,
            'control': u_hist,
            'output': y_hist,
            'initial_condition': x0,
            'metrics': stats,
            'controller_stats': self.controller.get_control_stats()
        }
    
    def _simulate_multiple(self, scenarios, tf):
        results = {}
        
        for name, x0 in scenarios:
            result = self._simulate_single(x0, tf)
            results[name] = result
        
        return results
    
    def _calculate_metrics(self, t, x_hist, u_hist, y_hist):
        control_mean = np.mean(np.abs(u_hist))
        control_max = np.max(np.abs(u_hist))
        control_std = np.std(u_hist)
        
        saturation_pct = np.sum(np.abs(u_hist) >= 0.95 * self.u_max) / len(u_hist) * 100
        
        errors = np.abs(self.reference - y_hist)
        error_final = errors[-1]
        error_max = np.max(errors)
        
        itae = np.trapezoid(errors * t, t)
        ise = np.trapezoid((self.reference - y_hist)**2, t)
        control_effort = np.trapezoid(u_hist**2, t)
        
        settling_time = t[-1]
        if len(y_hist) > 0:
            settling_threshold = 0.05 * np.abs(y_hist[0])
            for i, error in enumerate(errors):
                if error < settling_threshold:
                    settling_time = t[i]
                    break
        
        return {
            'control_mean': control_mean,
            'control_max': control_max,
            'control_std': control_std,
            'saturation_pct': saturation_pct,
            'error_final': error_final,
            'error_max': error_max,
            'itae': itae,
            'ise': ise,
            'control_effort': control_effort,
            'settling_time': settling_time
        }
    
    def save_plot(self, results, scenario_name="Simulacao"):
        is_multiple = not ('time' in results)
        
        plt.ioff()
        
        if is_multiple:
            self._plot_multiple_results(results, save_mode=True)
        else:
            self._plot_single_result(results, "LQR Optimal Controller", save_mode=True)
        
        filename = "LQR_final_result.png"
        filepath = os.path.join(self.results_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Gráfico salvo: {filename}")
        print(f"Local: {self.results_dir}")
        
        return filepath
    
    def plot_results(self, results, scenario_name="Simulação", save_plot=False):
        if isinstance(results, dict) and 'time' in results:
            self._plot_single_result(results, scenario_name)
        else:
            self._plot_multiple_results(results)
        
        if save_plot:
            return self.save_plot(results)
        return None
    
    def _plot_single_result(self, result, title, save_mode=False):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{title}', fontsize=16, fontweight='bold')
        
        t = result['time']
        
        axes[0,0].plot(t, result['output'], 'b-', linewidth=2.5, label='π gap')
        axes[0,0].axhline(0, ls='--', c='r', alpha=0.8, linewidth=2, label='meta = 0')
        axes[0,0].axhline(0.1, ls=':', c='g', alpha=0.6, label='±10%')
        axes[0,0].axhline(-0.1, ls=':', c='g', alpha=0.6)
        axes[0,0].set_xlabel('tempo (anos)')
        axes[0,0].set_ylabel('gap inflação (%)')
        axes[0,0].set_title('Gap de Inflação')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        axes[0,1].plot(t, result['control'], 'r-', linewidth=2.5, 
                       label=f'gap juros (médio: {result["metrics"]["control_mean"]:.3f})')
        axes[0,1].axhline(self.u_max, ls='--', c='k', alpha=0.8, linewidth=2, label='limite ±5')
        axes[0,1].axhline(-self.u_max, ls='--', c='k', alpha=0.8, linewidth=2)
        axes[0,1].axhline(0, ls=':', c='gray', alpha=0.5)
        axes[0,1].set_xlabel('tempo (anos)')
        axes[0,1].set_ylabel('gap juros (%)')
        axes[0,1].set_title(f'Controle (Sat: {result["metrics"]["saturation_pct"]:.1f}%)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        axes[1,0].plot(t, result['states'][:, 0], 'b-', linewidth=2, label='x₁ (π gap)')
        axes[1,0].plot(t, result['states'][:, 1], 'g-', linewidth=2, label='x₂ (π̇ gap)')
        axes[1,0].plot(t, result['states'][:, 2], 'r-', linewidth=2, label='x₃ (π̈ gap)')
        axes[1,0].set_xlabel('tempo (anos)')
        axes[1,0].set_ylabel('estados')
        axes[1,0].set_title('Estados do Sistema')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        if not save_mode:
            plt.show()
    
    def _plot_multiple_results(self, results, save_mode=False):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('COMPARAÇÃO DE CENÁRIOS - CONTROLADOR LQR OTIMIZADO', 
                    fontsize=16, fontweight='bold')
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        for i, (name, result) in enumerate(results.items()):
            color = colors[i % len(colors)]
            t = result['time']
            
            axes[0,0].plot(t, result['output'], color=color, linewidth=2, 
                          label=f'{name} (ctrl: {result["metrics"]["control_mean"]:.3f})')
            
            axes[0,1].plot(t, result['control'], color=color, linewidth=2, 
                          label=f'{name} (sat: {result["metrics"]["saturation_pct"]:.1f}%)')
        
        axes[0,0].axhline(0, ls='--', c='r', alpha=0.8)
        axes[0,0].set_xlabel('tempo (anos)')
        axes[0,0].set_ylabel('gap inflação (%)')
        axes[0,0].set_title('Gap de Inflação - Todos os Cenários')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        axes[0,1].axhline(self.u_max, ls='--', c='k', alpha=0.8)
        axes[0,1].axhline(-self.u_max, ls='--', c='k', alpha=0.8)
        axes[0,1].set_xlabel('tempo (anos)')
        axes[0,1].set_ylabel('gap juros (%)')
        axes[0,1].set_title('Controle - Todos os Cenários')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
        
        axes[1,0].axis('off')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        
        if not save_mode:
            plt.show()

if __name__ == "__main__":
    print("SIMULADOR DE POLÍTICA MONETÁRIA - LQR OTIMIZADO")
    print("="*60)
    
    simulator = MonetaryPolicySimulator()
    
    scenarios = [
        ("Suave", np.array([0.5, 0, 0])),
        ("Moderado", np.array([1.0, 0, 0])),
        ("Severo", np.array([2.0, 0, 0]))
    ]
    
    results = simulator.simulate(None, scenarios=scenarios)
    
    simulator.plot_results(results, "Cenários Múltiplos", save_plot=True)
    
    saved_file = simulator.save_plot(results)