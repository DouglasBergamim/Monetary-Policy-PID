#!/usr/bin/env python3
"""
Exemplo simples de uso do Simulator_Optimal.py com salvamento de gráficos

Este exemplo demonstra como salvar apenas o gráfico final na pasta Resultados
"""

import numpy as np
from Simulator_Optimal import MonetaryPolicySimulator

def exemplo_basico():
    """
    Exemplo básico: simular e salvar o gráfico
    """
    print("🎯 EXEMPLO BÁSICO: SIMULAÇÃO COM SALVAMENTO DE GRÁFICO")
    print("-" * 60)
    
    # Criar simulador
    simulator = MonetaryPolicySimulator()
    
    # Condição inicial: gap de inflação inicial de 1.5%
    x0 = np.array([1.5, 0, 0])
    
    # Executar simulação
    result = simulator.simulate(x0, tf=15.0)
    
    # Plotar E salvar automaticamente
    saved_file = simulator.plot_results(result, "Simulação Básica", save_plot=True)
    
    print(f"✅ Gráfico salvo como: LQR_final_result.png")
    
    return result

def exemplo_multiplos_cenarios():
    """
    Exemplo com múltiplos cenários
    """
    print("\n🎯 EXEMPLO: MÚLTIPLOS CENÁRIOS")
    print("-" * 60)
    
    # Criar simulador
    simulator = MonetaryPolicySimulator()
    
    # Definir cenários
    scenarios = [
        ("Leve", np.array([0.5, 0, 0])),
        ("Moderado", np.array([1.0, 0, 0])),
        ("Severo", np.array([2.0, 0, 0]))
    ]
    
    # Executar simulação
    results = simulator.simulate(None, tf=20.0, scenarios=scenarios)
    
    # Salvar apenas o gráfico de comparação
    saved_file = simulator.save_plot(results)
    
    print(f"✅ Gráfico de comparação salvo como: LQR_final_result.png")
    
    return results

if __name__ == "__main__":
    print("🚀 EXEMPLO SIMPLES - SALVAMENTO DE GRÁFICOS")
    print("=" * 70)
    
    try:
        # Executar exemplos
        resultado1 = exemplo_basico()
        resultado2 = exemplo_multiplos_cenarios()
        
        print("\n🎉 EXEMPLOS EXECUTADOS COM SUCESSO!")
        print("📁 Verifique a pasta 'Resultados' para os gráficos gerados")
        print("📊 Arquivos PNG de alta qualidade (300 DPI)")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        print("🔧 Verifique se o LQR_Optimal_Controller.py está disponível") 