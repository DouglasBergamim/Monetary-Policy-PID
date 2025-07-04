#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise da Postura do Banco Central: Hawkish vs Dovish
Baseado nos resultados da simulação econométrica
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configurar matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

def analisar_postura_bc():
    """
    Análise completa da postura do Banco Central
    """
    print("=" * 80)
    print("ANÁLISE DA POSTURA DO BANCO CENTRAL")
    print("Hawkish vs Dovish")
    print("=" * 80)
    
    # Parâmetros estimados da simulação
    beta_pi = 0.7865  # Persistência da inflação
    alpha = 0.2225    # Sensibilidade ao gap do produto
    beta_y = 0.8000   # Persistência do produto
    gamma = 0.4000    # Sensibilidade aos juros
    
    # Ganhos da política monetária (da simulação)
    K_inflacao = 1.5  # Resposta à inflação
    K_produto = 0.5   # Resposta ao produto
    
    print(f"\n📊 PARÂMETROS ESTIMADOS:")
    print(f"   β_π (persistência da inflação): {beta_pi:.4f}")
    print(f"   α (sensibilidade ao gap do produto): {alpha:.4f}")
    print(f"   β_y (persistência do produto): {beta_y:.4f}")
    print(f"   γ (sensibilidade aos juros): {gamma:.4f}")
    
    print(f"\n🎯 GANHOS DA POLÍTICA MONETÁRIA:")
    print(f"   K_π (resposta à inflação): {K_inflacao:.1f}")
    print(f"   K_y (resposta ao produto): {K_produto:.1f}")
    
    # Análise 1: Intensidade da Resposta
    print(f"\n" + "=" * 60)
    print("1. ANÁLISE DA INTENSIDADE DA RESPOSTA")
    print("=" * 60)
    
    # Critérios para classificação
    criterios = {
        'resposta_inflacao': K_inflacao,
        'resposta_produto': K_produto,
        'ratio_inflacao_produto': K_inflacao / K_produto,
        'sensibilidade_juros': gamma,
        'persistencia_inflacao': beta_pi
    }
    
    print(f"Resposta à inflação (K_π): {criterios['resposta_inflacao']:.1f}")
    print(f"Resposta ao produto (K_y): {criterios['resposta_produto']:.1f}")
    print(f"Ratio π/y: {criterios['ratio_inflacao_produto']:.1f}")
    print(f"Sensibilidade aos juros (γ): {criterios['sensibilidade_juros']:.3f}")
    print(f"Persistência da inflação (β_π): {criterios['persistencia_inflacao']:.3f}")
    
    # Classificação baseada em critérios
    pontos_hawkish = 0
    pontos_dovish = 0
    
    # Critério 1: Resposta à inflação
    if K_inflacao >= 1.5:
        pontos_hawkish += 2
        print(f"\n✓ HAWKISH: Resposta forte à inflação (K_π = {K_inflacao:.1f} ≥ 1.5)")
    elif K_inflacao >= 1.0:
        pontos_hawkish += 1
        print(f"\n~ MODERADO: Resposta moderada à inflação (K_π = {K_inflacao:.1f})")
    else:
        pontos_dovish += 1
        print(f"\n✓ DOVISH: Resposta suave à inflação (K_π = {K_inflacao:.1f} < 1.0)")
    
    # Critério 2: Priorização inflação vs produto
    ratio = K_inflacao / K_produto
    if ratio >= 3.0:
        pontos_hawkish += 2
        print(f"✓ HAWKISH: Forte priorização da inflação (ratio = {ratio:.1f} ≥ 3.0)")
    elif ratio >= 2.0:
        pontos_hawkish += 1
        print(f"~ MODERADO HAWKISH: Priorização da inflação (ratio = {ratio:.1f})")
    else:
        pontos_dovish += 1
        print(f"✓ DOVISH: Política mais equilibrada (ratio = {ratio:.1f} < 2.0)")
    
    # Critério 3: Sensibilidade aos juros
    if gamma >= 0.5:
        pontos_hawkish += 1
        print(f"✓ HAWKISH: Alta sensibilidade aos juros (γ = {gamma:.3f} ≥ 0.5)")
    elif gamma >= 0.3:
        print(f"~ MODERADO: Sensibilidade moderada aos juros (γ = {gamma:.3f})")
    else:
        pontos_dovish += 1
        print(f"✓ DOVISH: Baixa sensibilidade aos juros (γ = {gamma:.3f} < 0.3)")
    
    # Critério 4: Persistência da inflação
    if beta_pi >= 0.8:
        pontos_dovish += 1
        print(f"✓ DOVISH: Alta persistência inflacionária (β_π = {beta_pi:.3f} ≥ 0.8)")
    elif beta_pi >= 0.6:
        print(f"~ MODERADO: Persistência moderada (β_π = {beta_pi:.3f})")
    else:
        pontos_hawkish += 1
        print(f"✓ HAWKISH: Baixa persistência inflacionária (β_π = {beta_pi:.3f} < 0.6)")
    
    return criterios, pontos_hawkish, pontos_dovish

def simular_cenarios_comparativos():
    """
    Simular diferentes cenários para comparação
    """
    print(f"\n" + "=" * 60)
    print("2. SIMULAÇÃO DE CENÁRIOS COMPARATIVOS")
    print("=" * 60)
    
    # Parâmetros do sistema
    beta_pi, alpha = 0.7865, 0.2225
    beta_y, gamma = 0.8, 0.4
    A = np.array([[beta_pi, alpha], [0, beta_y]])
    B = np.array([[0], [gamma]])
    
    # Cenários de política
    cenarios = {
        'Muito Hawkish': np.array([[2.5, 0.3]]),    # Foco extremo na inflação
        'Hawkish': np.array([[2.0, 0.5]]),          # Foco forte na inflação
        'Atual (Estimado)': np.array([[1.5, 0.5]]), # Política estimada
        'Dovish': np.array([[1.0, 0.8]]),           # Mais equilibrado
        'Muito Dovish': np.array([[0.8, 1.0]])      # Foco no produto
    }
    
    # Estado inicial: choque inflacionário de +2 p.p.
    x0 = np.array([[2.0], [0.0]])
    n_steps = 20
    
    resultados = {}
    metricas = {}
    
    for nome, K in cenarios.items():
        # Simular
        X = np.zeros((2, n_steps + 1))
        X[:, [0]] = x0
        
        for t in range(n_steps):
            u_t = -K @ X[:, [t]]
            X[:, [t+1]] = A @ X[:, [t]] + B * u_t
        
        resultados[nome] = X
        
        # Calcular métricas
        tempo_convergencia = None
        for t in range(n_steps + 1):
            if abs(X[0, t]) < 0.1:  # Inflação < 0.1%
                tempo_convergencia = t
                break
        
        volatilidade_produto = np.std(X[1, :])
        reducao_inflacao = X[0, 0] - X[0, -1]
        
        metricas[nome] = {
            'tempo_convergencia': tempo_convergencia if tempo_convergencia else n_steps,
            'volatilidade_produto': volatilidade_produto,
            'reducao_inflacao': reducao_inflacao,
            'inflacao_final': X[0, -1],
            'produto_final': X[1, -1]
        }
    
    # Exibir métricas
    print(f"\n📈 MÉTRICAS COMPARATIVAS:")
    print(f"{'Cenário':<20} {'Converg.':<8} {'Vol.Prod.':<10} {'Red.Infl.':<10} {'Infl.Final':<12}")
    print("-" * 70)
    
    for nome, metrica in metricas.items():
        print(f"{nome:<20} {metrica['tempo_convergencia']:<8} "
              f"{metrica['volatilidade_produto']:<10.3f} "
              f"{metrica['reducao_inflacao']:<10.2f} "
              f"{metrica['inflacao_final']:<12.3f}")
    
    return resultados, metricas

def analisar_trade_offs(metricas):
    """
    Analisar trade-offs entre velocidade e volatilidade
    """
    print(f"\n" + "=" * 60)
    print("3. ANÁLISE DE TRADE-OFFS")
    print("=" * 60)
    
    # Extrair dados para análise
    nomes = list(metricas.keys())
    tempos = [metricas[nome]['tempo_convergencia'] for nome in nomes]
    volatilidades = [metricas[nome]['volatilidade_produto'] for nome in nomes]
    
    print(f"\n🔄 TRADE-OFF VELOCIDADE vs VOLATILIDADE:")
    
    # Encontrar política atual
    idx_atual = nomes.index('Atual (Estimado)')
    tempo_atual = tempos[idx_atual]
    vol_atual = volatilidades[idx_atual]
    
    print(f"\nPolítica Atual:")
    print(f"  - Tempo de convergência: {tempo_atual} períodos")
    print(f"  - Volatilidade do produto: {vol_atual:.3f}")
    
    # Comparar com alternativas
    print(f"\nComparação com alternativas:")
    
    for i, nome in enumerate(nomes):
        if nome == 'Atual (Estimado)':
            continue
            
        tempo = tempos[i]
        vol = volatilidades[i]
        
        delta_tempo = tempo - tempo_atual
        delta_vol = vol - vol_atual
        
        if delta_tempo < 0 and delta_vol > 0:
            tipo = "HAWKISH"
            desc = "mais rápida, mais volátil"
        elif delta_tempo > 0 and delta_vol < 0:
            tipo = "DOVISH"
            desc = "mais lenta, menos volátil"
        elif delta_tempo < 0 and delta_vol < 0:
            tipo = "SUPERIOR"
            desc = "mais rápida E menos volátil"
        else:
            tipo = "INFERIOR"
            desc = "mais lenta E mais volátil"
        
        print(f"  {nome}: {tipo} - {desc}")
        print(f"    Δ tempo: {delta_tempo:+d}, Δ volatilidade: {delta_vol:+.3f}")
    
    return tempo_atual, vol_atual

def criar_visualizacao_comparativa(resultados, metricas):
    """
    Criar visualização comparativa dos cenários
    """
    print(f"\n" + "=" * 60)
    print("4. GERAÇÃO DE VISUALIZAÇÕES")
    print("=" * 60)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise Comparativa: Postura do Banco Central', fontsize=16, fontweight='bold')
    
    cores = ['red', 'orange', 'blue', 'green', 'purple']
    
    # 1. Evolução da inflação
    for i, (nome, X) in enumerate(resultados.items()):
        periodos = range(X.shape[1])
        cor = cores[i]
        estilo = '-' if nome == 'Atual (Estimado)' else '--'
        largura = 3 if nome == 'Atual (Estimado)' else 2
        
        ax1.plot(periodos, X[0, :], color=cor, linestyle=estilo, 
                linewidth=largura, label=nome)
    
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Período')
    ax1.set_ylabel('Gap de Inflação (%)')
    ax1.set_title('Evolução do Gap de Inflação')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Evolução do produto
    for i, (nome, X) in enumerate(resultados.items()):
        periodos = range(X.shape[1])
        cor = cores[i]
        estilo = '-' if nome == 'Atual (Estimado)' else '--'
        largura = 3 if nome == 'Atual (Estimado)' else 2
        
        ax2.plot(periodos, X[1, :], color=cor, linestyle=estilo, 
                linewidth=largura, label=nome)
    
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Período')
    ax2.set_ylabel('Gap de Produto (%)')
    ax2.set_title('Evolução do Gap de Produto')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade-off velocidade vs volatilidade
    nomes = list(metricas.keys())
    tempos = [metricas[nome]['tempo_convergencia'] for nome in nomes]
    volatilidades = [metricas[nome]['volatilidade_produto'] for nome in nomes]
    
    for i, nome in enumerate(nomes):
        cor = cores[i]
        tamanho = 150 if nome == 'Atual (Estimado)' else 100
        marcador = 'o' if nome == 'Atual (Estimado)' else 's'
        
        ax3.scatter(tempos[i], volatilidades[i], color=cor, s=tamanho, 
                   marker=marcador, label=nome, alpha=0.7)
    
    ax3.set_xlabel('Tempo de Convergência (períodos)')
    ax3.set_ylabel('Volatilidade do Produto')
    ax3.set_title('Trade-off: Velocidade vs Volatilidade')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Adicionar setas indicativas
    ax3.annotate('HAWKISH\n(rápido, volátil)', xy=(0.1, 0.9), xycoords='axes fraction',
                ha='left', va='top', fontsize=10, color='red', weight='bold')
    ax3.annotate('DOVISH\n(lento, suave)', xy=(0.9, 0.1), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10, color='green', weight='bold')
    
    # 4. Métricas resumo
    ax4.axis('off')
    ax4.set_title('Resumo das Métricas')
    
    # Tabela de métricas
    tabela_texto = "Cenário                Converg.  Vol.Prod.  Red.Infl.\n"
    tabela_texto += "-" * 50 + "\n"
    
    for nome, metrica in metricas.items():
        tabela_texto += f"{nome:<20} {metrica['tempo_convergencia']:<8} "
        tabela_texto += f"{metrica['volatilidade_produto']:<9.3f} "
        tabela_texto += f"{metrica['reducao_inflacao']:<8.2f}\n"
    
    ax4.text(0.1, 0.9, tabela_texto, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/analise_postura_bc.png', dpi=300, bbox_inches='tight')
    print("✅ Visualização salva: analise_postura_bc.png")
    
    return fig

def conclusao_final(pontos_hawkish, pontos_dovish, tempo_atual, vol_atual):
    """
    Conclusão final sobre a postura do BC
    """
    print(f"\n" + "=" * 80)
    print("CONCLUSÃO FINAL: POSTURA DO BANCO CENTRAL")
    print("=" * 80)
    
    print(f"\n📊 PONTUAÇÃO DOS CRITÉRIOS:")
    print(f"   Pontos Hawkish: {pontos_hawkish}")
    print(f"   Pontos Dovish: {pontos_dovish}")
    
    # Determinação da postura
    if pontos_hawkish > pontos_dovish + 1:
        postura = "HAWKISH"
        cor = "🔴"
        descricao = "agressiva, priorizando controle rápido da inflação"
    elif pontos_dovish > pontos_hawkish + 1:
        postura = "DOVISH"
        cor = "🟢"
        descricao = "cautelosa, priorizando estabilidade do produto"
    else:
        postura = "MODERADA"
        cor = "🟡"
        descricao = "equilibrada entre inflação e produto"
    
    print(f"\n{cor} POSTURA IDENTIFICADA: {postura}")
    print(f"   Característica: Política {descricao}")
    
    print(f"\n🎯 CARACTERÍSTICAS OBSERVADAS:")
    
    if postura == "HAWKISH":
        print(f"   ✓ Resposta forte à inflação (K_π = 1.5)")
        print(f"   ✓ Priorização da estabilidade de preços")
        print(f"   ✓ Convergência relativamente rápida ({tempo_atual} períodos)")
        print(f"   ⚠ Volatilidade moderada do produto ({vol_atual:.3f})")
        
        print(f"\n📈 IMPLICAÇÕES:")
        print(f"   • Controle eficaz de choques inflacionários")
        print(f"   • Possível impacto no crescimento econômico")
        print(f"   • Credibilidade na meta de inflação")
        
    elif postura == "DOVISH":
        print(f"   ✓ Resposta equilibrada entre inflação e produto")
        print(f"   ✓ Menor volatilidade do produto")
        print(f"   ✓ Política mais gradual e previsível")
        print(f"   ⚠ Convergência mais lenta")
        
        print(f"\n📈 IMPLICAÇÕES:")
        print(f"   • Menor risco de recessão")
        print(f"   • Possível persistência inflacionária")
        print(f"   • Suporte ao crescimento econômico")
        
    else:  # MODERADA
        print(f"   ✓ Equilíbrio entre objetivos de inflação e produto")
        print(f"   ✓ Resposta proporcional aos desvios")
        print(f"   ✓ Política previsível e consistente")
        
        print(f"\n📈 IMPLICAÇÕES:")
        print(f"   • Política bem calibrada")
        print(f"   • Trade-off equilibrado")
        print(f"   • Flexibilidade para ajustes")
    
    print(f"\n🔍 EVIDÊNCIAS QUANTITATIVAS:")
    print(f"   • Ganho na inflação (K_π): 1.5 - Moderadamente alto")
    print(f"   • Ganho no produto (K_y): 0.5 - Moderado")
    print(f"   • Ratio π/y: 3.0 - Priorização da inflação")
    print(f"   • Tempo de convergência: {tempo_atual} períodos")
    print(f"   • Volatilidade do produto: {vol_atual:.3f}")
    
    return postura

def main():
    """
    Função principal da análise
    """
    # Análise dos parâmetros
    criterios, pontos_hawkish, pontos_dovish = analisar_postura_bc()
    
    # Simulação comparativa
    resultados, metricas = simular_cenarios_comparativos()
    
    # Análise de trade-offs
    tempo_atual, vol_atual = analisar_trade_offs(metricas)
    
    # Visualização
    fig = criar_visualizacao_comparativa(resultados, metricas)
    
    # Conclusão final
    postura = conclusao_final(pontos_hawkish, pontos_dovish, tempo_atual, vol_atual)
    
    return postura, criterios, metricas

if __name__ == "__main__":
    postura_final, criterios, metricas = main()

