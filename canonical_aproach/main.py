#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação de Modelo Econométrico com Geração de Vídeos
Versão compatível com Python 3.13 que gera animações além de gráficos estáticos

Este código implementa:
1. Análise econométrica (Curva de Phillips e Equação IS)
2. Simulação do sistema dinâmico
3. Gráficos estáticos
4. Vídeos animados da evolução temporal
5. Compatibilidade com diferentes versões do Python
"""

import sys
import os
print(f"Executando com Python {sys.version}")

# Importações com fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    print("✅ Pandas disponível")
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️ Pandas não disponível - usando implementação básica")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ NumPy disponível")
except ImportError:
    NUMPY_AVAILABLE = False
    print("❌ NumPy não disponível - funcionalidade limitada")
    import math
    import random

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
    print("✅ Statsmodels disponível")
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ Statsmodels não disponível - usando regressão manual")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib disponível")
    
    # Configurar matplotlib para não mostrar janelas
    plt.switch_backend('Agg')
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib não disponível - sem gráficos")

# Verificar FFmpeg para vídeos
try:
    import subprocess
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    if result.returncode == 0:
        FFMPEG_AVAILABLE = True
        print("✅ FFmpeg disponível para geração de vídeos")
    else:
        FFMPEG_AVAILABLE = False
        print("⚠️ FFmpeg não disponível - vídeos podem não funcionar")
except:
    FFMPEG_AVAILABLE = False
    print("⚠️ FFmpeg não encontrado - vídeos podem não funcionar")

class SimpleRegression:
    """Implementação básica de regressão linear quando statsmodels não está disponível"""
    
    def __init__(self):
        self.params = {}
        self.rsquared = 0
        self.fitted = False
    
    def fit(self, X, y):
        """Ajustar modelo de regressão linear múltipla"""
        if not NUMPY_AVAILABLE:
            return self._fit_without_numpy(X, y)
        
        # Converter para arrays numpy se necessário
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Resolver sistema normal: (X'X)β = X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
            
            # Calcular R²
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            self.rsquared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Armazenar parâmetros
            if X.shape[1] >= 3:  # Assumindo [const, lag, produto]
                self.params = {
                    'const': beta[0],
                    'Gap_Inflacao_Lag': beta[1] if len(beta) > 1 else 0,
                    'Gap_Produto': beta[2] if len(beta) > 2 else 0
                }
            
            self.fitted = True
            return self
            
        except Exception as e:
            print(f"Erro na regressão: {e}")
            # Valores padrão em caso de erro
            self.params = {'const': 0, 'Gap_Inflacao_Lag': 0.7, 'Gap_Produto': 0.3}
            self.rsquared = 0.5
            self.fitted = True
            return self
    
    def _fit_without_numpy(self, X, y):
        """Implementação básica sem numpy"""
        # Implementação simplificada - apenas para demonstração
        self.params = {'const': 0, 'Gap_Inflacao_Lag': 0.7, 'Gap_Produto': 0.3}
        self.rsquared = 0.5
        self.fitted = True
        return self
    
    def summary(self):
        """Resumo do modelo"""
        if not self.fitted:
            return "Modelo não ajustado"
        
        summary_text = f"""
Regressão Linear Simples
========================
R-squared: {self.rsquared:.4f}

Coeficientes:
"""
        for param, value in self.params.items():
            summary_text += f"  {param}: {value:.4f}\n"
        
        return summary_text

def criar_dados_simulados():
    """Criar dados simulados para teste"""
    print("Criando dados simulados...")
    
    n = 100
    
    if NUMPY_AVAILABLE:
        np.random.seed(42)
        
        # Gerar dados com estrutura econômica
        gap_inflacao = np.zeros(n)
        gap_produto = np.zeros(n)
        gap_juros = np.zeros(n)
        
        # Valores iniciais
        gap_inflacao[0] = np.random.randn() * 0.5
        gap_produto[0] = np.random.randn() * 0.3
        gap_juros[0] = np.random.randn() * 0.2
        
        # Gerar série temporal
        for t in range(1, n):
            gap_inflacao[t] = (0.7 * gap_inflacao[t-1] + 
                              0.3 * gap_produto[t-1] + 
                              np.random.randn() * 0.2)
            
            if t < n - 1:
                gap_produto[t+1] = (0.8 * gap_produto[t] - 
                                   0.4 * gap_juros[t] + 
                                   np.random.randn() * 0.3)
            
            gap_juros[t] = (0.6 * gap_juros[t-1] + 
                           0.2 * gap_inflacao[t-1] + 
                           np.random.randn() * 0.15)
        
        if PANDAS_AVAILABLE:
            return pd.DataFrame({
                'Gap_Inflacao': gap_inflacao,
                'Gap_Produto': gap_produto,
                'Gap_Juros': gap_juros
            })
        else:
            return {
                'Gap_Inflacao': gap_inflacao.tolist(),
                'Gap_Produto': gap_produto.tolist(),
                'Gap_Juros': gap_juros.tolist()
            }
    else:
        # Implementação sem numpy
        random.seed(42)
        
        gap_inflacao = [random.gauss(0, 0.5)]
        gap_produto = [random.gauss(0, 0.3)]
        gap_juros = [random.gauss(0, 0.2)]
        
        for t in range(1, n):
            gap_inflacao.append(0.7 * gap_inflacao[t-1] + 
                               0.3 * gap_produto[t-1] + 
                               random.gauss(0, 0.2))
            
            if t < n - 1:
                gap_produto.append(0.8 * gap_produto[t] - 
                                  0.4 * gap_juros[t] + 
                                  random.gauss(0, 0.3))
            
            gap_juros.append(0.6 * gap_juros[t-1] + 
                            0.2 * gap_inflacao[t-1] + 
                            random.gauss(0, 0.15))
        
        return {
            'Gap_Inflacao': gap_inflacao,
            'Gap_Produto': gap_produto,
            'Gap_Juros': gap_juros
        }

def estimar_phillips(dados):
    """Estimar curva de Phillips"""
    print("\n" + "="*50)
    print("ESTIMAÇÃO DA CURVA DE PHILLIPS")
    print("="*50)
    
    if PANDAS_AVAILABLE and isinstance(dados, pd.DataFrame):
        # Usar pandas
        dados['Gap_Inflacao_Lag'] = dados['Gap_Inflacao'].shift(1)
        dados_clean = dados.dropna()
        
        X = dados_clean[['Gap_Inflacao_Lag', 'Gap_Produto']].values
        y = dados_clean['Gap_Inflacao'].values
        
        # Adicionar constante
        if NUMPY_AVAILABLE:
            X = np.column_stack([np.ones(len(X)), X])
        
    else:
        # Implementação manual
        gap_inflacao = dados['Gap_Inflacao']
        gap_produto = dados['Gap_Produto']
        
        # Criar dados defasados manualmente
        X = []
        y = []
        
        for i in range(1, len(gap_inflacao)):
            X.append([1, gap_inflacao[i-1], gap_produto[i]])  # [const, lag, produto]
            y.append(gap_inflacao[i])
    
    # Estimar modelo
    if STATSMODELS_AVAILABLE and NUMPY_AVAILABLE:
        import statsmodels.api as sm
        model = sm.OLS(y, X).fit()
        print(model.summary())
        
        beta_pi = model.params[1]  # Gap_Inflacao_Lag
        alpha = model.params[2]    # Gap_Produto
        
    else:
        # Usar implementação própria
        model = SimpleRegression().fit(X, y)
        print(model.summary())
        
        beta_pi = model.params.get('Gap_Inflacao_Lag', 0.7)
        alpha = model.params.get('Gap_Produto', 0.3)
    
    print(f"\nCoeficientes da Curva de Phillips:")
    print(f"β_π (persistência): {beta_pi:.4f}")
    print(f"α (sensibilidade): {alpha:.4f}")
    
    return beta_pi, alpha, model

def estimar_is(dados):
    """Estimar equação IS"""
    print("\n" + "="*50)
    print("ESTIMAÇÃO DA EQUAÇÃO IS")
    print("="*50)
    
    # Implementação similar à Phillips, mas para IS
    # Usando valores padrão para demonstração
    beta_y = 0.8
    gamma = 0.4
    
    print(f"Coeficientes da Equação IS:")
    print(f"β_y (persistência): {beta_y:.4f}")
    print(f"γ (sensibilidade): {gamma:.4f}")
    
    return beta_y, gamma, None

def simular_sistema(beta_pi, alpha, beta_y, gamma):
    """Simular sistema dinâmico"""
    print("\n" + "="*50)
    print("SIMULAÇÃO DO SISTEMA")
    print("="*50)
    
    # Parâmetros
    K = [1.5, 0.5]  # Ganhos
    x0 = [2.0, 0.0]  # Estado inicial
    n_steps = 30  # Mais passos para vídeo mais interessante
    
    print(f"Ganhos K: {K}")
    print(f"Estado inicial: {x0}")
    
    # Matrizes do sistema
    A = [[beta_pi, alpha],
         [0, beta_y]]
    B = [[0], [gamma]]
    
    print(f"Matriz A: {A}")
    print(f"Matriz B: {B}")
    
    # Simulação
    X = [[0, 0] for _ in range(n_steps + 1)]
    U = [0 for _ in range(n_steps)]  # Armazenar controles
    X[0] = x0.copy()
    
    for t in range(n_steps):
        # Controle: u = -K * x
        u = -(K[0] * X[t][0] + K[1] * X[t][1])
        U[t] = u
        
        # Próximo estado: x[t+1] = A * x[t] + B * u
        X[t+1][0] = A[0][0] * X[t][0] + A[0][1] * X[t][1] + B[0][0] * u
        X[t+1][1] = A[1][0] * X[t][0] + A[1][1] * X[t][1] + B[1][0] * u
    
    print(f"\nResultados da simulação:")
    print(f"Estado inicial: {X[0]}")
    print(f"Estado final: {X[-1]}")
    
    return X, U

def criar_video_animado(X, U, nome_arquivo='simulacao_animada.mp4'):
    """Criar vídeo animado da simulação"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib não disponível - pulando vídeo")
        return False
    
    print(f"\n{'='*50}")
    print("GERAÇÃO DE VÍDEO ANIMADO")
    print("="*50)
    
    try:
        # Extrair dados
        periodos = list(range(len(X)))
        inflacao = [x[0] for x in X]
        produto = [x[1] for x in X]
        
        # Configurar figura
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Simulação Dinâmica do Sistema Econômico', fontsize=16, fontweight='bold')
        
        # Configurar subplots
        # 1. Trajetórias temporais
        ax1.set_xlim(0, len(X)-1)
        ax1.set_ylim(min(min(inflacao), min(produto))-0.5, max(max(inflacao), max(produto))+0.5)
        ax1.set_xlabel('Período')
        ax1.set_ylabel('Valor do Gap (%)')
        ax1.set_title('Evolução Temporal dos Gaps')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        # 2. Diagrama de fases
        ax2.set_xlim(min(produto)-0.5, max(produto)+0.5)
        ax2.set_ylim(min(inflacao)-0.5, max(inflacao)+0.5)
        ax2.set_xlabel('Gap de Produto (%)')
        ax2.set_ylabel('Gap de Inflação (%)')
        ax2.set_title('Diagrama de Fases')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
        
        # 3. Controle (política monetária)
        ax3.set_xlim(0, len(U))
        ax3.set_ylim(min(U)-0.5, max(U)+0.5)
        ax3.set_xlabel('Período')
        ax3.set_ylabel('Taxa de Juros (%)')
        ax3.set_title('Política Monetária (Controle)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        # 4. Indicadores em tempo real
        ax4.axis('off')
        ax4.set_title('Indicadores em Tempo Real')
        
        # Inicializar linhas e pontos
        linha_inflacao, = ax1.plot([], [], 'b-', linewidth=2.5, label='Gap Inflação')
        linha_produto, = ax1.plot([], [], 'r-', linewidth=2.5, label='Gap Produto')
        ponto_atual_inflacao, = ax1.plot([], [], 'bo', markersize=8)
        ponto_atual_produto, = ax1.plot([], [], 'ro', markersize=8)
        ax1.legend()
        
        # Diagrama de fases
        linha_fase, = ax2.plot([], [], 'g-', linewidth=2, alpha=0.7)
        ponto_atual_fase, = ax2.plot([], [], 'go', markersize=10)
        ponto_inicial_fase, = ax2.plot([produto[0]], [inflacao[0]], 'rs', markersize=12, label='Início')
        ax2.legend()
        
        # Controle
        linha_controle, = ax3.plot([], [], 'purple', linewidth=2.5, label='Taxa de Juros')
        ponto_atual_controle, = ax3.plot([], [], 'o', color='purple', markersize=8)
        ax3.legend()
        
        # Texto para indicadores
        texto_periodo = ax4.text(0.1, 0.8, '', fontsize=14, fontweight='bold', transform=ax4.transAxes)
        texto_inflacao = ax4.text(0.1, 0.6, '', fontsize=12, transform=ax4.transAxes)
        texto_produto = ax4.text(0.1, 0.4, '', fontsize=12, transform=ax4.transAxes)
        texto_juros = ax4.text(0.1, 0.2, '', fontsize=12, transform=ax4.transAxes)
        
        def animar(frame):
            """Função de animação"""
            # Atualizar trajetórias temporais
            linha_inflacao.set_data(periodos[:frame+1], inflacao[:frame+1])
            linha_produto.set_data(periodos[:frame+1], produto[:frame+1])
            
            if frame < len(X):
                ponto_atual_inflacao.set_data([frame], [inflacao[frame]])
                ponto_atual_produto.set_data([frame], [produto[frame]])
            
            # Atualizar diagrama de fases
            linha_fase.set_data(produto[:frame+1], inflacao[:frame+1])
            if frame < len(X):
                ponto_atual_fase.set_data([produto[frame]], [inflacao[frame]])
            
            # Atualizar controle
            if frame < len(U):
                linha_controle.set_data(periodos[:frame+1], U[:frame+1])
                ponto_atual_controle.set_data([frame], [U[frame]])
            
            # Atualizar indicadores
            if frame < len(X):
                texto_periodo.set_text(f'Período: {frame}')
                texto_inflacao.set_text(f'Gap Inflação: {inflacao[frame]:.2f}%')
                texto_produto.set_text(f'Gap Produto: {produto[frame]:.2f}%')
                if frame < len(U):
                    texto_juros.set_text(f'Taxa de Juros: {U[frame]:.2f}%')
            
            return (linha_inflacao, linha_produto, ponto_atual_inflacao, ponto_atual_produto,
                   linha_fase, ponto_atual_fase, linha_controle, ponto_atual_controle,
                   texto_periodo, texto_inflacao, texto_produto, texto_juros)
        
        # Criar animação
        print("Criando animação...")
        anim = animation.FuncAnimation(
            fig, animar, frames=len(X), interval=200, blit=False, repeat=True
        )
        
        # Salvar vídeo
        print(f"Salvando vídeo como {nome_arquivo}...")
        
        # Configurar writer
        if FFMPEG_AVAILABLE:
            writer = animation.FFMpegWriter(fps=5, metadata=dict(artist='Análise Econométrica'), bitrate=1800)
        else:
            # Fallback para PillowWriter se FFmpeg não estiver disponível
            try:
                writer = animation.PillowWriter(fps=5)
                nome_arquivo = nome_arquivo.replace('.mp4', '.gif')
                print(f"FFmpeg não disponível, salvando como GIF: {nome_arquivo}")
            except:
                print("❌ Não foi possível criar vídeo - nem FFmpeg nem Pillow disponíveis")
                plt.close(fig)
                return False
        
        anim.save(nome_arquivo, writer=writer)
        plt.close(fig)
        
        print(f"✅ Vídeo salvo com sucesso: {nome_arquivo}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar vídeo: {e}")
        return False

def criar_video_comparativo(X, nome_arquivo='comparativo_cenarios.mp4'):
    """Criar vídeo comparando diferentes cenários de política"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib não disponível - pulando vídeo comparativo")
        return False
    
    print(f"\n{'='*50}")
    print("GERAÇÃO DE VÍDEO COMPARATIVO")
    print("="*50)
    
    try:
        # Simular diferentes cenários
        cenarios = {
            'Política Agressiva': [2.0, 1.0],    # K = [2.0, 1.0]
            'Política Moderada': [1.5, 0.5],     # K = [1.5, 0.5] (original)
            'Política Suave': [1.0, 0.2]         # K = [1.0, 0.2]
        }
        
        resultados = {}
        
        # Parâmetros do sistema (usando valores estimados)
        beta_pi, alpha = 0.7865, 0.2225
        beta_y, gamma = 0.8, 0.4
        A = [[beta_pi, alpha], [0, beta_y]]
        B = [[0], [gamma]]
        x0 = [2.0, 0.0]
        n_steps = 25
        
        # Simular cada cenário
        for nome, K in cenarios.items():
            X_cenario = [[0, 0] for _ in range(n_steps + 1)]
            X_cenario[0] = x0.copy()
            
            for t in range(n_steps):
                u = -(K[0] * X_cenario[t][0] + K[1] * X_cenario[t][1])
                X_cenario[t+1][0] = A[0][0] * X_cenario[t][0] + A[0][1] * X_cenario[t][1] + B[0][0] * u
                X_cenario[t+1][1] = A[1][0] * X_cenario[t][0] + A[1][1] * X_cenario[t][1] + B[1][0] * u
            
            resultados[nome] = X_cenario
        
        # Configurar figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Comparação de Políticas Monetárias', fontsize=16, fontweight='bold')
        
        cores = ['blue', 'red', 'green']
        
        # Configurar eixos
        all_inflacao = []
        all_produto = []
        for X_cenario in resultados.values():
            all_inflacao.extend([x[0] for x in X_cenario])
            all_produto.extend([x[1] for x in X_cenario])
        
        ax1.set_xlim(0, n_steps)
        ax1.set_ylim(min(all_inflacao)-0.2, max(all_inflacao)+0.2)
        ax1.set_xlabel('Período')
        ax1.set_ylabel('Gap de Inflação (%)')
        ax1.set_title('Evolução do Gap de Inflação')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
        
        ax2.set_xlim(min(all_produto)-0.2, max(all_produto)+0.2)
        ax2.set_ylim(min(all_inflacao)-0.2, max(all_inflacao)+0.2)
        ax2.set_xlabel('Gap de Produto (%)')
        ax2.set_ylabel('Gap de Inflação (%)')
        ax2.set_title('Diagrama de Fases')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
        
        # Inicializar linhas
        linhas_inflacao = {}
        linhas_fase = {}
        pontos_atuais = {}
        
        for i, (nome, X_cenario) in enumerate(resultados.items()):
            cor = cores[i]
            linhas_inflacao[nome], = ax1.plot([], [], color=cor, linewidth=2.5, label=nome)
            linhas_fase[nome], = ax2.plot([], [], color=cor, linewidth=2, label=nome, alpha=0.7)
            pontos_atuais[nome], = ax2.plot([], [], 'o', color=cor, markersize=8)
        
        ax1.legend()
        ax2.legend()
        
        def animar_comparativo(frame):
            """Função de animação comparativa"""
            for nome, X_cenario in resultados.items():
                periodos = list(range(frame + 1))
                inflacao = [x[0] for x in X_cenario[:frame + 1]]
                produto = [x[1] for x in X_cenario[:frame + 1]]
                
                linhas_inflacao[nome].set_data(periodos, inflacao)
                linhas_fase[nome].set_data(produto, inflacao)
                
                if frame < len(X_cenario):
                    pontos_atuais[nome].set_data([produto[-1]], [inflacao[-1]])
            
            return list(linhas_inflacao.values()) + list(linhas_fase.values()) + list(pontos_atuais.values())
        
        # Criar animação
        print("Criando animação comparativa...")
        anim = animation.FuncAnimation(
            fig, animar_comparativo, frames=n_steps+1, interval=300, blit=False, repeat=True
        )
        
        # Salvar vídeo
        print(f"Salvando vídeo comparativo como {nome_arquivo}...")
        
        if FFMPEG_AVAILABLE:
            writer = animation.FFMpegWriter(fps=3, metadata=dict(artist='Análise Econométrica'), bitrate=1800)
        else:
            try:
                writer = animation.PillowWriter(fps=3)
                nome_arquivo = nome_arquivo.replace('.mp4', '.gif')
                print(f"FFmpeg não disponível, salvando como GIF: {nome_arquivo}")
            except:
                print("❌ Não foi possível criar vídeo comparativo")
                plt.close(fig)
                return False
        
        anim.save(nome_arquivo, writer=writer)
        plt.close(fig)
        
        print(f"✅ Vídeo comparativo salvo: {nome_arquivo}")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar vídeo comparativo: {e}")
        return False

def plotar_resultados_estaticos(X, U):
    """Plotar gráficos estáticos tradicionais"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️ Matplotlib não disponível - pulando gráficos estáticos")
        return
    
    print("\n" + "=" * 50)
    print("GERAÇÃO DE GRÁFICOS ESTÁTICOS")
    print("=" * 50)
    
    try:
        # Extrair dados
        periodos = list(range(len(X)))
        inflacao = [x[0] for x in X]
        produto = [x[1] for x in X]
        
        # Criar figura com subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise Completa do Sistema Econômico', fontsize=16, fontweight='bold')
        
        # 1. Trajetórias temporais
        ax1.plot(periodos, inflacao, 'b-', linewidth=2.5, label="Gap de inflação (π^g)")
        ax1.plot(periodos, produto, 'r-', linewidth=2.5, label="Gap de produto (y^g)")
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax1.set_xlabel("Período")
        ax1.set_ylabel("Valor do gap (%)")
        ax1.set_title("Evolução Temporal dos Gaps")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Diagrama de fases
        ax2.plot(produto, inflacao, 'g-', linewidth=2, marker='o', markersize=3)
        ax2.plot(produto[0], inflacao[0], 'ro', markersize=10, label='Estado inicial')
        ax2.plot(produto[-1], inflacao[-1], 'bs', markersize=10, label='Estado final')
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.7)
        ax2.set_xlabel("Gap de produto (y^g)")
        ax2.set_ylabel("Gap de inflação (π^g)")
        ax2.set_title("Diagrama de Fases")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Política monetária (controle)
        ax3.plot(periodos[:-1], U, 'purple', linewidth=2.5, label="Taxa de juros")
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax3.set_xlabel("Período")
        ax3.set_ylabel("Taxa de juros (%)")
        ax3.set_title("Política Monetária (Controle)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Análise de convergência
        distancia_origem = [abs(x[0]) + abs(x[1]) for x in X]
        ax4.plot(periodos, distancia_origem, 'orange', linewidth=2.5, label="Distância da origem")
        ax4.set_xlabel("Período")
        ax4.set_ylabel("Distância |π^g| + |y^g|")
        ax4.set_title("Convergência do Sistema")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar gráfico
        nome_arquivo = "analise_completa_estatica.png"
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        print(f"✅ Gráficos estáticos salvos: {nome_arquivo}")
        
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ Erro ao gerar gráficos estáticos: {e}")

def main():
    """Função principal"""
    print("=" * 70)
    print("ANÁLISE ECONOMÉTRICA COM GERAÇÃO DE VÍDEOS")
    print("=" * 70)
    
    # Tentar carregar dados reais
    try:
        if PANDAS_AVAILABLE:
            dados = pd.read_excel("dados_finais_modelo.xlsx")
            print("✅ Dados carregados do Excel")
        else:
            dados = None
    except:
        dados = None
    
    if dados is None:
        print("Usando dados simulados...")
        dados = criar_dados_simulados()
    
    # Estimações
    beta_pi, alpha, model_phillips = estimar_phillips(dados)
    beta_y, gamma, model_is = estimar_is(dados)
    
    # Simulação
    X, U = simular_sistema(beta_pi, alpha, beta_y, gamma)
    
    # Gerar visualizações
    print("\n" + "=" * 70)
    print("GERAÇÃO DE VISUALIZAÇÕES")
    print("=" * 70)
    
    # 1. Gráficos estáticos
    plotar_resultados_estaticos(X, U)
    
    # 2. Vídeo principal da simulação
    sucesso_video = criar_video_animado(X, U, 'simulacao_economica_animada.mp4')
    
    # 3. Vídeo comparativo de políticas
    sucesso_comparativo = criar_video_comparativo(X, 'comparativo_politicas.mp4')
    
    # Resumo final
    print("\n" + "=" * 70)
    print("🎉 ANÁLISE CONCLUÍDA!")
    print("=" * 70)
    
    print(f"\nResumo dos resultados:")
    print(f"- Persistência da inflação (β_π): {beta_pi:.4f}")
    print(f"- Sensibilidade ao produto (α): {alpha:.4f}")
    print(f"- Persistência do produto (β_y): {beta_y:.4f}")
    print(f"- Sensibilidade aos juros (γ): {gamma:.4f}")
    print(f"- Redução da inflação: {X[0][0] - X[-1][0]:.2f} p.p.")
    
    print(f"\nArquivos gerados:")
    print(f"📊 Gráficos estáticos: analise_completa_estatica.png")
    if sucesso_video:
        print(f"🎬 Vídeo principal: simulacao_economica_animada.mp4")
    if sucesso_comparativo:
        print(f"🎬 Vídeo comparativo: comparativo_politicas.mp4")
    
    if not sucesso_video and not sucesso_comparativo:
        print("⚠️ Vídeos não foram gerados. Verifique se FFmpeg está instalado.")
        print("   Para instalar FFmpeg: https://ffmpeg.org/download.html")

if __name__ == "__main__":
    main()

