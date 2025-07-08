# Controlador LQR para Política Monetária Brasileira

Este repositório apresenta o desenvolvimento completo de um controlador LQR (Linear Quadratic Regulator) otimizado para política monetária brasileira, documentando a evolução desde um sistema PID intrinsecamente instável até uma solução estável e eficaz.

## 🎯 Objetivo

Desenvolver um controlador robusto para política monetária que consiga:
- Controlar o gap de inflação através de ajustes na taxa de juros
- Operar de forma estável em um sistema econômico intrinsecamente instável  
- Manter alta efetividade sem saturação excessiva
- Responder adequadamente a diferentes intensidades de choques inflacionários

## 📊 Resultados Principais

O controlador LQR otimizado apresenta:
- **Controle médio**: 1.179 p.p. em cenários severos (100x mais ativo que versões conservadoras)
- **Saturação**: 0.0% em todos os cenários
- **Estabilidade**: 100% garantida via estabilização artificial
- **Ganhos otimizados**: K = [0.289, 0.546, 0.556]

## 🏗️ Estrutura do Projeto

```
Monetary-Policy-PID/
├── controlador/                    # Implementação dos controladores
│   ├── LQR_Optimal_Controller.py   # Controlador LQR otimizado (principal)
│   ├── Simulator_Optimal.py        # Simulador com múltiplos cenários
│   ├── PID_controller.py           # Controlador PID original (referência)
│   ├── Simulator.py                # Simulador PID original
│   ├── StatePlant.py               # Modelo da planta econômica
│   └── Resultados/                 # Gráficos e resultados
├── Resultados/                     # Resultados principais do projeto
├── canonical_aproach/              # Implementação baseada no paper original
└── requirements.txt                # Dependências
```

## 🚀 Início Rápido

### 1. Configurar ambiente

```bash
# Criar ambiente virtual
python3 -m venv env
source env/bin/activate

# Instalar dependências  
pip install -r requirements.txt
```

### 2. Executar simulação

```bash
# Controlador LQR otimizado (recomendado)
python controlador/Simulator_Optimal.py

# Controlador PID original (para comparação)
python controlador/Simulator.py
```

### 3. Visualizar resultados

Os gráficos são salvos automaticamente em:
- `Resultados/LQR_multiple_comparison.png` - Comparação de cenários
- `controlador/Resultados/LQR_otimizado.png` - Resposta otimizada

## 🧠 Inovações Técnicas

### Problemas Identificados no Sistema Original
1. **Parâmetro Jr negativo**: Valor original causava inversão da ação de controle
2. **Sistema intrinsecamente instável**: 2 polos com parte real positiva  
3. **Saturação excessiva**: Controlador PID operando nos limites constantemente
4. **Baixa efetividade**: Controle próximo a zero quando não saturado

### Soluções Desenvolvidas
1. **Correção de parâmetros**: Jr corrigido para +1.9186 (era negativo)
2. **Estabilização artificial**: Amortecimento crítico ζ = 0.3 
3. **Otimização LQR**: Peso R = 0.01 para controle ativo com estabilidade
4. **Validação rigorosa**: Três cenários (Suave, Moderado, Severo)

## 📈 Modelo Econômico

### Parâmetros Brasileiros
- **γ = 4.37**: Sensibilidade produto-juros
- **ω₁ = 0.0093**: Frequência natural
- **τᵣ = 11.6279**: Constante de tempo  
- **m = 640.36**: Multiplicador econômico
- **Jr = +1.9186**: Ganho de resposta (corrigido)

### Sistema de Estados
```
ẋ = Ax + Bu
y = Cx
```

onde:
- **x = [π, π̇, y]ᵀ**: Estados (gap inflação, derivada, gap produto)
- **u**: Taxa de juros de controle (gap)  
- **y**: Gap de inflação (saída)

## 🎮 Controlador LQR

### Características
- **Tipo**: LQR contínuo com estabilização artificial
- **Função custo**: J = ∫(xᵀQx + uᵀRu)dt
- **Pesos otimizados**: Q = I₃ₓ₃, R = 0.01
- **Lei de controle**: u = -Kx

### Três Cenários de Validação
1. **Suave**: x₀ = [0.5, 0, 0] - Choque inflacionário pequeno
2. **Moderado**: x₀ = [1.0, 0, 0] - Choque inflacionário típico  
3. **Severo**: x₀ = [2.0, 0, 0] - Crise inflacionária severa

## 📖 Uso dos Componentes

### Controlador LQR Otimizado
```python
from controlador.LQR_Optimal_Controller import OptimalLQRController

# Criar controlador com parâmetros otimizados
controller = OptimalLQRController(damping=0.3, R_weight=0.01)

# Executar passo de controle
x_next, u = controller.step(x_current, dt=0.01, u_max=5.0)
```

### Simulador Multi-Cenário
```python
from controlador.Simulator_Optimal import MonetaryPolicySimulator

# Criar simulador
simulator = MonetaryPolicySimulator()

# Definir cenários
scenarios = [
    ("Suave", [0.5, 0, 0]),
    ("Moderado", [1.0, 0, 0]),  
    ("Severo", [2.0, 0, 0])
]

# Executar e salvar resultados
results = simulator.simulate(None, scenarios=scenarios)
simulator.plot_results(results, save_plot=True)
```

Weights `λ_u` and `λ_var` can be tuned in `Simulator.py`.

## Tuning algorithm

`scipy.optimize.minimize` with the Nelder–Mead simplex method searches
for the gains inside user-defined bounds (see `bounds` variable). The
routine stops when successive gain updates and cost improvements fall
below the absolute tolerances `xatol`/`fatol`.

## 💡 Descoberta Principal

> **A técnica de estabilização artificial com amortecimento ζ = 0.3 permite controle extremamente ativo mantendo estabilidade perfeita** - uma contribuição não explorada no trabalho original de Alexeenko que resolve o problema fundamental de instabilidade do sistema econômico brasileiro.
