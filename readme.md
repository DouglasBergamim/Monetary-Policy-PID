# Manual do Usuário - Simulação do Modelo de Feedback de Estado

Este manual orienta a execução do código Python fornecido para simular a dinâmica do modelo de feedback de estado aplicado à política monetária brasileira.

## Pré-requisitos
- Python 3.8 ou superior (testado em 3.13)
- Bibliotecas: `numpy`, `matplotlib` (opcionalmente `pandas`, `statsmodels`)

## Como executar
1. Salve o código em um arquivo, por exemplo `simulacao.py`.
2. No terminal, navegue até a pasta onde o arquivo está salvo.
3. Execute o script com:
   ```bash
   python simulacao.py
   ```
4. O código irá:
   - Imprimir os resultados das simulações no terminal.
   - Gerar o gráfico `simulacao_python313.png` mostrando a trajetória dos gaps.

## Estrutura do código
- Define as matrizes A e B baseadas nos parâmetros estimados.
- Define o vetor de ganhos K.
- Simula a evolução dos estados (gaps de inflação e produto) por 20 períodos.
- Plota e salva o gráfico.

## Interpretação do gráfico
- Linha azul: gap de inflação ($\pi^g$) — mostra como a inflação retorna (ou não) para a meta após o choque inicial..
- Linha vermelha: gap de produto ($y^g$)— indica o desvio do PIB em relação ao potencial ao longo do tempo.
- Convergência indica estabilidade.
- O padrão de resposta permite analisar se o BC é "hawkish" (corrige rápido mas com volatilidade) ou "dovish" (mais lento mas suave).

## Personalização
- Altere `K=[1.5,0.5]` para testar outras políticas.
- Ajuste os parâmetros de A e B para simular outros cenários econômicos.

## Contato
Para dúvidas ou sugestões, entre em contato pelo e-mail informado no cabeçalho do paper.
