import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

df = pd.read_excel("../pesquisa/dados_cosolidados.xlsx")

# Corrigir possível separador decimal
df['Media Ponderada Selic'] = (
    df['Media Ponderada Selic']
    .astype(str)                          # garante texto para replace
    .str.replace(',', '.', regex=False)   # troca vírgula por ponto
    .astype(float)                        # converte para número
)

# Construir coluna de datas
df['Data'] = df['Ano'].astype(str) + "-T" + df['Trimestre'].astype(str)

# Gap de inflação
df['Gap_Inflacao'] = df['IPCA'] - df['Metas IPCA']

# Gap de produto
pib_ln = np.log(df['PIB dessazonalizado'])
_, pib_ln_tendencia = hpfilter(pib_ln, lamb=1600)
df['PIB_Potencial'] = np.exp(pib_ln_tendencia)
df['Gap_Produto'] = 100 * (df['PIB dessazonalizado'] / df['PIB_Potencial'] - 1)

# Gap de juros
taxa_neutra_nominal = 7.0
df['Gap_Juros'] = df['Media Ponderada Selic'] - taxa_neutra_nominal

df.to_excel("dados_finais_modelo.xlsx", index=False)
print("Arquivo 'dados_finais_modelo.xlsx' criado com sucesso!")
print(df.head())
