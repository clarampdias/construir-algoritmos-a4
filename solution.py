'''No terminal:
python -m pip install pandas numpy

Importação das bibliotecas necessárias'''
import pandas as pd
import numpy as np


def solution(arquivo):
    # 1. Leitura do arquivo CSV utilizando o Pandas
    df = pd.read_csv(arquivo)

    # 2. Seleção das colunas necessárias para a regressão
    # O identificador id_imovel é desconsiderado, pois não influencia o modelo.
    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    # 3. Construção da matriz X (variáveis preditoras)
    # A primeira coluna contém números 1 para representar o intercepto.
    # As outras colunas representam a área e a quantidade de quartos.
    X = np.column_stack((
        np.ones(len(area)),
        area,
        quartos
    ))

    # 4. Construção do vetor y (variável dependente)
    # Contém os preços reais de venda dos imóveis.
    y = np.array(preco)

    # 5. Cálculo dos coeficientes pela fórmula matricial:
    # beta = (X.T @ X)^-1 @ X.T @ y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    # 6. Retorno dos coeficientes: intercepto, área e quartos.
    # O resultado é convertido para uma lista Python.
    return beta.tolist()

# Fim do ficheiro solution.py

# Chamada da função e impressão do resultado no terminal
resultado = solution("imoveis_lar_perfeito.csv")
print(resultado)
