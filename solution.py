'''
Importação das bibliotecas necessárias
python -m pip install pandas numpy

'''
import pandas as pd
import numpy as np


def solution(arquivo):
    """Retorna [intercepto, coeficiente_area, coeficiente_quartos]."""
    # 1. Leitura do arquivo CSV (id_imovel e desconsiderado)
    df = pd.read_csv(arquivo)

    # 2. Selecao das colunas necessarias para a regressao
    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    # 3. Matriz X: coluna de uns (intercepto), area e quartos
    X = np.column_stack((np.ones(len(area)), area, quartos))

    # 4. Vetor y: precos de venda
    y = np.array(preco)

    # 5. Coeficientes pela formula matricial: beta = (X.T @ X)^-1 @ X.T @ y
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    # 6. Retorno como lista Python: [beta0, beta1, beta2]
    return beta.tolist()


def formatar(valor):
    """Formata no padrao brasileiro: 8265.26 -> 8.265,26"""
    texto = f"{valor:,.2f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


if __name__ == "__main__":
    b0, b1, b2 = solution("imoveis_lar_perfeito.csv")

    print("=" * 55)
    print("  REGRESSAO LINEAR MULTIPLA - IMOBILIARIA LAR PERFEITO")
    print("=" * 55)
    print(f"  Intercepto (beta0): {formatar(b0):>14}")
    print(f"  Area       (beta1): {formatar(b1):>14}  por m2")
    print(f"  Quartos    (beta2): {formatar(b2):>14}  por quarto")
    print("-" * 55)
    print("  EQUACAO DO MODELO:")
    print(f"  Preco = {formatar(b0)} + {formatar(b1)} * Area + {formatar(b2)} * Quartos")
    print("=" * 55)
