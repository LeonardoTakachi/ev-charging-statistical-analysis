"""
2º Semestre - Challenge Sprint 3
Análise Estatística e Regressão Linear

Integrantes:
Daniel Vieira Santos - RM 573326
Giovane Salazar Fioravante - RM 570396
Gustavo Bitencourt Lopes - RM 568885
Leonardo Basile Takachi - RM 569066
"""

import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ARQUIVO_CSV = "ev_charging_patterns.csv"
ARQUIVO_GRAFICO = "grafico_regressao_linear.png"
COLUNA_NORMAL = "Energy Consumed (kWh)"
COLUNA_X_REGRESSAO = "Charging Duration (hours)"
COLUNA_Y_REGRESSAO = "Energy Consumed (kWh)"


def carregar_base(caminho):
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{caminho}' não encontrado.")
        return None
    except Exception as erro:
        print(f"[ERRO] Não foi possível carregar a base: {erro}")
        return None


def preparar_dados(dados):
    colunas = [COLUNA_NORMAL, COLUNA_X_REGRESSAO]
    return dados[colunas].copy().dropna()


def classificar_evento(probabilidade):
    if probabilidade < 0.10:
        return "raro"
    if probabilidade < 0.40:
        return "pouco provável"
    if probabilidade < 0.90:
        return "provável"
    return "quase certo"


def probabilidade_acima_mediana(dados):
    serie = dados[COLUNA_NORMAL]
    media = serie.mean()
    mediana = serie.median()
    desvio = serie.std()
    probabilidade = 1 - norm.cdf(mediana, loc=media, scale=desvio)

    print("\n" + "=" * 70)
    print("01) PROBABILIDADE ACIMA DA MEDIANA")
    print("=" * 70)
    print(f"Média: {media:.4f} kWh")
    print(f"Mediana: {mediana:.4f} kWh")
    print(f"Desvio padrão: {desvio:.4f} kWh")
    print(f"Probabilidade: {probabilidade * 100:.2f}%")
    print(f"Classificação: {classificar_evento(probabilidade)}")


def probabilidade_intervalo_media_2s(dados):
    serie = dados[COLUNA_NORMAL]
    media = serie.mean()
    desvio = serie.std()
    inferior = media - 2 * desvio
    superior = media + 2 * desvio
    probabilidade = (
        norm.cdf(superior, loc=media, scale=desvio)
        - norm.cdf(inferior, loc=media, scale=desvio)
    )

    print("\n" + "=" * 70)
    print("02) PROBABILIDADE NO INTERVALO MÉDIA ± 2s")
    print("=" * 70)
    print(f"Limite inferior: {inferior:.4f} kWh")
    print(f"Limite superior: {superior:.4f} kWh")
    print(f"Probabilidade: {probabilidade * 100:.2f}%")
    print(f"Classificação: {classificar_evento(probabilidade)}")


def regressao_linear(dados):
    X = dados[[COLUNA_X_REGRESSAO]]
    y = dados[COLUNA_Y_REGRESSAO]
    modelo = LinearRegression().fit(X, y)
    previsoes = modelo.predict(X)
    intercepto = modelo.intercept_
    coeficiente = modelo.coef_[0]
    r2 = r2_score(y, previsoes)

    print("\n" + "=" * 70)
    print("03) MODELAGEM COM REGRESSÃO LINEAR")
    print("=" * 70)
    print(
        f"{COLUNA_Y_REGRESSAO} = {intercepto:.4f} + "
        f"({coeficiente:.4f} × {COLUNA_X_REGRESSAO})"
    )
    print(f"R²: {r2:.6f}")

    return modelo, X, y, intercepto, coeficiente, r2


def gerar_grafico_regressao(modelo, X, y, intercepto, coeficiente, r2):
    x_ordenado = X.sort_values(by=COLUNA_X_REGRESSAO)
    y_reta = modelo.predict(x_ordenado)

    plt.figure(figsize=(10, 6))
    plt.scatter(X[COLUNA_X_REGRESSAO], y, alpha=0.5, label="Sessões observadas")
    plt.plot(
        x_ordenado[COLUNA_X_REGRESSAO], y_reta,
        linewidth=2, label="Reta de regressão"
    )
    plt.title("Regressão Linear: Duração da Recarga x Energia Consumida")
    plt.xlabel("Duração da Recarga (horas)")
    plt.ylabel("Energia Consumida (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.text(
        0.02, 0.97,
        f"y = {intercepto:.2f} + {coeficiente:.2f}x\nR² = {r2:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.tight_layout()
    plt.savefig(ARQUIVO_GRAFICO, dpi=300)
    plt.close()


def main():
    dados = carregar_base(ARQUIVO_CSV)
    if dados is None:
        return
    dados_limpos = preparar_dados(dados)
    probabilidade_acima_mediana(dados_limpos)
    probabilidade_intervalo_media_2s(dados_limpos)
    resultado = regressao_linear(dados_limpos)
    gerar_grafico_regressao(*resultado)


if __name__ == "__main__":
    main()
