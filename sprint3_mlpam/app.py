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
    """Carrega a base de dados utilizada na análise."""
    try:
        return pd.read_csv(caminho)
    except FileNotFoundError:
        print(f"[ERRO] Arquivo '{caminho}' não encontrado.")
        return None
    except Exception as erro:
        print(f"[ERRO] Não foi possível carregar a base: {erro}")
        return None


def preparar_dados(dados):
    """
    Seleciona as variáveis utilizadas e remove registros
    com valores ausentes nessas colunas.
    """
    colunas_utilizadas = [
        COLUNA_NORMAL,
        COLUNA_X_REGRESSAO
    ]

    dados_limpos = dados[colunas_utilizadas].copy()
    dados_limpos = dados_limpos.dropna()

    return dados_limpos


def classificar_evento(probabilidade):
    """Classifica o evento de acordo com sua probabilidade."""
    if probabilidade < 0.10:
        return "raro"
    elif probabilidade < 0.40:
        return "pouco provável"
    elif probabilidade < 0.90:
        return "provável"
    else:
        return "quase certo"


def probabilidade_acima_mediana(dados):
    """
    Calcula a mediana e a probabilidade de observar
    um valor acima dela, assumindo Distribuição Normal.
    """
    serie = dados[COLUNA_NORMAL]

    media = serie.mean()
    mediana = serie.median()
    desvio_padrao = serie.std()

    probabilidade = 1 - norm.cdf(
        mediana,
        loc=media,
        scale=desvio_padrao
    )

    classificacao = classificar_evento(probabilidade)

    print("\n" + "=" * 70)
    print("01) PROBABILIDADE ACIMA DA MEDIANA")
    print("=" * 70)

    print(f"Variável analisada: {COLUNA_NORMAL}")
    print(f"Média:              {media:.4f} kWh")
    print(f"Mediana:            {mediana:.4f} kWh")
    print(f"Desvio padrão:      {desvio_padrao:.4f} kWh")

    print("\nCálculo:")
    print(f"P(X > {mediana:.4f})")
    print(f"Probabilidade:       {probabilidade:.6f}")
    print(f"Percentual:          {probabilidade * 100:.2f}%")
    print(f"Classificação:       {classificacao}")

    print("\nInterpretação:")
    print(
        f"Assumindo que {COLUNA_NORMAL} segue uma Distribuição Normal, "
        f"a probabilidade de uma sessão apresentar consumo de energia "
        f"acima da mediana ({mediana:.2f} kWh) é de aproximadamente "
        f"{probabilidade * 100:.2f}%."
    )

    return {
        "media": media,
        "mediana": mediana,
        "desvio_padrao": desvio_padrao,
        "probabilidade": probabilidade,
        "classificacao": classificacao
    }


def probabilidade_intervalo_media_2s(dados):
    """
    Calcula a probabilidade de a variável estar dentro do
    intervalo média ± 2 desvios padrão.
    """
    serie = dados[COLUNA_NORMAL]

    media = serie.mean()
    desvio_padrao = serie.std()

    limite_inferior = media - 2 * desvio_padrao
    limite_superior = media + 2 * desvio_padrao

    probabilidade = (
        norm.cdf(
            limite_superior,
            loc=media,
            scale=desvio_padrao
        )
        - norm.cdf(
            limite_inferior,
            loc=media,
            scale=desvio_padrao
        )
    )

    classificacao = classificar_evento(probabilidade)

    print("\n" + "=" * 70)
    print("02) PROBABILIDADE NO INTERVALO MÉDIA ± 2s")
    print("=" * 70)

    print(f"Variável analisada: {COLUNA_NORMAL}")
    print(f"Média:              {media:.4f} kWh")
    print(f"Desvio padrão:      {desvio_padrao:.4f} kWh")

    print("\nIntervalo:")
    print(f"Limite inferior:    {limite_inferior:.4f} kWh")
    print(f"Limite superior:    {limite_superior:.4f} kWh")

    print("\nCálculo:")
    print(
        f"P({limite_inferior:.4f} < X < {limite_superior:.4f})"
    )
    print(f"Probabilidade:       {probabilidade:.6f}")
    print(f"Percentual:          {probabilidade * 100:.2f}%")
    print(f"Classificação:       {classificacao}")

    print("\nInterpretação:")
    print(
        f"Assumindo uma Distribuição Normal, a probabilidade de o consumo "
        f"de energia estar entre {limite_inferior:.2f} kWh e "
        f"{limite_superior:.2f} kWh é de aproximadamente "
        f"{probabilidade * 100:.2f}%."
    )

    return {
        "media": media,
        "desvio_padrao": desvio_padrao,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "probabilidade": probabilidade,
        "classificacao": classificacao
    }


def regressao_linear(dados):
    """
    Ajusta um modelo de Regressão Linear simples utilizando
    a duração da recarga para estimar a energia consumida.
    """
    X = dados[[COLUNA_X_REGRESSAO]]
    y = dados[COLUNA_Y_REGRESSAO]

    modelo = LinearRegression()
    modelo.fit(X, y)

    previsoes = modelo.predict(X)

    intercepto = modelo.intercept_
    coeficiente = modelo.coef_[0]
    r2 = r2_score(y, previsoes)

    print("\n" + "=" * 70)
    print("03) MODELAGEM COM REGRESSÃO LINEAR")
    print("=" * 70)

    print(f"Variável independente (X): {COLUNA_X_REGRESSAO}")
    print(f"Variável dependente (y):   {COLUNA_Y_REGRESSAO}")

    print("\nEquação da reta ajustada:")
    print(
        f"{COLUNA_Y_REGRESSAO} = "
        f"{intercepto:.4f} + "
        f"({coeficiente:.4f} × {COLUNA_X_REGRESSAO})"
    )

    print("\nCoeficientes:")
    print(f"Intercepto:          {intercepto:.4f}")
    print(f"Coeficiente angular: {coeficiente:.4f}")
    print(f"R²:                  {r2:.6f}")

    print("\nInterpretação do intercepto:")
    print(
        f"O intercepto de {intercepto:.2f} indica que, segundo o modelo, "
        f"quando a duração da recarga é igual a 0 hora, o consumo estimado "
        f"seria de aproximadamente {intercepto:.2f} kWh."
    )

    print("\nInterpretação do coeficiente angular:")
    print(
        f"O coeficiente angular de {coeficiente:.4f} indica que, para cada "
        f"hora adicional de duração da recarga, o modelo estima uma variação "
        f"média de aproximadamente {coeficiente:.2f} kWh no consumo de energia."
    )

    print("\nInterpretação do R²:")
    print(
        f"O R² de {r2:.6f}, ou aproximadamente {r2 * 100:.2f}%, mostra que "
        f"a duração da recarga isoladamente explica apenas uma pequena parcela "
        f"da variação observada no consumo de energia."
    )
    print(
        "Assim, para um modelo de aprendizado de máquina com maior poder "
        "preditivo, seria necessário considerar outras variáveis."
    )

    return {
        "modelo": modelo,
        "X": X,
        "y": y,
        "previsoes": previsoes,
        "intercepto": intercepto,
        "coeficiente": coeficiente,
        "r2": r2
    }


def gerar_grafico_regressao(
    modelo,
    X,
    y,
    intercepto,
    coeficiente,
    r2,
    caminho_saida
):
    """Gera e salva o gráfico da reta de Regressão Linear."""
    x_ordenado = X.sort_values(by=COLUNA_X_REGRESSAO)
    y_reta = modelo.predict(x_ordenado)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        X[COLUNA_X_REGRESSAO],
        y,
        alpha=0.5,
        label="Sessões observadas"
    )

    plt.plot(
        x_ordenado[COLUNA_X_REGRESSAO],
        y_reta,
        linewidth=2,
        label="Reta de regressão"
    )

    plt.title(
        "Regressão Linear: Duração da Recarga x Energia Consumida"
    )
    plt.xlabel("Duração da Recarga (horas)")
    plt.ylabel("Energia Consumida (kWh)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.text(
        0.02,
        0.97,
        (
            f"y = {intercepto:.2f} + {coeficiente:.2f}x\n"
            f"R² = {r2:.4f}"
        ),
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("GRÁFICO DA REGRESSÃO LINEAR")
    print("=" * 70)
    print(f"Gráfico salvo em: {caminho_saida}")


def main():
    dados = carregar_base(ARQUIVO_CSV)

    if dados is None:
        return

    dados_limpos = preparar_dados(dados)

    print("\n" + "=" * 70)
    print("CHALLENGE SPRINT 3 - ANÁLISE ESTATÍSTICA")
    print("=" * 70)
    print(f"Registros utilizados: {len(dados_limpos)}")

    probabilidade_acima_mediana(dados_limpos)
    probabilidade_intervalo_media_2s(dados_limpos)

    resultado_regressao = regressao_linear(dados_limpos)

    gerar_grafico_regressao(
        resultado_regressao["modelo"],
        resultado_regressao["X"],
        resultado_regressao["y"],
        resultado_regressao["intercepto"],
        resultado_regressao["coeficiente"],
        resultado_regressao["r2"],
        ARQUIVO_GRAFICO
    )


if __name__ == "__main__":
    main()
