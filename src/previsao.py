"""Spec 07 — Previsão de partida + experimentos.

(1) Função ``prever_jogo`` que estima gols esperados e probabilidades V/E/D a partir do ELO
atual das seleções, usando o **modelo campeão** eleito na feature_06 (interface ``Preditor``).
(2) Experimentos que re-treinam o Poisson variando a recência e comparam o MAE no teste —
a lição central: mais sofisticação nem sempre melhora.

Gera as tabelas ``previsoes`` (os 72 jogos da Copa 2026) e ``experimentos_mae``.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import statsmodels.api as sm

import modelos
from db import get_engine, get_raw_connection
from preditor import montar_X, montar_features
from treino import CORTE, MODELS_DIR

DATA_REF = pd.Timestamp("2026-06-11")  # mesma âncora de recência da feature_03.
PESO_TORNEIO_COPA = 3                   # FIFA World Cup = nível 3.

DDL = """
DROP TABLE IF EXISTS previsoes;
CREATE TABLE previsoes (
    id                      bigint generated always as identity primary key,
    time_casa               text,
    time_visitante          text,
    gols_esperados_casa     double precision,
    gols_esperados_visitante double precision,
    prob_vitoria            double precision,
    prob_empate             double precision,
    prob_derrota            double precision
);

DROP TABLE IF EXISTS experimentos_mae;
CREATE TABLE experimentos_mae (
    id              bigint generated always as identity primary key,
    config          text,
    mae_casa        double precision,
    mae_visitante   double precision
);
"""


def carregar_modelos():
    """Carrega o ``Preditor`` do modelo campeão (definido por models/campeao.json)."""
    return modelos.carregar_campeao(MODELS_DIR)


def prever_jogo(time_casa, time_visitante, neutro, peso_torneio, *, elos, preditor):
    """Prediz um jogo a partir do ELO atual de cada seleção (peso_recencia=1.0, jogo no presente)."""
    feats = montar_features(elos[time_casa], elos[time_visitante], neutro, peso_torneio)
    lam_casa, lam_visit = preditor.lambdas(feats)
    p_vit, p_emp, p_der = preditor.probs_de_lambdas(lam_casa, lam_visit)
    return {
        "time_casa": time_casa,
        "time_visitante": time_visitante,
        "gols_esperados_casa": lam_casa,
        "gols_esperados_visitante": lam_visit,
        "prob_vitoria": p_vit,
        "prob_empate": p_emp,
        "prob_derrota": p_der,
    }


def gerar_previsoes(preditor) -> pd.DataFrame:
    eng = get_engine()
    elos = dict(pd.read_sql("SELECT selecao, elo FROM silver_elo_atual", eng).itertuples(index=False, name=None))
    copa = pd.read_sql("SELECT time_casa, time_visitante, neutro FROM silver_copa2026 ORDER BY data, id", eng)

    linhas = [
        prever_jogo(j.time_casa, j.time_visitante, j.neutro, PESO_TORNEIO_COPA,
                    elos=elos, preditor=preditor)
        for j in copa.itertuples(index=False)
    ]
    return pd.DataFrame(linhas)


def rodar_experimentos() -> pd.DataFrame:
    """Re-treina o Poisson variando a recência e mede o MAE no mesmo split temporal da feature_06."""
    gold = pd.read_sql("SELECT * FROM gold_atributos", get_engine(), parse_dates=["data"])
    idade_anos = (DATA_REF - gold["data"]).dt.days / 365.25

    # (config, meia-vida); None = recência desligada (peso_recencia constante 1.0).
    configs = [("sem_recencia", None), ("meia_vida_3", 3), ("meia_vida_5", 5), ("meia_vida_10", 10)]
    resultados = []

    for nome, meia_vida in configs:
        df = gold.copy()
        df["peso_recencia"] = 1.0 if meia_vida is None else 0.5 ** (idade_anos / meia_vida)

        treino = df[df["data"] < CORTE]
        teste = df[df["data"] >= CORTE]
        peso_amostra = treino["peso_torneio"] * treino["peso_recencia"]

        Xtr = montar_X(treino)
        fam = sm.families.Poisson()
        mc = sm.GLM(treino["gols_casa"], Xtr, family=fam, var_weights=peso_amostra).fit()
        mv = sm.GLM(treino["gols_visitante"], Xtr, family=fam, var_weights=peso_amostra).fit()

        Xte = montar_X(teste)
        mae_casa = float(np.mean(np.abs(mc.predict(Xte) - teste["gols_casa"].to_numpy())))
        mae_visit = float(np.mean(np.abs(mv.predict(Xte) - teste["gols_visitante"].to_numpy())))
        resultados.append({"config": nome, "mae_casa": mae_casa, "mae_visitante": mae_visit})

    return pd.DataFrame(resultados)


def gravar(previsoes: pd.DataFrame, experimentos: pd.DataFrame) -> None:
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            for df, tabela, cols in (
                (previsoes, "previsoes",
                 ["time_casa", "time_visitante", "gols_esperados_casa", "gols_esperados_visitante",
                  "prob_vitoria", "prob_empate", "prob_derrota"]),
                (experimentos, "experimentos_mae", ["config", "mae_casa", "mae_visitante"]),
            ):
                buf = io.StringIO()
                df[cols].to_csv(buf, index=False, header=False, na_rep="")
                buf.seek(0)
                cur.copy_expert(
                    f"COPY {tabela} ({', '.join(cols)}) FROM STDIN WITH (FORMAT csv, NULL '')", buf
                )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    print("Carregando modelo campeão...")
    preditor = carregar_modelos()
    print(f"  campeão: {preditor.nome}")

    print("Gerando previsões dos 72 jogos da Copa 2026...")
    previsoes = gerar_previsoes(preditor)

    print("Rodando experimentos (re-treino variando recência)...")
    experimentos = rodar_experimentos()

    gravar(previsoes, experimentos)

    print("\n" + "=" * 60)
    print("RELATÓRIO PREVISÃO + EXPERIMENTOS")
    print("=" * 60)
    print(f"  previsoes: {len(previsoes)} jogos da Copa 2026")
    soma = (previsoes["prob_vitoria"] + previsoes["prob_empate"] + previsoes["prob_derrota"])
    print(f"  soma das probabilidades: min={soma.min():.4f} max={soma.max():.4f} (≈1)")
    print("  exemplos:")
    for _, r in previsoes.head(3).iterrows():
        print(f"    {r['time_casa']} x {r['time_visitante']}: "
              f"λ={r['gols_esperados_casa']:.2f}-{r['gols_esperados_visitante']:.2f} "
              f"V/E/D={r['prob_vitoria']:.2f}/{r['prob_empate']:.2f}/{r['prob_derrota']:.2f}")
    print("\n  experimentos_mae (ordenado por mae_casa):")
    for _, r in experimentos.sort_values("mae_casa").iterrows():
        print(f"    {r['config']:<14} mae_casa={r['mae_casa']:.4f}  mae_visitante={r['mae_visitante']:.4f}")
    print("=" * 60)
    print("\n✓ Previsão + experimentos concluídos.")


if __name__ == "__main__":
    main()
