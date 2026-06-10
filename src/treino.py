"""Spec 06 — Treino + competição de modelos + eleição do campeão.

Treina três modelos candidatos sobre ``gold_atributos`` (Poisson GLM, Poisson+Dixon-Coles e
LightGBM Poisson), avalia todos no MESMO holdout temporal e elege o campeão pela métrica
**log-loss** (regra de pontuação própria; desempate por Brier). O campeão é persistido em
``models/`` (+ ``models/campeao.json``) e passa a alimentar a previsão e o Monte Carlo.

Saídas: ``comparacao_modelos`` (uma linha por modelo), ``metricas_validacao`` (o campeão) e
``docs/modelo_campeao.md`` (relatório técnico de qual venceu e por quê).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import modelos
import poisson
from db import get_engine, get_raw_connection
from preditor import montar_X  # re-exportado para previsao.py/monte_carlo.py

# montar_X é re-exportado por compatibilidade; o silenciador evita aviso de import não usado.
_ = montar_X

CORTE = pd.Timestamp("2024-01-01")  # treino < CORTE <= teste
MODELS_DIR = "models"
DOCS_DIR = "docs"
RELATORIO_MD = os.path.join(DOCS_DIR, "modelo_campeao.md")

# Nomes amigáveis para o relatório.
NOME_MODELO = {
    "poisson": "Poisson GLM",
    "dixon_coles": "Poisson + Dixon-Coles",
    "lgbm": "LightGBM (Poisson)",
}

DDL = """
DROP TABLE IF EXISTS comparacao_modelos;
CREATE TABLE comparacao_modelos (
    id              bigint generated always as identity primary key,
    modelo          text,
    mae_casa        double precision,
    mae_visitante   double precision,
    log_loss        double precision,
    brier           double precision,
    acuracia        double precision,
    eh_campeao      boolean
);

DROP TABLE IF EXISTS metricas_validacao;
CREATE TABLE metricas_validacao (
    id              bigint generated always as identity primary key,
    modelo          text,
    mae_casa        double precision,
    mae_visitante   double precision,
    log_loss        double precision,
    brier           double precision,
    acuracia        double precision
);
"""

MET_COLS = ["mae_casa", "mae_visitante", "log_loss", "brier", "acuracia"]


def avaliar(preditor, teste: pd.DataFrame) -> dict:
    """Métricas do modelo no holdout: MAE de gols, log-loss, Brier e acurácia V/E/D."""
    lam_casa, lam_visit = preditor.lambdas_lote(teste)
    gc = teste["gols_casa"].to_numpy()
    gv = teste["gols_visitante"].to_numpy()

    probs = np.array([preditor.probs_de_lambdas(lam_casa[i], lam_visit[i]) for i in range(len(lam_casa))])
    real = poisson.resultado_real(gc, gv)
    previsto = np.array(["VED"[int(np.argmax(p))] for p in probs])

    return {
        "mae_casa": float(np.mean(np.abs(lam_casa - gc))),
        "mae_visitante": float(np.mean(np.abs(lam_visit - gv))),
        "log_loss": poisson.log_loss_ved(probs, real),
        "brier": poisson.brier_ved(probs, real),
        "acuracia": float(np.mean(previsto == real)),
    }


def eleger_campeao(comp: pd.DataFrame) -> str:
    """Campeão = menor log-loss; desempate por Brier, depois MAE médio."""
    comp = comp.copy()
    comp["mae_medio"] = (comp["mae_casa"] + comp["mae_visitante"]) / 2
    melhor = comp.sort_values(["log_loss", "brier", "mae_medio"]).iloc[0]
    return str(melhor["modelo"])


def persistir_banco(comp: pd.DataFrame, campeao: str) -> None:
    conn = get_raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(DDL)
            for _, r in comp.iterrows():
                cur.execute(
                    "INSERT INTO comparacao_modelos (modelo, mae_casa, mae_visitante, log_loss, "
                    "brier, acuracia, eh_campeao) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (r["modelo"], r["mae_casa"], r["mae_visitante"], r["log_loss"],
                     r["brier"], r["acuracia"], bool(r["modelo"] == campeao)),
                )
            c = comp[comp["modelo"] == campeao].iloc[0]
            cur.execute(
                "INSERT INTO metricas_validacao (modelo, mae_casa, mae_visitante, log_loss, "
                "brier, acuracia) VALUES (%s, %s, %s, %s, %s, %s)",
                (c["modelo"], c["mae_casa"], c["mae_visitante"], c["log_loss"],
                 c["brier"], c["acuracia"]),
            )
        conn.commit()
    finally:
        conn.close()


def persistir_campeao(preditor, comp: pd.DataFrame, campeao: str) -> dict:
    """Salva os artefatos do campeão + ``models/campeao.json`` (família, rho, métricas)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    meta = preditor.salvar(MODELS_DIR)
    c = comp[comp["modelo"] == campeao].iloc[0]
    meta["metricas"] = {k: float(c[k]) for k in MET_COLS}
    with open(os.path.join(MODELS_DIR, modelos.CAMPEAO_JSON), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def gerar_relatorio_md(comp: pd.DataFrame, campeao: str, n_treino: int, n_teste: int) -> None:
    """Gera docs/modelo_campeao.md com a tabela de métricas e a justificativa técnica."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    ordenado = comp.sort_values("log_loss").reset_index(drop=True)
    c = comp[comp["modelo"] == campeao].iloc[0]
    base = comp[comp["modelo"] == "poisson"].iloc[0]
    ganho_ll = (base["log_loss"] - c["log_loss"]) / base["log_loss"] * 100 if base["log_loss"] else 0.0

    linhas = [
        "# Modelo campeão — competição de previsão de gols (Copa 2026)",
        "",
        f"Comparação no **holdout temporal** (treino `< {CORTE.date()}`: {n_treino:,} jogos | "
        f"teste `>= {CORTE.date()}`: {n_teste:,} jogos). Eleição por **log-loss** (menor é melhor); "
        "desempate por Brier.",
        "",
        "| Modelo | log-loss | Brier | MAE casa | MAE visit. | Acurácia V/E/D | Campeão |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in ordenado.iterrows():
        marca = "✅" if r["modelo"] == campeao else ""
        linhas.append(
            f"| {NOME_MODELO.get(r['modelo'], r['modelo'])} | {r['log_loss']:.4f} | {r['brier']:.4f} | "
            f"{r['mae_casa']:.4f} | {r['mae_visitante']:.4f} | {r['acuracia']*100:.1f}% | {marca} |"
        )

    linhas += [
        "",
        "## Por que este modelo venceu",
        "",
        f"O **{NOME_MODELO.get(campeao, campeao)}** obteve o menor log-loss "
        f"({c['log_loss']:.4f}), métrica que mede a *qualidade das probabilidades* — exatamente o "
        f"que o Monte Carlo consome ao simular o torneio. ",
    ]
    if campeao == "poisson":
        linhas.append(
            "Nem a correção de Dixon-Coles nem o LightGBM superaram o GLM linear no agregado: "
            "com a janela de dados disponível, a sofisticação extra não se converteu em "
            "probabilidades melhores — a lição central do projeto (*mais complexidade nem sempre "
            "ganha*)."
        )
    elif campeao == "dixon_coles":
        linhas.append(
            f"A correção de empates (ρ = {comp[comp['modelo']=='dixon_coles'].iloc[0].get('rho', float('nan'))}) "
            "melhorou a calibração dos placares baixos (0-0, 1-1), onde a Poisson independente "
            f"subestima empates — reduzindo o log-loss em {ganho_ll:.1f}% sobre o GLM puro, sem o "
            "risco de overfitting do boosting."
        )
    else:  # lgbm
        linhas.append(
            f"O boosting capturou não-linearidades e interações entre os atributos (ex.: efeito do "
            f"`dif_elo` modulado por mando/peso de torneio) que o GLM linear não representa, "
            f"reduzindo o log-loss em {ganho_ll:.1f}% sobre o baseline. Ainda assim, a margem é "
            "modesta — consistente com o baixo sinal/ruído do futebol."
        )

    linhas += [
        "",
        "### Métricas — o que cada uma diz",
        "- **log-loss** (primária): penaliza confiança alta em previsões erradas; sensível à "
        "calibração das probabilidades V/E/D.",
        "- **Brier**: erro quadrático das probabilidades; apoio à log-loss.",
        "- **MAE de gols**: erro médio do λ previsto vs. gols reais (mandante/visitante).",
        "- **Acurácia V/E/D**: fração de resultados acertados — intuitiva, mas ignora a "
        "qualidade da probabilidade, por isso não decide a eleição.",
        "",
        f"> Gerado automaticamente por `src/treino.py`. Fonte: tabela `comparacao_modelos`.",
        "",
    ]
    with open(RELATORIO_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def main() -> None:
    print("Lendo gold_atributos...")
    df = pd.read_sql("SELECT * FROM gold_atributos", get_engine(), parse_dates=["data"])
    treino = df[df["data"] < CORTE]
    teste = df[df["data"] >= CORTE]
    print(f"  treino: {len(treino):,} jogos (< {CORTE.date()}) | teste: {len(teste):,} jogos")

    print("Treinando e avaliando candidatos: poisson, dixon_coles, lgbm...")
    linhas = []
    preditores = {}
    for nome, fabrica in modelos.FABRICAS.items():
        pred = fabrica(treino)
        preditores[nome] = pred
        m = avaliar(pred, teste)
        m["modelo"] = nome
        if nome == "dixon_coles":
            m["rho"] = round(pred.rho, 5)
        linhas.append(m)

    comp = pd.DataFrame(linhas)
    campeao = eleger_campeao(comp)

    meta = persistir_campeao(preditores[campeao], comp, campeao)
    persistir_banco(comp, campeao)
    gerar_relatorio_md(comp, campeao, len(treino), len(teste))

    print("\n" + "=" * 72)
    print("COMPETIÇÃO DE MODELOS — holdout temporal")
    print("=" * 72)
    print(f"  {'modelo':<22}{'log_loss':>10}{'brier':>9}{'mae_casa':>10}{'mae_vis':>9}{'acur':>8}")
    for _, r in comp.sort_values("log_loss").iterrows():
        marca = "  <== CAMPEÃO" if r["modelo"] == campeao else ""
        print(f"  {NOME_MODELO.get(r['modelo'], r['modelo']):<22}"
              f"{r['log_loss']:>10.4f}{r['brier']:>9.4f}{r['mae_casa']:>10.4f}"
              f"{r['mae_visitante']:>9.4f}{r['acuracia']*100:>7.1f}%{marca}")
    print("=" * 72)
    print(f"  Campeão: {NOME_MODELO.get(campeao, campeao)} (família={meta['familia']})")
    print(f"  Artefatos + campeao.json em {MODELS_DIR}/  |  relatório em {RELATORIO_MD}")
    print("\n✓ Competição concluída.")


if __name__ == "__main__":
    main()
