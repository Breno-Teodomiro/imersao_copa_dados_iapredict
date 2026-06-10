"""Os modelos concorrentes da competição (feature 06) e como persistir/recarregar o campeão.

Três famílias, todas embrulhadas na interface ``Preditor`` de ``preditor.py``:

- **poisson**      — dois GLM Poisson lineares (baseline do projeto).
- **dixon_coles**  — os mesmos λ do GLM + correção ``rho`` de empates (Dixon-Coles).
- **lgbm**         — dois LightGBM com ``objective="poisson"`` (não-linear, sobre os 6 atributos).

A comparação justa usa exatamente o mesmo treino, o mesmo peso de amostra
(``peso_torneio × peso_recencia``) e os mesmos 6 atributos — isolando o efeito de cada
ingrediente (correção de empate; não-linearidade).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar

import poisson
from preditor import ATRIBUTOS, Preditor, montar_X

CAMPEAO_JSON = "campeao.json"


# --------------------------------------------------------------------------- #
# GLM Poisson (compartilhado por poisson e dixon_coles)
# --------------------------------------------------------------------------- #
def _peso_amostra(treino: pd.DataFrame) -> pd.Series:
    return treino["peso_torneio"] * treino["peso_recencia"]


def _fit_glm(treino: pd.DataFrame):
    """Dois GLM Poisson (gols do mandante e do visitante), ponderados por torneio × recência."""
    X = montar_X(treino)
    w = _peso_amostra(treino)
    fam = sm.families.Poisson()
    mc = sm.GLM(treino["gols_casa"], X, family=fam, var_weights=w).fit()
    mv = sm.GLM(treino["gols_visitante"], X, family=fam, var_weights=w).fit()
    return mc, mv


class PreditorPoisson(Preditor):
    nome = "poisson"

    def __init__(self, modelo_casa, modelo_visit):
        self.mc = modelo_casa
        self.mv = modelo_visit

    def lambdas(self, feats):
        X = montar_X(pd.DataFrame([feats]))
        return float(self.mc.predict(X)[0]), float(self.mv.predict(X)[0])

    def lambdas_lote(self, df):
        X = montar_X(df)
        return np.asarray(self.mc.predict(X)), np.asarray(self.mv.predict(X))

    def salvar(self, models_dir):
        self.mc.save(os.path.join(models_dir, "modelo_poisson_casa.pkl"))
        self.mv.save(os.path.join(models_dir, "modelo_poisson_visitante.pkl"))
        return {"familia": self.nome, "rho": None}


# --------------------------------------------------------------------------- #
# Dixon-Coles: GLM + ajuste de rho por máxima verossimilhança nos placares baixos
# --------------------------------------------------------------------------- #
def _tau(x, y, lc, lv, rho):
    """Fator de correção de Dixon-Coles (vetorizado); 1 fora dos placares baixos."""
    t = np.ones_like(lc, dtype=float)
    t = np.where((x == 0) & (y == 0), 1.0 - lc * lv * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lc * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + lv * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return t


def _ajustar_rho(lc, lv, gc, gv) -> float:
    """Ajusta ``rho`` maximizando a log-verossimilhança de Dixon-Coles no treino.

    Só os jogos com ambos os placares ≤ 1 contribuem (fora deles τ = 1, log τ = 0).
    """
    mask = (gc <= 1) & (gv <= 1)
    lc_m, lv_m, gc_m, gv_m = lc[mask], lv[mask], gc[mask], gv[mask]

    def neg_ll(rho):
        tau = _tau(gc_m, gv_m, lc_m, lv_m, rho)
        if np.any(tau <= 0):
            return 1e12
        return -np.sum(np.log(tau))

    res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


class PreditorDixonColes(PreditorPoisson):
    nome = "dixon_coles"

    def __init__(self, modelo_casa, modelo_visit, rho):
        super().__init__(modelo_casa, modelo_visit)
        self.rho = float(rho)

    def probs_de_lambdas(self, lam_casa, lam_visit):
        return poisson.probabilidades_da_matriz(
            poisson.matriz_dixon_coles(lam_casa, lam_visit, self.rho)
        )

    def amostrador(self, feats):
        lam_casa, lam_visit = self.lambdas(feats)
        flat = poisson.matriz_dixon_coles(lam_casa, lam_visit, self.rho).ravel()
        flat = flat / flat.sum()
        n = poisson.MAX_GOLS + 1
        return lambda: divmod(int(np.random.choice(flat.size, p=flat)), n)

    def salvar(self, models_dir):
        meta = super().salvar(models_dir)  # reusa os mesmos .pkl do GLM
        meta.update({"familia": self.nome, "rho": self.rho})
        return meta


# --------------------------------------------------------------------------- #
# LightGBM com objetivo Poisson
# --------------------------------------------------------------------------- #
def _X_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    X = df[ATRIBUTOS].copy()
    X["neutro"] = X["neutro"].astype(int)
    return X


class PreditorLGBM(Preditor):
    nome = "lgbm"

    def __init__(self, modelo_casa, modelo_visit):
        self.mc = modelo_casa
        self.mv = modelo_visit

    def lambdas(self, feats):
        X = _X_lgbm(pd.DataFrame([feats]))
        return float(self.mc.predict(X)[0]), float(self.mv.predict(X)[0])

    def lambdas_lote(self, df):
        X = _X_lgbm(df)
        return np.asarray(self.mc.predict(X)), np.asarray(self.mv.predict(X))

    def salvar(self, models_dir):
        self.mc.save_model(os.path.join(models_dir, "modelo_lgbm_casa.txt"))
        self.mv.save_model(os.path.join(models_dir, "modelo_lgbm_visitante.txt"))
        return {"familia": self.nome, "rho": None}


# --------------------------------------------------------------------------- #
# Fábricas de treino
# --------------------------------------------------------------------------- #
def treinar_poisson(treino: pd.DataFrame) -> PreditorPoisson:
    return PreditorPoisson(*_fit_glm(treino))


def treinar_dixon_coles(treino: pd.DataFrame) -> PreditorDixonColes:
    mc, mv = _fit_glm(treino)
    X = montar_X(treino)
    lc = np.asarray(mc.predict(X))
    lv = np.asarray(mv.predict(X))
    rho = _ajustar_rho(lc, lv, treino["gols_casa"].to_numpy(), treino["gols_visitante"].to_numpy())
    return PreditorDixonColes(mc, mv, rho)


def treinar_lgbm(treino: pd.DataFrame) -> PreditorLGBM:
    import lightgbm as lgb  # API nativa (Booster) — não exige scikit-learn

    X = _X_lgbm(treino)
    w = _peso_amostra(treino).to_numpy()
    params = {
        "objective": "poisson", "learning_rate": 0.05, "num_leaves": 31,
        "min_child_samples": 50, "bagging_fraction": 0.8, "bagging_freq": 1,
        "feature_fraction": 0.8, "seed": 42, "num_threads": 0, "verbose": -1,
    }
    mc = lgb.train(params, lgb.Dataset(X, label=treino["gols_casa"], weight=w), num_boost_round=400)
    mv = lgb.train(params, lgb.Dataset(X, label=treino["gols_visitante"], weight=w), num_boost_round=400)
    return PreditorLGBM(mc, mv)


FABRICAS = {
    "poisson": treinar_poisson,
    "dixon_coles": treinar_dixon_coles,
    "lgbm": treinar_lgbm,
}


# --------------------------------------------------------------------------- #
# Carregar o campeão persistido
# --------------------------------------------------------------------------- #
def carregar_campeao(models_dir: str) -> Preditor:
    """Reconstrói o ``Preditor`` campeão a partir de ``models/campeao.json`` + artefatos."""
    with open(os.path.join(models_dir, CAMPEAO_JSON), encoding="utf-8") as f:
        meta = json.load(f)

    familia = meta["familia"]
    if familia in ("poisson", "dixon_coles"):
        mc = sm.load(os.path.join(models_dir, "modelo_poisson_casa.pkl"))
        mv = sm.load(os.path.join(models_dir, "modelo_poisson_visitante.pkl"))
        if familia == "poisson":
            return PreditorPoisson(mc, mv)
        return PreditorDixonColes(mc, mv, meta["rho"])

    if familia == "lgbm":
        import lightgbm as lgb

        mc = lgb.Booster(model_file=os.path.join(models_dir, "modelo_lgbm_casa.txt"))
        mv = lgb.Booster(model_file=os.path.join(models_dir, "modelo_lgbm_visitante.txt"))
        return PreditorLGBM(mc, mv)

    raise ValueError(f"família de modelo desconhecida em campeao.json: {familia!r}")
