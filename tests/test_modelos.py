"""Testes da competição de modelos: treino, avaliação, eleição e persistência do campeão."""

import json
import os

import numpy as np
import pytest

import modelos
import treino


@pytest.fixture(scope="module")
def split(gold_sintetico):
    tr = gold_sintetico[gold_sintetico["data"] < treino.CORTE]
    te = gold_sintetico[gold_sintetico["data"] >= treino.CORTE]
    return tr, te


@pytest.fixture(scope="module")
def preditores(split):
    tr, _ = split
    return {nome: fab(tr) for nome, fab in modelos.FABRICAS.items()}


def test_fabricas_cobrem_as_tres_familias():
    assert set(modelos.FABRICAS) == {"poisson", "dixon_coles", "lgbm"}


@pytest.mark.parametrize("nome", ["poisson", "dixon_coles", "lgbm"])
def test_probs_somam_um(preditores, nome, feats):
    pv, pe, pd_ = preditores[nome].probs_ved(feats)
    assert pv + pe + pd_ == pytest.approx(1.0, abs=1e-6)
    assert min(pv, pe, pd_) >= 0.0


@pytest.mark.parametrize("nome", ["poisson", "dixon_coles", "lgbm"])
def test_lambdas_positivos(preditores, nome, feats):
    lc, lv = preditores[nome].lambdas(feats)
    assert lc > 0 and lv > 0


@pytest.mark.parametrize("nome", ["poisson", "dixon_coles", "lgbm"])
def test_amostrador_placar_valido(preditores, nome, feats):
    np.random.seed(0)
    sorteia = preditores[nome].amostrador(feats)
    for _ in range(20):
        gc, gv = sorteia()
        assert gc >= 0 and gv >= 0
        assert isinstance(gc, (int, np.integer)) and isinstance(gv, (int, np.integer))


def test_avaliar_retorna_metricas_finitas(preditores, split):
    _, te = split
    m = treino.avaliar(preditores["poisson"], te)
    for chave in ("mae_casa", "mae_visitante", "log_loss", "brier", "acuracia"):
        assert np.isfinite(m[chave])
    assert 0.0 <= m["acuracia"] <= 1.0


def test_eleger_campeao_escolhe_menor_log_loss():
    import pandas as pd
    comp = pd.DataFrame([
        {"modelo": "poisson", "log_loss": 1.05, "brier": 0.64, "mae_casa": 1.0, "mae_visitante": 0.8},
        {"modelo": "lgbm", "log_loss": 0.99, "brier": 0.61, "mae_casa": 1.0, "mae_visitante": 0.8},
        {"modelo": "dixon_coles", "log_loss": 1.04, "brier": 0.63, "mae_casa": 1.0, "mae_visitante": 0.8},
    ])
    assert treino.eleger_campeao(comp) == "lgbm"


def test_dixon_coles_ajusta_rho(preditores):
    rho = preditores["dixon_coles"].rho
    assert np.isfinite(rho)
    assert -0.2 <= rho <= 0.2


@pytest.mark.parametrize("nome", ["poisson", "dixon_coles", "lgbm"])
def test_roundtrip_salvar_carregar(preditores, nome, feats, tmp_path):
    pred = preditores[nome]
    meta = pred.salvar(str(tmp_path))
    meta["familia"] = pred.nome
    with open(os.path.join(tmp_path, modelos.CAMPEAO_JSON), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    carregado = modelos.carregar_campeao(str(tmp_path))
    assert carregado.nome == pred.nome
    assert np.allclose(pred.lambdas(feats), carregado.lambdas(feats), rtol=1e-6)
