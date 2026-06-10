"""Testes da matemática de placar: matriz Poisson, correção Dixon-Coles e métricas."""

import numpy as np
import pytest

import poisson


def test_matriz_placares_soma_um():
    m = poisson.matriz_placares(1.4, 1.1)
    assert m.shape == (poisson.MAX_GOLS + 1, poisson.MAX_GOLS + 1)
    assert m.sum() == pytest.approx(1.0, abs=1e-6)


def test_probabilidades_resultado_somam_um():
    assert sum(poisson.probabilidades_resultado(1.4, 1.1)) == pytest.approx(1.0, abs=1e-9)


def test_dixon_coles_renormalizada():
    m = poisson.matriz_dixon_coles(1.3, 1.1, -0.05)
    assert m.sum() == pytest.approx(1.0, abs=1e-9)


def test_dixon_coles_rho_negativo_aumenta_empate():
    emp_poi = np.trace(poisson.matriz_placares(1.3, 1.1))
    emp_dc = np.trace(poisson.matriz_dixon_coles(1.3, 1.1, -0.05))
    assert emp_dc > emp_poi


def test_dixon_coles_rho_zero_igual_poisson():
    m_poi = poisson.matriz_placares(1.3, 1.1)
    m_dc = poisson.matriz_dixon_coles(1.3, 1.1, 0.0)
    assert np.allclose(m_poi, m_dc, atol=1e-9)


def test_log_loss_previsao_perfeita_e_uniforme():
    reais = np.array(["V", "E", "D", "V"])
    perfeito = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    uniforme = np.full((4, 3), 1 / 3)
    assert poisson.log_loss_ved(perfeito, reais) == pytest.approx(0.0, abs=1e-9)
    assert poisson.log_loss_ved(uniforme, reais) == pytest.approx(np.log(3), abs=1e-9)


def test_brier_perfeito_zero_e_limite():
    reais = np.array(["V", "D"])
    perfeito = np.array([[1, 0, 0], [0, 0, 1]], float)
    pessimo = np.array([[0, 0, 1], [1, 0, 0]], float)
    assert poisson.brier_ved(perfeito, reais) == pytest.approx(0.0, abs=1e-9)
    assert poisson.brier_ved(pessimo, reais) == pytest.approx(2.0, abs=1e-9)


def test_resultado_real():
    gc = np.array([2, 1, 0])
    gv = np.array([0, 1, 3])
    assert list(poisson.resultado_real(gc, gv)) == ["V", "E", "D"]
