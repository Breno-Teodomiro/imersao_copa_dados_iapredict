"""Utilitários de Poisson compartilhados pelo pipeline (features 06, 07 e 08).

Dado λ_casa e λ_visitante (gols esperados de cada lado), modela o placar como duas Poisson
independentes e deriva probabilidades de resultado (vitória/empate/derrota do mandante).

Inclui também a correção de **Dixon-Coles** (``matriz_dixon_coles``), que ajusta os placares
baixos (0-0, 1-0, 0-1, 1-1) onde a Poisson independente subestima empates, e as métricas de
avaliação probabilística (``log_loss_ved``, ``brier_ved``) usadas na competição de modelos.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOLS = 10

# Índice de cada rótulo V/E/D nas matrizes de probabilidade (ordem do mandante).
_IDX_VED = {"V": 0, "E": 1, "D": 2}


def matriz_placares(lam_casa: float, lam_visit: float, max_gols: int = MAX_GOLS) -> np.ndarray:
    """Matriz (max_gols+1 × max_gols+1) com P(placar = i×j), Poisson independentes."""
    p_casa = poisson.pmf(np.arange(max_gols + 1), lam_casa)
    p_visit = poisson.pmf(np.arange(max_gols + 1), lam_visit)
    return np.outer(p_casa, p_visit)


def matriz_dixon_coles(
    lam_casa: float, lam_visit: float, rho: float, max_gols: int = MAX_GOLS
) -> np.ndarray:
    """Matriz de placares com a correção de Dixon-Coles (parâmetro ``rho``), renormalizada.

    A correção τ multiplica apenas as quatro células de placar baixo; ``rho`` < 0 aumenta a
    massa de empate (0-0, 1-1) e reduz 1-0/0-1, corrigindo a sub-dispersão da Poisson.
    """
    m = matriz_placares(lam_casa, lam_visit, max_gols)
    m[0, 0] *= 1.0 - lam_casa * lam_visit * rho
    m[0, 1] *= 1.0 + lam_casa * rho
    m[1, 0] *= 1.0 + lam_visit * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else m


def probabilidades_da_matriz(m: np.ndarray) -> tuple[float, float, float]:
    """Deriva (P(vitória casa), P(empate), P(vitória visitante)) de uma matriz de placares."""
    p_vit = np.tril(m, -1).sum()   # gols_casa > gols_visit
    p_emp = np.trace(m)            # diagonal: gols_casa == gols_visit
    p_der = np.triu(m, 1).sum()    # gols_casa < gols_visit
    return float(p_vit), float(p_emp), float(p_der)


def probabilidades_resultado(
    lam_casa: float, lam_visit: float, max_gols: int = MAX_GOLS
) -> tuple[float, float, float]:
    """Retorna (P(vitória casa), P(empate), P(vitória visitante)) — Poisson independentes."""
    return probabilidades_da_matriz(matriz_placares(lam_casa, lam_visit, max_gols))


def resultado_previsto(lam_casa: float, lam_visit: float, max_gols: int = MAX_GOLS) -> str:
    """Resultado mais provável: 'V' (vitória casa), 'E' (empate) ou 'D' (derrota casa)."""
    return "VED"[int(np.argmax(probabilidades_resultado(lam_casa, lam_visit, max_gols)))]


def resultados_previstos(
    lam_casa: np.ndarray, lam_visit: np.ndarray, max_gols: int = MAX_GOLS
) -> np.ndarray:
    """Versão vetorizada: array de 'V'/'E'/'D' para arrays de λ."""
    rotulos = np.array(["V", "E", "D"])
    saida = np.empty(len(lam_casa), dtype="<U1")
    for i, (lc, lv) in enumerate(zip(lam_casa, lam_visit)):
        probs = probabilidades_resultado(lc, lv, max_gols)
        saida[i] = rotulos[int(np.argmax(probs))]
    return saida


def resultado_real(gols_casa: np.ndarray, gols_visit: np.ndarray) -> np.ndarray:
    """Rotula o resultado observado como 'V'/'E'/'D' (do ponto de vista do mandante)."""
    return np.where(gols_casa > gols_visit, "V", np.where(gols_casa == gols_visit, "E", "D"))


def _onehot_ved(reais: np.ndarray) -> np.ndarray:
    """Matriz N×3 (one-hot) do resultado real, na ordem V/E/D."""
    y = np.zeros((len(reais), 3))
    for i, r in enumerate(reais):
        y[i, _IDX_VED[r]] = 1.0
    return y


def log_loss_ved(probs: np.ndarray, reais: np.ndarray, eps: float = 1e-15) -> float:
    """Log-loss multiclasse (V/E/D). Penaliza confiança alta em previsões erradas.

    ``probs``: array N×3 com (P(V), P(E), P(D)); ``reais``: array de 'V'/'E'/'D'.
    Métrica **primária** de eleição do modelo campeão — quanto menor, melhor.
    """
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    escolhidas = p[np.arange(len(reais)), [_IDX_VED[r] for r in reais]]
    return float(-np.mean(np.log(escolhidas)))


def brier_ved(probs: np.ndarray, reais: np.ndarray) -> float:
    """Brier score multiclasse (V/E/D): erro quadrático médio das probabilidades. Menor é melhor."""
    y = _onehot_ved(np.asarray(reais))
    return float(np.mean(np.sum((np.asarray(probs, dtype=float) - y) ** 2, axis=1)))
