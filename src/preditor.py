"""Interface única de previsão de placar — o "motor" do pipeline.

Permite que ``previsao.py``, ``monte_carlo.py`` e ``app.py`` troquem de modelo (Poisson,
Dixon-Coles, LightGBM) sem reescrita: todo modelo é embrulhado num ``Preditor`` que expõe
sempre os mesmos métodos. Um *jogo* é descrito por um dicionário de atributos (``montar_features``)
e o ``Preditor`` devolve gols esperados (λ), probabilidades V/E/D e um sorteador de placar para
o Monte Carlo.

As fábricas que treinam cada modelo e devolvem um ``Preditor`` ficam em ``modelos.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import poisson

# Os 6 atributos do modelo (identificadores NÃO entram como feature).
ATRIBUTOS = ["elo_casa", "elo_visitante", "dif_elo", "neutro", "peso_torneio", "peso_recencia"]


def montar_X(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de features do GLM: atributos na ordem canônica, ``neutro`` inteiro, + constante."""
    X = df[ATRIBUTOS].copy()
    X["neutro"] = X["neutro"].astype(int)
    return sm.add_constant(X, has_constant="add")


def montar_features(
    elo_casa: float, elo_visit: float, neutro: bool, peso_torneio: float, peso_recencia: float = 1.0
) -> dict:
    """Dicionário de atributos de um confronto (``peso_recencia=1.0`` = jogo no presente)."""
    return {
        "elo_casa": elo_casa,
        "elo_visitante": elo_visit,
        "dif_elo": elo_casa - elo_visit,
        "neutro": bool(neutro),
        "peso_torneio": peso_torneio,
        "peso_recencia": peso_recencia,
    }


class Preditor:
    """Contrato comum a todos os modelos. Subclasses implementam ``lambdas``/``lambdas_lote``.

    O default trata o placar como duas Poisson independentes; modelos com estrutura própria
    (Dixon-Coles) sobrescrevem ``probs_de_lambdas`` e ``amostrador``.
    """

    nome = "base"

    # --- a implementar nas subclasses ---
    def lambdas(self, feats: dict) -> tuple[float, float]:
        """(λ_casa, λ_visitante) para um confronto."""
        raise NotImplementedError

    def lambdas_lote(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Versão vetorizada para um DataFrame de confrontos (avaliação)."""
        raise NotImplementedError

    def salvar(self, models_dir: str) -> dict:
        """Persiste os artefatos e devolve os metadados (família, rho, …) para o campeao.json."""
        raise NotImplementedError

    # --- comportamento comum (Poisson independente por padrão) ---
    def probs_de_lambdas(self, lam_casa: float, lam_visit: float) -> tuple[float, float, float]:
        return poisson.probabilidades_resultado(lam_casa, lam_visit)

    def probs_ved(self, feats: dict) -> tuple[float, float, float]:
        return self.probs_de_lambdas(*self.lambdas(feats))

    def amostrador(self, feats: dict):
        """Devolve uma função sem argumentos que sorteia um placar (gc, gv) via ``np.random``.

        O Monte Carlo cacheia o sorteador por confronto, então o custo de montar features /
        prever λ é pago uma vez por par (casa, visitante, neutro).
        """
        lam_casa, lam_visit = self.lambdas(feats)
        return lambda: (int(np.random.poisson(lam_casa)), int(np.random.poisson(lam_visit)))
