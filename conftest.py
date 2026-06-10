"""Configuração do pytest: expõe ``src/`` no path (imports flat) e dados sintéticos.

Os testes validam a competição de modelos sem tocar no banco: geram um ``gold_atributos``
sintético a partir de uma Poisson conhecida e exercitam treino/avaliação/persistência.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


@pytest.fixture(scope="session")
def gold_sintetico() -> pd.DataFrame:
    """DataFrame no formato de ``gold_atributos``, gerado de uma Poisson conhecida.

    Datas espalhadas de 2010 a 2026 para permitir o split temporal (treino < 2024 <= teste).
    """
    rng = np.random.default_rng(7)
    n = 2500

    elo_casa = rng.normal(1550, 80, n)
    elo_visit = rng.normal(1500, 80, n)
    dif = elo_casa - elo_visit
    neutro = rng.random(n) < 0.3
    peso_torneio = rng.choice([1, 2, 3], n, p=[0.5, 0.35, 0.15]).astype(float)

    datas = pd.Series(pd.to_datetime("2010-01-01") + pd.to_timedelta(rng.integers(0, 16 * 365, n), unit="D"))
    idade = (pd.Timestamp("2026-06-11") - datas).dt.days / 365.25
    peso_recencia = 0.5 ** (idade / 5)

    lam_c = np.clip(1.3 + 0.0018 * dif + 0.15 * (~neutro), 0.2, 4.0)
    lam_v = np.clip(1.1 - 0.0018 * dif, 0.2, 4.0)

    return pd.DataFrame({
        "data": datas, "elo_casa": elo_casa, "elo_visitante": elo_visit, "dif_elo": dif,
        "neutro": neutro, "peso_torneio": peso_torneio, "peso_recencia": peso_recencia,
        "gols_casa": rng.poisson(lam_c), "gols_visitante": rng.poisson(lam_v),
    })


@pytest.fixture()
def feats():
    """Atributos de um confronto qualquer (mandante mais forte)."""
    return {"elo_casa": 1600.0, "elo_visitante": 1500.0, "dif_elo": 100.0,
            "neutro": True, "peso_torneio": 3, "peso_recencia": 1.0}
