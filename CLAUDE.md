# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é este projeto

IAPredict é um pipeline de Machine Learning que estima, de forma **probabilística**, o desempenho
de cada seleção na Copa do Mundo 2026. Construído **ao vivo** com Spec-Driven Development: cada uma
das 9 features (`.llm/feature_01..09.md`, guiadas pelo `.llm/prd.md`) gera código Python e produz
um resultado verificável no banco. O pipeline aprende com o histórico de jogos, estima gols
esperados (λ) por partida, converte em probabilidades V/E/D e **simula o torneio 10.000 vezes**
(Monte Carlo) para responder "quem leva a taça?".

O pipeline está **completo e materializado** em `src/` + `app.py` (dashboard Streamlit) + modelos
treinados em `models/`.

## Comandos

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # preencher DATABASE_URL (pooler Supabase porta 6543) e CSV_PATH

# Pipeline, na ordem (cada script é idempotente: faz DROP TABLE IF EXISTS + recria)
python src/bronze.py        # 01 — ingestão do CSV cru
python src/silver.py        # 02 — limpeza + split anti-leakage
python src/pesos.py         # 03 — pesos (torneio + recência)
python src/elo.py           # 04 — ELO sequencial
python src/gold.py          # 05 — tabela de treino gold_atributos
python src/treino.py        # 06 — treina 3 modelos, elege o campeão (ver abaixo)
python src/previsao.py      # 07 — previsões dos 72 jogos + experimentos de recência
python src/monte_carlo.py   # 08 — simulação de Monte Carlo (N=10.000)

streamlit run app.py        # 09 — dashboard (3 páginas)
```

Não há suíte de testes nem linter configurado. A validação de cada etapa é a **Verificação (SQL)**
no `.llm/feature_NN.md` correspondente, mais o relatório que cada script imprime.

**Convenção de imports:** os módulos de `src/` usam imports "flat" (`from db import ...`,
`import poisson`) e são executados a partir da raiz (`python src/bronze.py`). O `app.py` replica
isso com `sys.path.insert(0, "src")`.

## Arquitetura: pipeline medallion

`bronze → silver → gold`, materializado como tabelas no Supabase/Postgres. As etapas são
sequenciais; cada uma depende da anterior:

| # | Etapa | Módulo | Saída principal |
|---|------|--------|-----------------|
| 01 | Bronze (ingestão) | `src/bronze.py` | `bronze_jogos` |
| 02 | Silver (limpeza + anti-leakage) | `src/silver.py` | `silver_jogos`, `silver_copa2026` |
| 03 | Pesos (torneio + recência) | `src/pesos.py` | `silver_ponderado` |
| 04 | ELO | `src/elo.py` | `silver_elo_pre_jogo`, `silver_elo_atual` |
| 05 | Atributos Gold | `src/gold.py` | `gold_atributos` |
| 06 | Competição de modelos + validação | `src/treino.py` | `models/*` + `comparacao_modelos`, `metricas_validacao` |
| 07 | Previsão + experimentos | `src/previsao.py` | `previsoes`, `experimentos_mae` |
| 08 | Monte Carlo | `src/monte_carlo.py` | `gold_probabilidades_copa` |
| 09 | Dashboard | `app.py` | aplicação Streamlit |

Tabelas seguem a nomenclatura medallion (`bronze_/silver_/gold_`). Saídas de modelo
(`metricas_validacao`, `comparacao_modelos`, `previsoes`, `experimentos_mae`) não levam prefixo.

Ao mexer nas etapas mais delicadas (**04 ELO** e **08 Monte Carlo**), peça o plano antes de gerar
código.

## Competição de modelos (feature 06)

A feature_06 **não** treina um único modelo: treina três candidatos no mesmo holdout temporal,
compara por métricas e elege o campeão, que passa a alimentar previsão (07) e Monte Carlo (08).

- **`poisson`** — dois GLM Poisson lineares (`statsmodels`), baseline.
- **`dixon_coles`** — os mesmos λ do GLM + correção `rho` de empates/placares baixos.
- **`lgbm`** — dois LightGBM com `objective="poisson"` (não-linear sobre os 6 atributos).

**Métricas** (holdout `>= 2024-01-01`): **log-loss** (primária — qualidade da probabilidade,
que o Monte Carlo consome), **Brier** (apoio), **MAE** de gols e acurácia V/E/D (intuição).
Campeão = menor log-loss (desempate por Brier). Resultados em `comparacao_modelos`; o campeão e
suas métricas em `metricas_validacao`; explicação técnica em `docs/modelo_campeao.md`.

### Interface `Preditor` (como trocar de modelo)

`src/preditor.py` define a interface única e `src/modelos.py` as 3 famílias + persistência. Um
`Preditor` expõe: `lambdas(feats)`, `lambdas_lote(df)`, `probs_de_lambdas(lc, lv)`,
`probs_ved(feats)` e `amostrador(feats)` (sorteador de placar para o Monte Carlo — Poisson
independente por padrão; Dixon-Coles sorteia da matriz corrigida). `previsao.py`/`monte_carlo.py`/
`app.py` consomem **só** essa interface, então o motor é intercambiável. O campeão é persistido em
`models/` + `models/campeao.json` (família, `rho`, métricas); `modelos.carregar_campeao()` o
reconstrói.

## Convenções inegociáveis (toda a modelagem)

- **Anti-leakage (crítico):** os 72 jogos da Copa 2026 ficam isolados em `silver_copa2026` e
  **nunca** entram no treino. Ranking/head-to-head só como contexto, jamais como feature derivada
  do resultado.
- **Janela temporal:** apenas jogos de **2006 em diante** (`DATA_CORTE` em `silver.py`).
- **`peso_torneio`** (ordinal 1/2/3): amistoso=1 / eliminatória e continental=2 / Copa e finais=3.
- **`peso_recencia`** (decaimento exponencial, meia-vida 5 anos): `0.5 ** (idade_anos / 5)`; âncora
  de recência = `2026-06-11` (início da Copa).
- **ELO:** todos começam em 1500; cálculo **sequencial por data**; grava-se sempre o ELO
  **pré-jogo** (anti-leakage). Mando de campo `HFA=100`; fator `K` por peso `{1:20, 2:40, 3:60}`.
- **Holdout temporal:** treino `< 2024-01-01`, teste `>=` (`CORTE` em `treino.py`, reusado em
  07/08).
- **Poisson:** gol é contagem (≥0) — por isso Poisson, não regressão linear.
- **`gold_atributos`:** exclui amistosos (só jogos competitivos); features do modelo são os 6 de
  `ATRIBUTOS` (`elo_casa, elo_visitante, dif_elo, neutro, peso_torneio, peso_recencia`) —
  identificadores nunca entram como feature.
- **Monte Carlo:** `N_SIMULACOES=10000`, `SEED=42`.

## Módulos compartilhados

- `src/db.py` — conexão. `get_engine()` (SQLAlchemy, p/ pandas) e `get_raw_connection()` (psycopg2
  cru, necessário para carga em massa via `COPY`).
- `src/poisson.py` — matriz de placares, correção Dixon-Coles, probabilidades V/E/D e métricas
  (`log_loss_ved`, `brier_ved`). Reusado por 06/07/08.
- `src/preditor.py` / `src/modelos.py` — interface de modelo e as 3 famílias (ver acima).
- `src/bandeiras.py` — emoji por seleção (só no dashboard).

## Dados

- `data/results.csv` — ~49.450 jogos internacionais (1872 → jogos agendados da Copa 2026).
  Colunas: `date, home_team, away_team, home_score, away_score, tournament, city, country, neutral`.
  O campo `tournament` tem alta cardinalidade e valores com vírgulas/aspas — tratar o parsing.
- `data/grupos_copa2026.csv` — grupos A–L (seed). `data/calendario_copa2026.csv` — chaveamento do
  mata-mata (slots). Usados nas features 07–08 e no dashboard.

## Banco de dados (Supabase / Postgres)

A persistência do pipeline é via **`DATABASE_URL`** (definida no `.env`): consultas/pandas pelo
engine SQLAlchemy e **carga em massa via `COPY`** (psycopg2). **Não há `.mcp.json`** neste repo —
se o MCP do Supabase estiver disponível na sessão, use-o apenas para inspeção/validação interativa
(`list_tables`, `get_logs`, `get_advisors`), **não** como caminho de carga.

Cada feature termina com uma **Verificação (SQL)** em `.llm/feature_NN.md` que deve ser executada
para validar a etapa.

## Skills disponíveis

Skills do Supabase em `.agents/skills/` (ver `skills-lock.json`):
- `supabase` — orientação geral de desenvolvimento e segurança.
- `supabase-postgres-best-practices` — práticas de schema, índices, RLS e performance
  (consultar `references/` ao modelar tabelas e queries).
