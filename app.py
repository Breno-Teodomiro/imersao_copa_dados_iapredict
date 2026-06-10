"""Spec 09 — Dashboard Streamlit do IAPredict (Copa 2026).

Quatro páginas: probabilidades pré-computadas, simulação ao vivo, comparação de modelos e
explorador de partidas. Lê o banco via DATABASE_URL e reaproveita os módulos de src/.
Visual premium (tema em .streamlit/config.toml) com bandeiras reais (flagcdn).

Rodar local:  streamlit run app.py
Deploy:       Streamlit Cloud (definir DATABASE_URL em Secrets).
"""

from __future__ import annotations

import os
import sys

import altair as alt
import pandas as pd
import streamlit as st

# Os módulos de src/ usam imports "flat" (from db import ...); replicamos o padrão.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Ponte de segredo: no Streamlit Cloud a connection string vem de st.secrets; localmente do .env.
try:
    if "DATABASE_URL" in st.secrets and "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except Exception:
    pass

from db import get_engine  # noqa: E402
from previsao import PESO_TORNEIO_COPA, carregar_modelos, prever_jogo  # noqa: E402
from monte_carlo import NOMES_RODADA, preparar, simular_torneio_detalhado, slots_terceiros  # noqa: E402
from bandeiras import bandeira_url, com_bandeira  # noqa: E402

TOP_N = 12  # padrão de quantas seleções mostrar na página de probabilidades

st.set_page_config(page_title="IAPredict — Copa 2026", page_icon="🏆", layout="wide")

FASES_PT = {
    "prob_grupo": "Passa do grupo", "prob_oitavas": "Oitavas", "prob_quartas": "Quartas",
    "prob_semi": "Semi", "prob_final": "Final", "prob_campea": "Campeã",
}

# Paleta (espelha o config.toml)
OURO, ESMERALDA, CINZA, VERMELHO = "#F2C14E", "#22C55E", "#94A3B8", "#EF4444"


# --------------------------------------------------------------------------- #
# CSS premium (só o que widget nativo não cobre: hero, pódio, confrontos, "VS")
# --------------------------------------------------------------------------- #
def _injeta_css() -> None:
    st.markdown(
        """
        <style>
        .iap-hero {
            display:flex; align-items:center; gap:20px;
            background:
                radial-gradient(1200px 200px at 0% 0%, rgba(242,193,78,.18), transparent 60%),
                linear-gradient(135deg, #0E1A33 0%, #11271F 100%);
            border:1px solid #2A3550; border-left:5px solid #F2C14E;
            border-radius:18px; padding:22px 26px; margin:4px 0 18px 0;
            box-shadow:0 18px 40px -20px rgba(0,0,0,.7);
        }
        .iap-hero-badge {
            font-size:44px; line-height:1;
            filter:drop-shadow(0 4px 10px rgba(242,193,78,.45));
        }
        .iap-hero-title { font-family:Sora,Inter,sans-serif; font-weight:800;
            font-size:30px; letter-spacing:-.5px; line-height:1.1; color:#FFFFFF; }
        .iap-hero-sub { color:#A7B2CC; font-size:14.5px; margin-top:5px; }

        .iap-podium {
            position:relative; text-align:center; border-radius:18px; padding:20px 14px 16px;
            background:linear-gradient(180deg, #15203A 0%, #0E1729 100%);
            border:1px solid #28324E; overflow:hidden;
        }
        .iap-podium::before { content:""; position:absolute; inset:0 0 auto 0; height:4px; }
        .iap-podium.r1 { border-color:#6E5A1E; box-shadow:0 0 0 1px rgba(242,193,78,.35), 0 16px 36px -22px rgba(242,193,78,.7); }
        .iap-podium.r1::before { background:linear-gradient(90deg,#F2C14E,#FFE9A8); }
        .iap-podium.r2::before { background:linear-gradient(90deg,#C7CEDC,#8C94A6); }
        .iap-podium.r3::before { background:linear-gradient(90deg,#D08B5B,#9A6336); }
        .iap-podium .medal { font-size:26px; }
        .iap-podium img { width:96px; height:64px; object-fit:cover; border-radius:8px;
            margin:8px auto 10px; box-shadow:0 8px 18px -8px rgba(0,0,0,.8); border:1px solid #2A3550; }
        .iap-podium .team { font-family:Sora,sans-serif; font-weight:700; font-size:17px; color:#fff; }
        .iap-podium .val { color:#F2C14E; font-weight:700; font-size:22px; margin-top:2px; }
        .iap-podium .lbl { color:#8A94AC; font-size:11.5px; text-transform:uppercase; letter-spacing:.6px; }

        .iap-vs {
            display:flex; align-items:center; justify-content:center; gap:26px;
            background:linear-gradient(135deg,#0E1A33,#11271F);
            border:1px solid #2A3550; border-radius:18px; padding:22px; margin:6px 0 14px;
        }
        .iap-vs .side { display:flex; flex-direction:column; align-items:center; gap:10px; min-width:150px; }
        .iap-vs img { width:120px; height:80px; object-fit:cover; border-radius:10px;
            border:1px solid #2A3550; box-shadow:0 10px 22px -10px rgba(0,0,0,.8); }
        .iap-vs .team { font-family:Sora,sans-serif; font-weight:700; font-size:18px; color:#fff; text-align:center; }
        .iap-vs .vs { font-family:Sora,sans-serif; font-weight:800; font-size:30px; color:#F2C14E; opacity:.9; }

        .iap-bracket { display:flex; flex-direction:column; gap:8px; }
        .iap-row {
            display:grid; grid-template-columns:1fr auto 1fr; align-items:center; gap:14px;
            background:#101A2E; border:1px solid #232E48; border-radius:12px; padding:9px 16px;
        }
        .iap-row .home { justify-self:end; }
        .iap-row .away { justify-self:start; }
        .iap-row .side { display:flex; align-items:center; gap:9px; color:#C9D2E6; font-size:15px; }
        .iap-row img { width:30px; height:20px; object-fit:cover; border-radius:4px; border:1px solid #2A3550; }
        .iap-row .score { font-family:Sora,sans-serif; font-weight:700; color:#fff; font-size:16px;
            background:#0A1120; border:1px solid #28324E; border-radius:8px; padding:3px 12px; white-space:nowrap; }
        .iap-row .win { color:#FFE9A8; font-weight:700; }
        .iap-row .pen { color:#8A94AC; font-size:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Recursos cacheados
# --------------------------------------------------------------------------- #
@st.cache_resource
def _engine():
    return get_engine()


@st.cache_resource
def _preditor_e_elos():
    preditor = carregar_modelos()
    elos = dict(pd.read_sql("SELECT selecao, elo FROM silver_elo_atual", _engine()).itertuples(index=False, name=None))
    return preditor, elos


@st.cache_resource
def _preparacao_torneio():
    return preparar()


@st.cache_data
def _probabilidades() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM gold_probabilidades_copa ORDER BY prob_campea DESC", _engine())


NOME_MODELO = {"poisson": "Poisson GLM", "dixon_coles": "Poisson + Dixon-Coles", "lgbm": "LightGBM (Poisson)"}


@st.cache_data
def _modelo_campeao() -> str:
    try:
        nome = pd.read_sql("SELECT modelo FROM metricas_validacao ORDER BY id DESC LIMIT 1", _engine()).iloc[0]["modelo"]
        return NOME_MODELO.get(nome, nome)
    except Exception:
        return "—"


@st.cache_data
def _comparacao() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM comparacao_modelos ORDER BY log_loss", _engine())


# --------------------------------------------------------------------------- #
# Componentes visuais
# --------------------------------------------------------------------------- #
def hero(titulo: str, subtitulo: str, icone: str = "🏆") -> None:
    st.markdown(
        f'<div class="iap-hero"><div class="iap-hero-badge">{icone}</div>'
        f'<div><div class="iap-hero-title">{titulo}</div>'
        f'<div class="iap-hero-sub">{subtitulo}</div></div></div>',
        unsafe_allow_html=True,
    )


def _podium_card(rank: int, team: str, valor: str, label: str) -> str:
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}[rank]
    url = bandeira_url(team, 160)
    img = f'<img src="{url}" alt="">' if url else ""
    return (
        f'<div class="iap-podium r{rank}"><div class="medal">{medal}</div>{img}'
        f'<div class="team">{team}</div><div class="val">{valor}</div>'
        f'<div class="lbl">{label}</div></div>'
    )


def podio(itens: list[tuple[int, str, str, str]]) -> None:
    """itens: lista de (rank, time, valor, label). Renderiza 3 cartões lado a lado."""
    cols = st.columns(len(itens), gap="medium")
    for col, (rank, team, valor, label) in zip(cols, itens):
        col.markdown(_podium_card(rank, team, valor, label), unsafe_allow_html=True)


def _linha_confronto(casa, gc, gv, visit, vencedor, penaltis) -> str:
    uc, uv = bandeira_url(casa, 40), bandeira_url(visit, 40)
    cls_c = "side win" if vencedor == casa else "side"
    cls_v = "side win" if vencedor == visit else "side"
    pen = ' <span class="pen">(pên.)</span>' if penaltis else ""
    return (
        f'<div class="iap-row">'
        f'<div class="home {cls_c}"><span>{casa}</span><img src="{uc}"></div>'
        f'<div class="score">{gc} - {gv}{pen}</div>'
        f'<div class="away {cls_v}"><img src="{uv}"><span>{visit}</span></div>'
        f'</div>'
    )


def confrontos(jogos: list[tuple]) -> None:
    linhas = "".join(_linha_confronto(*j) for j in jogos)
    st.markdown(f'<div class="iap-bracket">{linhas}</div>', unsafe_allow_html=True)


def _flag_col(df: pd.DataFrame, w: int = 80) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "bandeira", df["selecao"].map(lambda n: bandeira_url(n, w)))
    return df


# --------------------------------------------------------------------------- #
# Páginas
# --------------------------------------------------------------------------- #
def pagina_probabilidades():
    hero(
        "Quem leva a taça?",
        f"10.000 simulações de Monte Carlo · modelo campeão "
        f"<b style='color:#F2C14E'>{_modelo_campeao()}</b> · Copa do Mundo 2026",
    )

    base = _probabilidades()
    opcoes = [8, 12, 16, 24]
    n = st.segmented_control("Quantas seleções", opcoes, default=TOP_N, key="topn") or TOP_N
    df = base.head(n).copy()

    # Pódio das 3 favoritas
    top3 = df.head(3)
    podio([
        (1, top3.iloc[0]["selecao"], f"{top3.iloc[0]['prob_campea']*100:.1f}%", "Campeã"),
        (2, top3.iloc[1]["selecao"], f"{top3.iloc[1]['prob_campea']*100:.1f}%", "Vice favorita"),
        (3, top3.iloc[2]["selecao"], f"{top3.iloc[2]['prob_campea']*100:.1f}%", "Terceira"),
    ])
    st.space("medium")

    # Gráfico de barras com bandeira real ao fim de cada barra
    st.subheader("Favoritas ao título")
    fav = df[["selecao", "prob_campea"]].copy()
    fav["pct"] = (fav["prob_campea"] * 100).round(1)
    fav["url"] = fav["selecao"].map(lambda s: bandeira_url(s, 40))
    eixo_y = alt.Y("selecao:N", sort="-x", title=None,
                   axis=alt.Axis(labelFontSize=13, labelColor="#E7ECF6"))
    base_c = alt.Chart(fav).encode(y=eixo_y)
    barras = base_c.mark_bar(color=OURO, cornerRadiusEnd=6, height=20).encode(
        x=alt.X("pct:Q", title="% de título", axis=alt.Axis(grid=False, labelColor="#8A94AC")),
        tooltip=[alt.Tooltip("selecao:N", title="Seleção"), alt.Tooltip("pct:Q", title="% campeã")],
    )
    flags = base_c.mark_image(width=26, height=17).encode(x="pct:Q", url="url:N")
    rotulos = base_c.mark_text(align="left", dx=20, color="#E7ECF6", fontWeight="bold", fontSize=12).encode(
        x="pct:Q", text=alt.Text("pct:Q", format=".1f"),
    )
    chart = (barras + flags + rotulos).properties(height=max(260, 30 * len(fav))).configure_view(strokeWidth=0)
    st.altair_chart(chart, theme="streamlit")

    # Tabela por fase: bandeira real + barra de progresso na coluna "Campeã"
    st.subheader("Probabilidade por fase")
    tab = _flag_col(df, 80)
    colcfg = {
        "bandeira": st.column_config.ImageColumn(" ", width="small"),
        "selecao": st.column_config.TextColumn("Seleção"),
    }
    for chave, nome in FASES_PT.items():
        tab[chave] = (tab[chave] * 100).round(1)
        if chave == "prob_campea":
            colcfg[chave] = st.column_config.ProgressColumn(
                nome, min_value=0.0, max_value=float(tab[chave].max()), format="%.1f%%")
        else:
            colcfg[chave] = st.column_config.NumberColumn(nome, format="%.1f%%")
    st.dataframe(
        tab[["bandeira", "selecao", *FASES_PT.keys()]],
        column_config=colcfg, hide_index=True,
    )


def pagina_simulacao():
    hero("Simulação ao vivo",
         "Uma Copa inteira sorteada do zero — grupos, mata-mata e campeã. "
         "Muda a cada clique; é aleatória, não é o palpite final.", icone="🎲")

    if st.button("Simular novamente", type="primary", icon=":material/casino:"):
        st.session_state["_sim_n"] = st.session_state.get("_sim_n", 0) + 1

    grupo_de, times_do_grupo, jogos_grupo, calendario, lambdas = _preparacao_torneio()
    slots_3 = slots_terceiros(calendario)
    r = simular_torneio_detalhado(grupo_de, times_do_grupo, jogos_grupo, calendario, lambdas, slots_3)
    st.balloons()

    podio([
        (1, r["campeao"], "Campeã", "1º lugar"),
        (2, r["vice"], "Vice", "2º lugar"),
        (3, r["terceiro"], "Terceiro", "3º lugar"),
    ])
    st.space("medium")

    st.subheader("Mata-mata")
    for rodada, jogos in r["mata_mata"].items():
        st.markdown(f"**{NOMES_RODADA.get(rodada, rodada)}**")
        confrontos(jogos)
        st.space("small")

    st.subheader("Fase de grupos")
    grupos = sorted(r["grupos"])
    for faixa in range(0, len(grupos), 2):
        cols = st.columns(2, gap="medium")
        for col, grupo in zip(cols, grupos[faixa:faixa + 2]):
            with col.expander(f"Grupo {grupo}", icon=":material/groups:"):
                for casa, gc, gv, visit in r["grupos"][grupo]["jogos"]:
                    st.write(f"{com_bandeira(casa)} **{gc} - {gv}** {com_bandeira(visit)}")
                tabela = _flag_col(r["grupos"][grupo]["classificacao"])
                st.dataframe(
                    tabela, hide_index=True,
                    column_config={"bandeira": st.column_config.ImageColumn(" ", width="small"),
                                   "selecao": st.column_config.TextColumn("Seleção")},
                )


def pagina_explorador():
    hero("Explorador de partidas",
         "Escolha duas seleções e veja, ao vivo, os gols esperados (xG) e as probabilidades "
         "de vitória, empate e derrota.", icone="🔍")
    preditor, elos = _preditor_e_elos()
    times = sorted(elos)

    c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
    casa = c1.selectbox("Mandante", times, format_func=com_bandeira,
                        index=times.index("Brazil") if "Brazil" in times else 0)
    fora = c2.selectbox("Visitante", times, format_func=com_bandeira,
                        index=times.index("Spain") if "Spain" in times else 1)
    neutro = c3.toggle("Campo neutro", value=True)

    if casa == fora:
        st.warning("Escolha duas seleções diferentes.", icon=":material/sports_soccer:")
        return

    # "VS" com as bandeiras reais
    uc, uv = bandeira_url(casa, 320), bandeira_url(fora, 320)
    st.markdown(
        f'<div class="iap-vs"><div class="side"><img src="{uc}"><div class="team">{casa}</div></div>'
        f'<div class="vs">VS</div>'
        f'<div class="side"><img src="{uv}"><div class="team">{fora}</div></div></div>',
        unsafe_allow_html=True,
    )

    p = prever_jogo(casa, fora, neutro, PESO_TORNEIO_COPA, elos=elos, preditor=preditor)

    m1, m2 = st.columns(2)
    m1.metric(f"xG — {casa}", round(p["gols_esperados_casa"], 2))
    m2.metric(f"xG — {fora}", round(p["gols_esperados_visitante"], 2))

    st.subheader("Probabilidades de resultado")
    res = pd.DataFrame({
        "res": [f"Vitória {casa}", "Empate", f"Vitória {fora}"],
        "pct": [p["prob_vitoria"] * 100, p["prob_empate"] * 100, p["prob_derrota"] * 100],
        "cor": [ESMERALDA, CINZA, VERMELHO],
    })
    barra = (
        alt.Chart(res).mark_bar().encode(
            x=alt.X("pct:Q", stack="zero", title=None, axis=alt.Axis(grid=False, labelColor="#8A94AC")),
            color=alt.Color("res:N", scale=alt.Scale(domain=res["res"].tolist(), range=res["cor"].tolist()),
                            legend=alt.Legend(orient="bottom", title=None, labelColor="#E7ECF6")),
            order=alt.Order("pct:Q", sort="descending"),
            tooltip=[alt.Tooltip("res:N", title="Resultado"), alt.Tooltip("pct:Q", title="%", format=".1f")],
        ).properties(height=70).configure_view(strokeWidth=0)
    )
    st.altair_chart(barra)

    p1, p2, p3 = st.columns(3)
    p1.metric(f"Vitória {casa}", f"{p['prob_vitoria']*100:.1f}%")
    p2.metric("Empate", f"{p['prob_empate']*100:.1f}%")
    p3.metric(f"Vitória {fora}", f"{p['prob_derrota']*100:.1f}%")


def pagina_comparacao():
    hero("Comparação de modelos",
         "Competição da feature 06 no holdout temporal (jogos de 2024 em diante). "
         "Campeão = menor <b>log-loss</b> (qualidade da probabilidade); desempate por Brier.",
         icone="⚖️")
    try:
        df = _comparacao()
    except Exception:
        st.info("A tabela `comparacao_modelos` ainda não existe. Rode `python src/treino.py`.",
                icon=":material/info:")
        return
    if df.empty:
        st.info("Sem dados de comparação. Rode `python src/treino.py`.", icon=":material/info:")
        return

    df = df.copy()
    df["Modelo"] = df["modelo"].map(lambda m: NOME_MODELO.get(m, m))
    campeao = df.loc[df["log_loss"].idxmin(), "Modelo"]
    st.success(f"Modelo campeão: **{campeao}**", icon=":material/trophy:")

    st.subheader("log-loss por modelo (menor é melhor)")
    grafico = (
        alt.Chart(df).mark_bar(cornerRadiusEnd=6, height=26).encode(
            x=alt.X("log_loss:Q", title="log-loss", axis=alt.Axis(grid=False, labelColor="#8A94AC")),
            y=alt.Y("Modelo:N", sort="x", title=None, axis=alt.Axis(labelColor="#E7ECF6", labelFontSize=13)),
            color=alt.condition(alt.datum.eh_campeao, alt.value(OURO), alt.value("#3C4B6C")),
            tooltip=[alt.Tooltip("Modelo:N"), alt.Tooltip("log_loss:Q", format=".4f"),
                     alt.Tooltip("brier:Q", format=".4f")],
        ).properties(height=160).configure_view(strokeWidth=0)
    )
    st.altair_chart(grafico)

    st.subheader("Métricas")
    tabela = df.rename(columns={
        "log_loss": "log-loss", "brier": "Brier", "mae_casa": "MAE casa",
        "mae_visitante": "MAE visit.", "acuracia": "Acurácia V/E/D", "eh_campeao": "Campeão",
    }).copy()
    tabela["Acurácia V/E/D"] = (tabela["Acurácia V/E/D"] * 100).round(1)
    tabela["Campeão"] = tabela["Campeão"].map({True: "🏆", False: ""})
    for col in ("log-loss", "Brier", "MAE casa", "MAE visit."):
        tabela[col] = tabela[col].round(4)
    st.dataframe(
        tabela[["Modelo", "log-loss", "Brier", "MAE casa", "MAE visit.", "Acurácia V/E/D", "Campeão"]],
        hide_index=True,
        column_config={"Acurácia V/E/D": st.column_config.NumberColumn("Acurácia V/E/D", format="%.1f%%")},
    )

    relatorio = os.path.join("docs", "modelo_campeao.md")
    if os.path.exists(relatorio):
        with st.expander("Relatório técnico — por que este modelo venceu", icon=":material/description:"):
            with open(relatorio, encoding="utf-8") as f:
                st.markdown(f.read())


# --------------------------------------------------------------------------- #
# Navegação
# --------------------------------------------------------------------------- #
def main() -> None:
    _injeta_css()
    paginas = [
        st.Page(pagina_probabilidades, title="Quem leva a taça", icon=":material/emoji_events:", default=True),
        st.Page(pagina_simulacao, title="Simulação ao vivo", icon=":material/casino:"),
        st.Page(pagina_comparacao, title="Comparação de modelos", icon=":material/balance:"),
        st.Page(pagina_explorador, title="Explorador de partidas", icon=":material/travel_explore:"),
    ]
    nav = st.navigation(paginas, position="sidebar")
    with st.sidebar:
        st.space("small")
        st.caption("⚽ **IAPredict** — pipeline de ML da Copa 2026")
        st.caption(f"Modelo campeão: **{_modelo_campeao()}**")
    nav.run()


if __name__ == "__main__":
    main()
