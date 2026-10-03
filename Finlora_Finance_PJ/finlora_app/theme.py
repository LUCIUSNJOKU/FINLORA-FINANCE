"""Visual identity + reusable UI components for the Finlora app."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Semantic palette (matches the colours used in the EDA notebook)
LEGIT, FRAUD = "#2A6FDB", "#E63946"
TEAL, AMBER = "#00A896", "#F4A300"
INK, MUTED, PAPER, RULE = "#0B1B2B", "#5C6B7A", "#F6F4EF", "#E2DDD2"
FONT = "IBM Plex Sans, Segoe UI, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
.stApp { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
h1, h2, h3, h4 { font-family: 'Fraunces', Georgia, serif !important; color: #0B1B2B; letter-spacing: -0.012em; font-weight: 600 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #F6F4EF; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }

.fl-hero { background: #0B1B2B; color: #F6F4EF; border-radius: 18px; padding: 2.1rem 2.4rem 2rem 2.4rem; margin-bottom: 1.4rem; position: relative; overflow: hidden; }
.fl-hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%; border: 36px solid rgba(230,57,70,.16); }
.fl-hero .fl-kicker { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #F4A300; margin-bottom: .6rem; }
.fl-hero h1 { color: #F6F4EF; font-size: 2.35rem; line-height: 1.12; margin: 0 0 .7rem 0; padding: 0; max-width: 780px; }
.fl-hero p { color: #C5CFDA; font-size: 1.04rem; line-height: 1.55; max-width: 760px; margin: 0; }

.fl-head { margin: .2rem 0 1.4rem 0; padding-bottom: 1rem; border-bottom: 1px solid #E2DDD2; }
.fl-head .fl-kicker { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #E63946; margin-bottom: .35rem; }
.fl-head h1 { font-size: 2.05rem; margin: 0 0 .45rem 0; padding: 0; }
.fl-head p { color: #5C6B7A; font-size: 1.02rem; line-height: 1.55; max-width: 820px; margin: 0; }

.fl-kpi { background: #FFFFFF; border: 1px solid #E2DDD2; border-left: 4px solid #0B1B2B; border-radius: 10px; padding: .85rem 1rem .8rem 1rem; height: 100%; }
.fl-kpi .l { font-size: .8rem; color: #5C6B7A; margin-bottom: .15rem; }
.fl-kpi .v { font-family: 'IBM Plex Mono', monospace; font-size: 1.62rem; font-weight: 600; color: #0B1B2B; line-height: 1.25; }
.fl-kpi .s { font-size: .78rem; color: #5C6B7A; margin-top: .15rem; }
.fl-kpi.fraud { border-left-color: #E63946; } .fl-kpi.legit { border-left-color: #2A6FDB; }
.fl-kpi.teal { border-left-color: #00A896; } .fl-kpi.amber { border-left-color: #F4A300; }

.fl-call { background: #FFFFFF; border: 1px solid #E2DDD2; border-radius: 10px; padding: .9rem 1.1rem; margin: .4rem 0 1rem 0; font-size: .96rem; line-height: 1.55; }
.fl-call b.t { display: block; margin-bottom: .15rem; font-family: 'Fraunces', serif; font-size: 1.02rem; }
.fl-call.info { border-left: 4px solid #2A6FDB; } .fl-call.risk { border-left: 4px solid #E63946; }
.fl-call.good { border-left: 4px solid #00A896; } .fl-call.warn { border-left: 4px solid #F4A300; background: #FFF9EC; }

.fl-step { min-height: 210px; background: #FFFFFF; border: 1px solid #E2DDD2; border-radius: 12px; padding: 1rem 1.1rem; height: 100%; }
.fl-step .n { font-family: 'IBM Plex Mono', monospace; font-size: .75rem; color: #E63946; }
.fl-step .h { font-family: 'Fraunces', serif; font-size: 1.12rem; font-weight: 600; margin: .1rem 0 .3rem 0; }
.fl-step .d { font-size: .88rem; color: #5C6B7A; line-height: 1.45; }
.fl-step .o { font-family: 'IBM Plex Mono', monospace; font-size: .82rem; margin-top: .55rem; color: #0B1B2B; }

.fl-pill { display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .8rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.fl-pill.fraud { background: #FDE6E8; color: #B3202D; } .fl-pill.ok { background: #DDF3F0; color: #00766A; }
.fl-pill.warn { background: #FFF0CC; color: #8A5B00; }

.fl-brand { font-family: 'Fraunces', serif; font-size: 1.35rem; font-weight: 600; color: #F6F4EF; margin: .2rem 0 0 0; }
.fl-brand-sub { font-size: .78rem; color: #8FA0B3; margin-bottom: .6rem; }
[data-testid="stAppDeployButton"], .stAppDeployButton { display: none; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #8FA0B3; margin-top: .7rem; }
.fl-table { width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #E2DDD2; border-radius: 10px; overflow: hidden; background: #fff; font-size: .9rem; margin: .3rem 0 1rem 0; }
.fl-table th { background: #ECE8DF; color: #0B1B2B; font-weight: 600; text-align: left; padding: .6rem .8rem; font-size: .82rem; }
.fl-table td { padding: .6rem .8rem; border-top: 1px solid #EFEBE2; vertical-align: top; line-height: 1.45; color: #1d2b3a; }
.fl-table td.num { text-align: right; font-family: 'IBM Plex Mono', monospace; white-space: nowrap; }
.fl-table td.key { font-weight: 600; white-space: nowrap; }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 500; }
div[data-testid="stDataFrame"] { border: 1px solid #E2DDD2; border-radius: 8px; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "_plotly_ready" not in st.session_state:
        st.session_state["_plotly_ready"] = True
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=[LEGIT, FRAUD, TEAL, AMBER, "#7A5CFA", "#8A99A8"],
        xaxis=dict(gridcolor="#E7E2D8", zerolinecolor="#D5CFC2", linecolor="#CFC8B9"),
        yaxis=dict(gridcolor="#E7E2D8", zerolinecolor="#D5CFC2", linecolor="#CFC8B9"),
        legend=dict(orientation="h", y=1.08, x=0, title_text=""),
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(font=dict(family="Fraunces, Georgia, serif", size=17), x=0),
        hoverlabel=dict(font_family=FONT),
    )
    pio.templates["finlora"] = tpl
    pio.templates.default = "finlora"


def style(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, template="finlora")
    if title:
        fig.update_layout(title_text=title)
    return fig


def show(fig: go.Figure, height: int = 380, title: str | None = None, key: str | None = None) -> None:
    st.plotly_chart(style(fig, height, title), width="stretch",
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, key=key)


def header(kicker: str, title: str, lede: str) -> None:
    st.markdown(f'<div class="fl-head"><div class="fl-kicker">{kicker}</div><h1>{title}</h1><p>{lede}</p></div>',
                unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "", tone: str = "") -> None:
    small = ' style="font-size:1.12rem;padding:.35rem 0"' if len(str(value)) > 11 else ""
    col.markdown(f'<div class="fl-kpi {tone}"><div class="l">{label}</div><div class="v"{small}>{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def kpis(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        kpi(c, *it)


def callout(text: str, kind: str = "info", title: str | None = None) -> None:
    t = f'<b class="t">{title}</b>' if title else ""
    st.markdown(f'<div class="fl-call {kind}">{t}{text}</div>', unsafe_allow_html=True)


def section(title: str, sub: str | None = None) -> None:
    st.markdown(f"### {title}")
    if sub:
        st.caption(sub)


def pct(x: float, d: int = 2) -> str:
    return f"{x * 100:.{d}f}%"


def num(x) -> str:
    return f"{int(x):,}"


def html_table(df, key_col: str | None = None, num_cols: tuple = ()) -> None:
    """Wrapped, presentation-grade table (st.dataframe cannot wrap long text)."""
    import html as _h
    head = "".join(f"<th>{_h.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            cls = "num" if c in num_cols else ("key" if c == key_col else "")
            txt = f"{v:,}" if c in num_cols and isinstance(v, (int, float)) else _h.escape(str(v))
            tds.append(f'<td class="{cls}">{txt}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    st.markdown(f'<table class="fl-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)
