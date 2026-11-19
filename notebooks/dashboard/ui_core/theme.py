# ============================================================
# HTML RENDERER AND DASHBOARD THEME
# ============================================================
#
# Extracted verbatim from app_lstm.py (sections 2-3).
# The helper function bodies and the stylesheet string are
# unchanged; the stylesheet is now exposed as THEME_CSS so that
# app_lstm.py can still render it in the same position.
# ============================================================

from textwrap import dedent

import streamlit as st


def render_html(html):
    """
    Render an HTML fragment safely in Streamlit.

    The fragment is dedented and stripped before it reaches
    st.markdown(..., unsafe_allow_html=True). Removing the
    leading indentation is what stops Streamlit's Markdown
    parser from treating the HTML as an indented code block
    and printing the tags literally on the page.
    """
    st.markdown(
        dedent(html).strip(),
        unsafe_allow_html=True,
    )


def render_section(title, subtitle=None):
    """
    Render one consistent section heading.

    Using a dedicated helper keeps every section visually
    identical without duplicating HTML blocks.
    """
    subtitle_block = ""

    if subtitle:

        subtitle_block = (
            '<div class="section-subtitle">'
            f"{subtitle}"
            "</div>"
        )

    render_html(
        '<div class="section-title">'
        f"{title}"
        "</div>"
        + subtitle_block
    )


THEME_CSS = """<style>

    /* --------------------------------------------------------
       TYPOGRAPHY + APP SHELL
    -------------------------------------------------------- */

    :root {
        --bg-deep: #050b08;
        --bg-mid: #08130d;
        --bg-soft: #0b1711;
        --surface: rgba(14, 29, 21, 0.92);
        --surface-2: rgba(17, 34, 25, 0.92);
        --border: rgba(255, 255, 255, 0.07);
        --border-strong: rgba(255, 255, 255, 0.12);

        --text: #f5f7f5;
        --text-soft: #b9c9be;
        --text-muted: #8fa197;

        --accent: #35d977;
        --accent-soft: #7ce8a1;
        --accent-dim: rgba(53, 217, 119, 0.14);

        --positive: #62df8d;
        --negative: #ff7e7e;
        --neutral: #d4dcd7;

        --warning: #ffd184;
    }


    html,
    body,
    [class*="css"] {
        font-family:
            "Inter",
            "Segoe UI",
            system-ui,
            -apple-system,
            sans-serif;
    }


    .stApp {
        background:
            radial-gradient(
                circle at 92% 6%,
                rgba(37, 160, 84, 0.16),
                transparent 30%
            ),
            radial-gradient(
                circle at 8% 92%,
                rgba(20, 92, 56, 0.10),
                transparent 26%
            ),
            linear-gradient(
                135deg,
                #050b08 0%,
                #08130d 50%,
                #0b1711 100%
            );

        color: var(--text);
    }


    .block-container {
        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 3.5rem;
    }


    h1,
    h2,
    h3 {
        color: #ffffff;
        letter-spacing: -0.015em;
    }


    a {
        color: var(--accent-soft);
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #050b08 0%,
                #08130d 100%
            );

        border-right:
            1px solid var(--border);
    }


    section[data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
        font-weight: 850;
    }


    section[data-testid="stSidebar"] .stMetric {
        min-height: 74px;
    }


    /* --------------------------------------------------------
       HEADER
    -------------------------------------------------------- */

    .main-title {
        font-size: 2.9rem;
        font-weight: 900;
        line-height: 1.05;

        margin-bottom: 0.35rem;

        background:
            linear-gradient(
                90deg,
                #ffffff,
                var(--accent-soft)
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }


    .main-subtitle {
        color: var(--text-muted);

        font-size: 1.05rem;

        margin-bottom: 0.9rem;
    }


    .research-pill {
        display: inline-block;

        padding: 5px 12px;

        border-radius: 999px;

        background: var(--accent-dim);

        border: 1px solid rgba(53, 217, 119, 0.32);

        color: var(--accent-soft);

        font-size: 0.72rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.14em;
    }


    /* --------------------------------------------------------
       SECTION HEADINGS
    -------------------------------------------------------- */

    .section-title {
        color: #ffffff;

        font-size: 1.4rem;
        font-weight: 820;

        margin-top: 1.7rem;
        margin-bottom: 0.55rem;

        padding-left: 12px;

        border-left: 3px solid var(--accent);
    }


    .section-subtitle {
        color: var(--text-muted);

        font-size: 0.92rem;

        margin-top: -0.15rem;
        margin-bottom: 1rem;

        padding-left: 12px;
    }


    /* --------------------------------------------------------
       PLAYER HERO
    -------------------------------------------------------- */

    .hero-card {
        background:
            linear-gradient(
                135deg,
                rgba(19, 54, 36, 0.97),
                rgba(8, 25, 17, 0.97)
            );

        border: 1px solid rgba(100, 225, 145, 0.20);

        border-radius: 22px;

        padding: 28px 30px;

        min-height: 208px;

        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
    }


    .player-name {
        color: #ffffff;

        font-size: 2.3rem;
        font-weight: 900;
        line-height: 1.05;

        margin-bottom: 0.55rem;
    }


    .player-club {
        color: var(--text-soft);

        font-size: 1.05rem;
        font-weight: 700;

        margin-bottom: 0.35rem;
    }


    .player-meta {
        color: var(--text-muted);

        font-size: 0.95rem;
        line-height: 1.7;
    }


    /* --------------------------------------------------------
       MODEL BADGES
    -------------------------------------------------------- */

    .badge-lstm,
    .badge-gb {
        display: inline-block;

        margin-top: 13px;

        padding: 7px 13px;

        border-radius: 999px;

        font-size: 0.78rem;
        font-weight: 850;

        letter-spacing: 0.04em;
    }


    .badge-lstm {
        background: rgba(54, 211, 114, 0.12);
        border: 1px solid rgba(54, 211, 114, 0.34);
        color: #72e49a;
    }


    .badge-gb {
        background: rgba(255, 187, 79, 0.11);
        border: 1px solid rgba(255, 187, 79, 0.32);
        color: var(--warning);
    }


    /* --------------------------------------------------------
       STANDARD CARDS
    -------------------------------------------------------- */

    .dashboard-card {
        background: var(--surface);

        border: 1px solid var(--border);

        border-radius: 18px;

        padding: 20px 22px;

        min-height: 138px;

        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
    }


    .card-label {
        color: var(--text-muted);

        font-size: 0.74rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.09em;

        margin-bottom: 8px;
    }


    .card-value {
        color: #ffffff;

        font-size: 1.7rem;
        font-weight: 900;

        line-height: 1.15;
    }


    .card-note {
        color: #819288;

        font-size: 0.84rem;

        margin-top: 8px;
        line-height: 1.5;
    }


    /* --------------------------------------------------------
       CHANGE COLOURS
    -------------------------------------------------------- */

    .positive-value {
        color: var(--positive);
    }


    .negative-value {
        color: var(--negative);
    }


    .neutral-value {
        color: var(--neutral);
    }


    /* --------------------------------------------------------
       LSTM SEQUENCE CARDS
    -------------------------------------------------------- */

    .sequence-card {
        background:
            linear-gradient(
                145deg,
                rgba(16, 33, 24, 0.96),
                rgba(10, 23, 16, 0.96)
            );

        border: 1px solid var(--border);

        border-radius: 17px;

        padding: 18px;

        min-height: 228px;

        text-align: center;

        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.16);
    }


    .sequence-card-active {
        border: 1px solid rgba(73, 224, 133, 0.42);

        box-shadow: 0 12px 30px rgba(38, 150, 86, 0.18);
    }


    .sequence-label {
        color: #68df92;

        font-size: 0.82rem;
        font-weight: 900;

        text-transform: uppercase;
        letter-spacing: 0.08em;
    }


    .sequence-date {
        color: #ffffff;

        font-size: 1rem;
        font-weight: 750;

        margin-top: 6px;
        margin-bottom: 11px;
    }


    .sequence-market-value {
        color: #ffffff;

        font-size: 1.55rem;
        font-weight: 900;

        margin-bottom: 12px;
    }


    .sequence-detail {
        color: #91a298;

        font-size: 0.86rem;
        line-height: 1.75;
    }


    /* --------------------------------------------------------
       ROUTING / INFO PANELS
    -------------------------------------------------------- */

    .routing-success {
        background: rgba(41, 159, 86, 0.10);

        border: 1px solid rgba(71, 213, 122, 0.25);

        border-radius: 15px;

        padding: 15px 17px;

        color: #a6ddb9;

        line-height: 1.6;

        margin-top: 14px;
    }


    .routing-warning {
        background: rgba(210, 151, 49, 0.09);

        border: 1px solid rgba(240, 183, 76, 0.24);

        border-radius: 15px;

        padding: 15px 17px;

        color: #e4c992;

        line-height: 1.6;

        margin-top: 14px;
    }


    .info-panel {
        background: rgba(255, 255, 255, 0.025);

        border: 1px solid rgba(255, 255, 255, 0.065);

        border-radius: 15px;

        padding: 16px 18px;

        color: #a6b4aa;

        font-size: 0.90rem;
        line-height: 1.65;
    }


    /* --------------------------------------------------------
       FINAL FORECAST CARD
    -------------------------------------------------------- */

    .forecast-card {
        background:
            linear-gradient(
                135deg,
                rgba(20, 72, 43, 0.96),
                rgba(8, 33, 21, 0.96)
            );

        border: 1px solid rgba(76, 230, 130, 0.28);

        border-radius: 22px;

        padding: 30px 32px;

        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.22);
    }


    .forecast-label {
        color: #a2c2ad;

        font-size: 0.78rem;
        font-weight: 850;

        text-transform: uppercase;
        letter-spacing: 0.10em;
    }


    .forecast-number {
        color: #ffffff;

        font-size: 2.7rem;
        font-weight: 900;

        line-height: 1.15;

        margin: 7px 0;
    }


    .forecast-change {
        font-size: 1.05rem;
        font-weight: 820;
    }


    .forecast-grid {
        display: flex;
        flex-wrap: wrap;

        gap: 26px;

        margin-top: 20px;

        padding-top: 16px;

        border-top: 1px solid rgba(255, 255, 255, 0.10);
    }


    .forecast-grid-item {
        min-width: 150px;
    }


    .forecast-grid-label {
        color: #8fae9c;

        font-size: 0.70rem;
        font-weight: 800;

        text-transform: uppercase;
        letter-spacing: 0.09em;

        margin-bottom: 4px;
    }


    .forecast-grid-value {
        color: #ffffff;

        font-size: 0.98rem;
        font-weight: 700;
    }


    /* --------------------------------------------------------
       STREAMLIT NATIVE COMPONENTS
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: rgba(14, 29, 21, 0.82);

        border: 1px solid rgba(255, 255, 255, 0.065);

        border-radius: 15px;

        padding: 15px 17px;
    }


    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
    }


    div[data-baseweb="select"] > div {
        background-color: rgba(14, 29, 21, 0.92);

        border-color: rgba(255, 255, 255, 0.10);
    }


    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;

        border: 1px solid rgba(255, 255, 255, 0.06);
    }


    details[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);

        border: 1px solid rgba(255, 255, 255, 0.07);

        border-radius: 14px;

        overflow: hidden;
    }


    details[data-testid="stExpander"] summary {
        font-weight: 700;
        color: var(--text-soft);
    }


    hr {
        border-color: rgba(255, 255, 255, 0.07) !important;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;

        color: #718078;

        font-size: 0.82rem;

        padding: 12px 0 15px 0;
    }


    .footer strong {
        color: var(--text-soft);
    }

    </style>"""
