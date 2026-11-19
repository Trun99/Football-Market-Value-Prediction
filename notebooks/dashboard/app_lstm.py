# ============================================================
# FOOTBALL MARKET INTELLIGENCE
# LSTM + GRADIENT BOOSTING DEPLOYMENT DASHBOARD
# Final Year Project
# ============================================================

from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Football Market Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. HTML RENDERER
# ============================================================

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


# ============================================================
# 3. DASHBOARD THEME (premium dark football / analytics)
# ============================================================
#
# This stylesheet only affects presentation. It never touches
# model logic, feature order, sequence rules or predictions.
# ============================================================

render_html(
    """<style>

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
)


# ============================================================
# 4. PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


TEMPORAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_market_value_dataset.csv"
)


PLAYERS_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "players.csv"
)


VALUATIONS_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "player_valuations.csv"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "experiment_2"
)


GB_MODEL_PATH = (
    MODEL_DIR
    / "gradient_boosting_top4_future_value.joblib"
)


LSTM_MODEL_PATH = (
    MODEL_DIR
    / "lstm_future_market_value.keras"
)


LSTM_FEATURE_SCALER_PATH = (
    MODEL_DIR
    / "lstm_feature_scaler.joblib"
)


LSTM_TARGET_SCALER_PATH = (
    MODEL_DIR
    / "lstm_target_scaler.joblib"
)


# ============================================================
# 5. MODEL CONFIGURATION
# ============================================================

MODEL_FEATURES = [

    "current_market_value",

    "age_at_snapshot",

    "minutes_last_6m",

    "goals_last_6m",
]


SEQUENCE_LENGTH = 3


MAX_GAP_DAYS = 240


FEATURE_LABELS = {

    "current_market_value":
        "Current Market Value",

    "age_at_snapshot":
        "Age at Snapshot",

    "minutes_last_6m":
        "Minutes — Last 6 Months",

    "goals_last_6m":
        "Goals — Last 6 Months",
}


# ============================================================
# 7. SAFE VALUE CONVERSION
# ============================================================

def to_float(value):

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return None


    if not np.isfinite(value):

        return None


    return value


# ============================================================
# 8. FORMAT EURO
# ============================================================

def format_eur(value):

    value = to_float(value)


    if value is None:

        return "N/A"


    sign = (
        "-"
        if value < 0
        else ""
    )


    absolute_value = abs(value)


    if absolute_value >= 1_000_000_000:

        return (
            f"{sign}€"
            f"{absolute_value / 1_000_000_000:,.2f}B"
        )


    if absolute_value >= 1_000_000:

        return (
            f"{sign}€"
            f"{absolute_value / 1_000_000:,.2f}M"
        )


    if absolute_value >= 1_000:

        return (
            f"{sign}€"
            f"{absolute_value / 1_000:,.1f}K"
        )


    return (
        f"{sign}€"
        f"{absolute_value:,.0f}"
    )


# ============================================================
# 9. FORMAT INTEGER
# ============================================================

def format_integer(value):

    value = to_float(value)


    if value is None:

        return "N/A"


    return f"{value:,.0f}"


# ============================================================
# 10. FORMAT DECIMAL
# ============================================================

def format_decimal(
    value,
    decimals=1,
):

    value = to_float(value)


    if value is None:

        return "N/A"


    return (
        f"{value:.{decimals}f}"
    )


# ============================================================
# 11. SAFE TEXT
# ============================================================

def safe_text(value):

    if value is None:

        return "Not available"


    try:

        if pd.isna(value):

            return "Not available"

    except Exception:

        pass


    text = str(value).strip()


    if not text:

        return "Not available"


    return text


# ============================================================
# 12. FORMAT DATE
# ============================================================

def format_date(value):

    try:

        value = pd.Timestamp(value)

    except Exception:

        return "N/A"


    if pd.isna(value):

        return "N/A"


    return value.strftime(
        "%d %b %Y"
    )


# ============================================================
# 13. CHECK REQUIRED FILES
# ============================================================

REQUIRED_FILES = {

    "Temporal dataset":
        TEMPORAL_DATA_PATH,

    "Player profiles":
        PLAYERS_DATA_PATH,

    "Player valuations":
        VALUATIONS_DATA_PATH,

    "Gradient Boosting model":
        GB_MODEL_PATH,

    "LSTM model":
        LSTM_MODEL_PATH,

    "LSTM feature scaler":
        LSTM_FEATURE_SCALER_PATH,

    "LSTM target scaler":
        LSTM_TARGET_SCALER_PATH,
}


missing_files = []


for label, file_path in REQUIRED_FILES.items():

    if not file_path.exists():

        missing_files.append(
            (
                label,
                file_path,
            )
        )


if missing_files:

    st.error(
        "Required project files are missing."
    )


    for label, file_path in missing_files:

        st.write(
            f"**{label}:** "
            f"`{file_path}`"
        )


    st.stop()


# ============================================================
# 14. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    temporal = pd.read_csv(
        TEMPORAL_DATA_PATH,
        low_memory=False,
    )


    players = pd.read_csv(
        PLAYERS_DATA_PATH,
        low_memory=False,
    )


    valuations = pd.read_csv(
        VALUATIONS_DATA_PATH,
        low_memory=False,
    )


    temporal[
        "snapshot_date"
    ] = pd.to_datetime(
        temporal[
            "snapshot_date"
        ],
        errors="coerce",
    )


    if (
        "target_date"
        in temporal.columns
    ):

        temporal[
            "target_date"
        ] = pd.to_datetime(
            temporal[
                "target_date"
            ],
            errors="coerce",
        )


    valuations[
        "date"
    ] = pd.to_datetime(
        valuations[
            "date"
        ],
        errors="coerce",
    )


    for dataframe in [

        temporal,

        players,

        valuations,
    ]:

        dataframe[
            "player_id"
        ] = pd.to_numeric(
            dataframe[
                "player_id"
            ],
            errors="coerce",
        )


    for feature in MODEL_FEATURES:

        temporal[
            feature
        ] = pd.to_numeric(
            temporal[
                feature
            ],
            errors="coerce",
        )


    return (
        temporal,
        players,
        valuations,
    )


# ============================================================
# 15. LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    gradient_boosting = (
        joblib.load(
            GB_MODEL_PATH
        )
    )


    lstm = (
        keras.models.load_model(
            LSTM_MODEL_PATH,
            compile=False,
        )
    )


    feature_scaler = (
        joblib.load(
            LSTM_FEATURE_SCALER_PATH
        )
    )


    target_scaler = (
        joblib.load(
            LSTM_TARGET_SCALER_PATH
        )
    )


    return (
        gradient_boosting,
        lstm,
        feature_scaler,
        target_scaler,
    )


# ============================================================
# 16. LOAD PROJECT RESOURCES
# ============================================================

try:

    (
        temporal_df,
        players_df,
        valuations_df,
    ) = load_data()


    (
        gb_model,
        lstm_model,
        feature_scaler,
        target_scaler,
    ) = load_models()


except Exception as load_error:

    st.error(
        "The dashboard could not load "
        "the project resources."
    )


    st.exception(
        load_error
    )


    st.stop()


# ============================================================
# 17. PREPARE TEMPORAL DATA
# ============================================================

temporal_sorted = (

    temporal_df

    .dropna(
        subset=[
            "player_id",
            "snapshot_date",
        ]
    )

    .sort_values(
        [
            "player_id",
            "snapshot_date",
        ]
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 18. LATEST SNAPSHOT PER PLAYER
# ============================================================

latest_snapshots = (

    temporal_sorted

    .groupby(
        "player_id",
        as_index=False,
    )

    .tail(1)

    .copy()
)


# ============================================================
# 19. PLAYER PROFILE COLUMNS
# ============================================================

possible_profile_columns = [

    "player_id",

    "name",

    "first_name",

    "last_name",

    "image_url",

    "position",

    "sub_position",

    "current_club_id",

    "current_club_name",

    "country_of_citizenship",

    "date_of_birth",
]


profile_columns = [

    column

    for column
    in possible_profile_columns

    if column
    in players_df.columns
]


player_profiles = (

    players_df[
        profile_columns
    ]

    .drop_duplicates(
        subset=[
            "player_id"
        ]
    )

    .copy()
)


# ============================================================
# 20. MERGE PLAYER PROFILES
# ============================================================

player_data = (

    latest_snapshots

    .merge(

        player_profiles,

        on="player_id",

        how="left",

        suffixes=(
            "",
            "_profile",
        ),
    )
)


# ============================================================
# 21. RESOLVE DUPLICATE PROFILE COLUMNS
# ============================================================

profile_fields = [

    "name",

    "image_url",

    "position",

    "sub_position",

    "current_club_name",

    "country_of_citizenship",
]


for column in profile_fields:

    profile_column = (
        f"{column}_profile"
    )


    if (
        column
        not in player_data.columns

        and

        profile_column
        in player_data.columns
    ):

        player_data[
            column
        ] = player_data[
            profile_column
        ]


    elif (

        column
        in player_data.columns

        and

        profile_column
        in player_data.columns
    ):

        player_data[
            column
        ] = player_data[
            column
        ].fillna(
            player_data[
                profile_column
            ]
        )


# ============================================================
# 22. VERIFY PLAYER NAME
# ============================================================

if (
    "name"
    not in player_data.columns
):

    st.error(
        "No usable player name column "
        "was found."
    )

    st.stop()


# ============================================================
# 23. CHECK GB INPUT AVAILABILITY
# ============================================================

player_data[
    "gb_ready"
] = (

    player_data[
        MODEL_FEATURES
    ]

    .notna()

    .all(
        axis=1
    )
)


prediction_players = (

    player_data[

        player_data[
            "name"
        ].notna()

        &

        player_data[
            "gb_ready"
        ]

    ]

    .copy()
)


# ============================================================
# 24. GET PLAYER HISTORY
# ============================================================

def get_player_history(
    player_id,
):

    history = (

        temporal_sorted[

            temporal_sorted[
                "player_id"
            ] == player_id

        ]

        .sort_values(
            "snapshot_date"
        )

        .reset_index(
            drop=True
        )
    )


    return history


# ============================================================
# 25. CHECK LSTM ELIGIBILITY
# ============================================================

def get_lstm_sequence(
    player_id,
):

    history = (
        get_player_history(
            player_id
        )
    )


    # --------------------------------------------------------
    # Need at least 3 observations
    # --------------------------------------------------------

    if (
        len(history)
        < SEQUENCE_LENGTH
    ):

        return {

            "eligible":
                False,

            "reason":
                (
                    f"Only {len(history)} "
                    "historical snapshot(s) "
                    "are available. "
                    f"At least {SEQUENCE_LENGTH} "
                    "are required."
                ),

            "sequence":
                None,

            "gaps":
                None,
        }


    # --------------------------------------------------------
    # Latest 3 observations
    # --------------------------------------------------------

    sequence = (

        history

        .tail(
            SEQUENCE_LENGTH
        )

        .copy()

        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    if (

        sequence[
            MODEL_FEATURES
        ]

        .isna()

        .any()

        .any()
    ):

        return {

            "eligible":
                False,

            "reason":
                (
                    "The latest three observations "
                    "contain missing model features."
                ),

            "sequence":
                sequence,

            "gaps":
                None,
        }


    # --------------------------------------------------------
    # Calculate temporal gaps
    # --------------------------------------------------------

    gaps = (

        sequence[
            "snapshot_date"
        ]

        .diff()

        .dt.days

        .dropna()

        .to_numpy(
            dtype=float
        )
    )


    if (
        len(gaps)
        != SEQUENCE_LENGTH - 1
    ):

        return {

            "eligible":
                False,

            "reason":
                (
                    "The sequence dates could "
                    "not be validated."
                ),

            "sequence":
                sequence,

            "gaps":
                gaps,
        }


    # --------------------------------------------------------
    # Validate gaps
    # --------------------------------------------------------

    invalid_gap = (

        (gaps <= 0).any()

        or

        (
            gaps
            > MAX_GAP_DAYS
        ).any()
    )


    if invalid_gap:

        gap_text = ", ".join(

            str(
                int(gap)
            )

            for gap
            in gaps
        )


        return {

            "eligible":
                False,

            "reason":
                (
                    "The latest sequence violates "
                    f"the {MAX_GAP_DAYS}-day "
                    "maximum gap constraint. "
                    f"Observed gaps: "
                    f"{gap_text} days."
                ),

            "sequence":
                sequence,

            "gaps":
                gaps,
        }


    return {

        "eligible":
            True,

        "reason":
            None,

        "sequence":
            sequence,

        "gaps":
            gaps,
    }


# ============================================================
# 26. LSTM PREDICTION
# ============================================================

def predict_with_lstm(
    sequence,
):

    raw_sequence = (

        sequence[
            MODEL_FEATURES
        ]

        .to_numpy(
            dtype=np.float64
        )
    )


    expected_shape = (

        SEQUENCE_LENGTH,

        len(
            MODEL_FEATURES
        ),
    )


    if (
        raw_sequence.shape
        != expected_shape
    ):

        raise ValueError(

            "Invalid LSTM input shape. "

            f"Expected "
            f"{expected_shape}, "

            f"received "
            f"{raw_sequence.shape}."
        )


    if (
        not
        np.isfinite(
            raw_sequence
        ).all()
    ):

        raise ValueError(
            "LSTM input contains "
            "NaN or infinite values."
        )


    # --------------------------------------------------------
    # Scale the 3 x 4 matrix
    # --------------------------------------------------------

    scaled_sequence_2d = (

        feature_scaler

        .transform(
            raw_sequence
        )
    )


    # --------------------------------------------------------
    # Convert to:
    # (batch, timesteps, features)
    # = (1, 3, 4)
    # --------------------------------------------------------

    scaled_sequence_3d = (

        scaled_sequence_2d

        .reshape(

            1,

            SEQUENCE_LENGTH,

            len(
                MODEL_FEATURES
            ),
        )
    )


    # --------------------------------------------------------
    # Predict scaled target
    # --------------------------------------------------------

    scaled_prediction = (

        lstm_model

        .predict(

            scaled_sequence_3d,

            verbose=0,
        )

        .reshape(
            -1,
            1,
        )
    )


    # --------------------------------------------------------
    # Convert prediction back to EUR
    # --------------------------------------------------------

    prediction_eur = (

        target_scaler

        .inverse_transform(
            scaled_prediction
        )

        .reshape(-1)[0]
    )


    prediction_eur = float(
        prediction_eur
    )


    if (
        not
        np.isfinite(
            prediction_eur
        )
    ):

        raise ValueError(
            "LSTM produced a "
            "non-finite prediction."
        )


    # --------------------------------------------------------
    # Market value cannot be negative
    # --------------------------------------------------------

    return max(
        0.0,
        prediction_eur,
    )


# ============================================================
# 27. GRADIENT BOOSTING PREDICTION
# ============================================================

def predict_with_gradient_boosting(
    player_row,
):

    model_input = pd.DataFrame(

        [
            {

                feature:
                    float(
                        player_row[
                            feature
                        ]
                    )

                for feature
                in MODEL_FEATURES
            }
        ]
    )


    prediction = float(

        gb_model

        .predict(
            model_input
        )[0]
    )


    if (
        not
        np.isfinite(
            prediction
        )
    ):

        raise ValueError(
            "Gradient Boosting produced "
            "a non-finite prediction."
        )


    prediction = max(
        0.0,
        prediction,
    )


    return (
        prediction,
        model_input,
    )


# ============================================================
# 28. AUTOMATIC MODEL ROUTER
# ============================================================

def predict_player(
    player_row,
):

    player_id = int(
        player_row[
            "player_id"
        ]
    )


    lstm_check = (

        get_lstm_sequence(
            player_id
        )
    )


    # --------------------------------------------------------
    # Primary model = LSTM
    # --------------------------------------------------------

    if (
        lstm_check[
            "eligible"
        ]
    ):

        try:

            prediction = (

                predict_with_lstm(

                    lstm_check[
                        "sequence"
                    ]
                )
            )


            return {

                "success":
                    True,

                "model":
                    "LSTM",

                "model_display":
                    "LSTM Sequence Model",

                "prediction":
                    prediction,

                "sequence":
                    lstm_check[
                        "sequence"
                    ],

                "gaps":
                    lstm_check[
                        "gaps"
                    ],

                "fallback_reason":
                    None,

                "gb_input":
                    None,
            }


        except Exception as lstm_error:

            fallback_reason = (

                "A valid sequence was found, "
                "but the LSTM prediction "
                "failed at runtime. "

                "Gradient Boosting was used "
                "instead. "

                "Technical reason: "

                f"{str(lstm_error)}"
            )


    else:

        fallback_reason = (

            lstm_check[
                "reason"
            ]
        )


    # --------------------------------------------------------
    # Gradient Boosting fallback
    # --------------------------------------------------------

    (
        prediction,
        gb_input,
    ) = (

        predict_with_gradient_boosting(
            player_row
        )
    )


    return {

        "success":
            True,

        "model":
            "Gradient Boosting",

        "model_display":
            "Gradient Boosting Fallback",

        "prediction":
            prediction,

        "sequence":
            lstm_check[
                "sequence"
            ],

        "gaps":
            lstm_check[
                "gaps"
            ],

        "fallback_reason":
            fallback_reason,

        "gb_input":
            gb_input,
    }


# ============================================================
# 29. HEADER
# ============================================================

render_html(
    '<div class="main-title">'
    "Football Market Intelligence"
    "</div>"
    '<div class="main-subtitle">'
    "AI-Based Football Transfer Market Value Prediction"
    "</div>"
    '<div class="research-pill">'
    "Research Prototype"
    "</div>"
)


st.caption(

    "Final Year Project research prototype. "

    "Forecasts use the latest available historical "
    "dataset and are neither live valuations nor "

    "actual transfer-fee estimates."
)


# ============================================================
# 30. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "⚽ Model Intelligence"
    )


    st.caption(
        "Experiment 2 — "
        "Future Market Value Prediction"
    )


    st.divider()


    st.subheader(
        "Deployment"
    )


    st.write(
        "**Primary model:** "
        "LSTM sequence model"
    )


    st.write(
        "**Fallback model:** "
        "Gradient Boosting"
    )


    st.write(
        "**Forecast horizon:** "
        "~6 months"
    )


    st.write(
        f"**Sequence length:** "
        f"{SEQUENCE_LENGTH} snapshots"
    )


    st.write(
        f"**Maximum gap:** "
        f"{MAX_GAP_DAYS} days"
    )


    st.write(
        f"**Model features:** "
        f"{len(MODEL_FEATURES)}"
    )


    for feature in MODEL_FEATURES:

        st.caption(
            f"• {FEATURE_LABELS[feature]}"
        )


    st.divider()


    latest_dataset_date = (

        temporal_sorted[
            "snapshot_date"
        ]

        .max()
    )


    st.caption(
        "Latest dataset observation"
    )


    st.write(
        f"**"
        f"{format_date(latest_dataset_date)}"
        f"**"
    )


    st.caption(
        "Historical dataset — not a live "
        "valuation feed."
    )


# ============================================================
# 31. BUILD PLAYER SEARCH OPTIONS
# ============================================================

search_options = (

    prediction_players[
        [
            "player_id",
            "name",
        ]
    ]

    .copy()
)


if (
    "current_club_name"
    in prediction_players.columns
):

    search_options[
        "club"
    ] = (

        prediction_players[
            "current_club_name"
        ]

        .fillna(
            "Club unavailable"
        )

        .astype(str)
    )


else:

    search_options[
        "club"
    ] = (
        "Club unavailable"
    )


# The search label also carries the latest known position so the
# user can distinguish players with the same name.

if (
    "position"
    in prediction_players.columns
):

    search_options[
        "position"
    ] = (

        prediction_players[
            "position"
        ]

        .fillna(
            "Position unavailable"
        )

        .astype(str)
    )


else:

    search_options[
        "position"
    ] = (
        "Position unavailable"
    )


search_options[
    "search_label"
] = (

    search_options[
        "name"
    ].astype(str)

    + " • "

    + search_options[
        "club"
    ]

    + " • "

    + search_options[
        "position"
    ]

    + " • ID "

    + search_options[
        "player_id"
    ]

    .astype(int)

    .astype(str)
)


search_options = (

    search_options

    .sort_values(
        [
            "name",
            "player_id",
        ]
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# 32. PLAYER SEARCH UI
# ============================================================

render_section(

    "Player Search",

    "Search the historical dataset and generate an "
    "approximately six-month future market value forecast.",
)


selected_label = (

    st.selectbox(

        "Player",

        options=
            search_options[
                "search_label"
            ].tolist(),

        index=None,

        placeholder=
            "Type a player name...",

        label_visibility=
            "collapsed",
    )
)


if (
    selected_label
    is None
):

    render_html(
        '<div class="info-panel">'
        "Search for a player above to view their historical "
        "market-value journey, recent form, LSTM modelling "
        "sequence and the AI forecast."
        "</div>"
    )


    st.stop()


# ============================================================
# 33. SELECT PLAYER
# ============================================================

selected_player_id = int(

    search_options.loc[

        search_options[
            "search_label"
        ] == selected_label,

        "player_id",

    ].iloc[0]
)


selected_rows = (

    prediction_players[

        prediction_players[
            "player_id"
        ] == selected_player_id

    ]

    .sort_values(
        "snapshot_date"
    )
)


if (
    selected_rows.empty
):

    st.error(
        "The selected player "
        "could not be found."
    )

    st.stop()


selected_player = (

    selected_rows

    .iloc[-1]
)


snapshot_date = (

    pd.Timestamp(

        selected_player[
            "snapshot_date"
        ]
    )
)

# ============================================================
# 33b. FORECAST ACTION
# ============================================================

generate_forecast = st.button(
    "Predict",
    type="primary",
    use_container_width=True,
)

if not generate_forecast:
    render_html(
        '<div class="info-panel">'
        "Player selected. Click "
        "<strong>Predict</strong> "
        "to generate the future market value forecast. "
        "This action does not retrain either model."
        "</div>"
    )

    st.stop()

# ============================================================
# 34. GENERATE PREDICTION
# ============================================================

try:

    prediction_result = (

        predict_player(
            selected_player
        )
    )


except Exception as prediction_error:

    st.error(
        "The prediction could "
        "not be generated."
    )


    st.exception(
        prediction_error
    )


    st.stop()


# ============================================================
# 35. CALCULATE FORECAST VALUES
# ============================================================

predicted_value = float(

    prediction_result[
        "prediction"
    ]
)


current_value = float(

    selected_player[
        "current_market_value"
    ]
)


absolute_change = (

    predicted_value
    - current_value
)


if (
    current_value > 0
):

    percentage_change = (

        absolute_change

        / current_value

        * 100
    )


else:

    percentage_change = np.nan


forecast_date = (

    snapshot_date

    + pd.DateOffset(
        months=6
    )
)


# ============================================================
# 36. CHANGE DIRECTION
# ============================================================

if (
    absolute_change > 0
):

    change_class = (
        "positive-value"
    )

    change_arrow = "▲"


elif (
    absolute_change < 0
):

    change_class = (
        "negative-value"
    )

    change_arrow = "▼"


else:

    change_class = (
        "neutral-value"
    )

    change_arrow = "—"


# ============================================================
# 37. FORMAT FORECAST VALUES
# ============================================================

current_value_text = (

    format_eur(
        current_value
    )
)


predicted_value_text = (

    format_eur(
        predicted_value
    )
)


absolute_change_text = (

    format_eur(
        absolute_change
    )
)


if (
    np.isfinite(
        percentage_change
    )
):

    percentage_change_text = (

        f"{percentage_change:+.2f}%"
    )


else:

    percentage_change_text = (
        "N/A"
    )


# ============================================================
# 38. PLAYER PROFILE VALUES
# ============================================================

player_name = (

    safe_text(

        selected_player.get(
            "name"
        )
    )
)


club_name = (

    safe_text(

        selected_player.get(
            "current_club_name"
        )
    )
)


position = (

    safe_text(

        selected_player.get(
            "position"
        )
    )
)


sub_position = (

    safe_text(

        selected_player.get(
            "sub_position"
        )
    )
)


nationality = (

    safe_text(

        selected_player.get(
            "country_of_citizenship"
        )
    )
)


image_url = (

    selected_player.get(
        "image_url"
    )
)


# ============================================================
# 39. PLAYER OVERVIEW
# ============================================================

st.divider()


render_section(
    "Player Overview",
    "Identity and latest available dataset snapshot "
    "for the selected player.",
)


snapshot_date_text = (

    format_date(
        snapshot_date
    )
)


image_column, profile_column = (

    st.columns(
        [
            1,
            4,
        ],
        gap="large",
    )
)


with image_column:

    if (

        image_url
        is not None

        and

        not pd.isna(
            image_url
        )
    ):

        try:

            st.image(
                image_url,
                width="stretch",
            )


        except Exception:

            st.info(
                "Player image unavailable."
            )


    else:

        st.info(
            "Player image unavailable."
        )


with profile_column:

    if (
        prediction_result[
            "model"
        ] == "LSTM"
    ):

        badge_html = (
            '<div class="badge-lstm">'
            "LSTM Sequence Model"
            "</div>"
        )


    else:

        badge_html = (
            '<div class="badge-gb">'
            "Gradient Boosting Fallback"
            "</div>"
        )


    render_html(
        '<div class="hero-card">'
        '<div class="player-name">'
        f"{player_name}"
        "</div>"
        '<div class="player-club">'
        f"{club_name}"
        "</div>"
        '<div class="player-meta">'
        f"{position}"
        " &nbsp; • &nbsp; "
        f"{sub_position}"
        " &nbsp; • &nbsp; "
        f"{nationality}"
        "<br>"
        "Latest snapshot: "
        f"{snapshot_date_text}"
        "</div>"
        f"{badge_html}"
        "</div>"
    )


# ============================================================
# 40. FORECAST OVERVIEW
# ============================================================

render_section(
    "Forecast Overview",
    "Current position, the approximately six-month "
    "forecast and the implied change.",
)


forecast_date_text = (

    format_date(
        forecast_date
    )
)


overview_1, overview_2, overview_3 = (

    st.columns(
        3,
        gap="medium",
    )
)


with overview_1:

    render_html(
        '<div class="dashboard-card">'
        '<div class="card-label">'
        "Current Market Value"
        "</div>"
        '<div class="card-value">'
        f"{current_value_text}"
        "</div>"
        '<div class="card-note">'
        "Latest available dataset observation"
        "</div>"
        "</div>"
    )


with overview_2:

    render_html(
        '<div class="dashboard-card">'
        '<div class="card-label">'
        "Predicted Market Value (~6 months)"
        "</div>"
        '<div class="card-value">'
        f"{predicted_value_text}"
        "</div>"
        '<div class="card-note">'
        "Approximate forecast date: "
        f"{forecast_date_text}"
        "</div>"
        "</div>"
    )


with overview_3:

    render_html(
        '<div class="dashboard-card">'
        '<div class="card-label">'
        "Forecast Change"
        "</div>"
        '<div class="card-value '
        f"{change_class}"
        '">'
        f"{change_arrow} "
        f"{absolute_change_text}"
        "</div>"
        '<div class="card-note">'
        "Percentage change: "
        f"{percentage_change_text}"
        "</div>"
        "</div>"
    )


# ============================================================
# 42. MARKET VALUE JOURNEY
# ============================================================

render_section(
    "Market Value Journey",
    "Historical valuations up to the model snapshot, "
    "followed by the AI forecast point.",
)


valuation_history = (

    valuations_df[

        valuations_df[
            "player_id"
        ] == selected_player_id

    ]

    .copy()
)


valuation_history = (

    valuation_history[

        valuation_history[
            "date"
        ] <= snapshot_date

    ]

    .sort_values(
        "date"
    )
)


if (

    not valuation_history.empty

    and

    "market_value_in_eur"
    in valuation_history.columns
):

    chart_history = (

        valuation_history[
            [
                "date",
                "market_value_in_eur",
            ]
        ]

        .dropna()

        .rename(

            columns={

                "market_value_in_eur":
                    "Market Value (€)"
            }
        )

        .set_index(
            "date"
        )
    )


    forecast_points = (

        pd.DataFrame(

            {

                "Market Value (€)":
                    [
                        current_value,
                        predicted_value,
                    ]
            },

            index=[
                snapshot_date,
                forecast_date,
            ],
        )
    )


    chart_data = (

        pd.concat(
            [
                chart_history,
                forecast_points,
            ]
        )
    )


    chart_data = (

        chart_data[

            ~chart_data.index

            .duplicated(
                keep="last"
            )
        ]

        .sort_index()
    )


    st.line_chart(

        chart_data,

        height=390,
    )


    st.caption(

        "Historical values are restricted "
        "to observations on or before "
        "the model snapshot. "

        "The final point represents "
        "the AI forecast."
    )


else:

    st.info(

        "Historical market-value "
        "information is unavailable "
        "for this player."
    )


# ============================================================
# 45. LSTM HISTORICAL SEQUENCE
# ============================================================

if (
    prediction_result[
        "model"
    ] == "LSTM"
):

    render_section(
        "LSTM Historical Sequence",
        "The primary model consumes the latest three valid "
        "observations: t-2 → t-1 → t.",
    )


    sequence = (

        prediction_result[
            "sequence"
        ]

        .copy()

        .reset_index(
            drop=True
        )
    )


    sequence_labels = [

        "t-2",

        "t-1",

        "t",
    ]


    sequence_columns = (

        st.columns(
            SEQUENCE_LENGTH,
            gap="medium",
        )
    )


    for index in range(
        SEQUENCE_LENGTH
    ):

        row = (

            sequence.iloc[
                index
            ]
        )


        # ----------------------------------------------------
        # Format every value BEFORE it is inserted into HTML.
        #
        # This avoids complex format expressions inside the
        # f-string and prevents the previous
        # "Invalid format specifier '.1f'" problem.
        # ----------------------------------------------------

        sequence_label_text = (

            sequence_labels[
                index
            ]
        )


        sequence_date_text = (

            format_date(

                row[
                    "snapshot_date"
                ]
            )
        )


        sequence_value_text = (

            format_eur(

                row[
                    "current_market_value"
                ]
            )
        )


        sequence_age_text = (

            format_decimal(

                row[
                    "age_at_snapshot"
                ],

                decimals=1,
            )
        )


        sequence_minutes_text = (

            format_integer(

                row[
                    "minutes_last_6m"
                ]
            )
        )


        sequence_goals_text = (

            format_integer(

                row[
                    "goals_last_6m"
                ]
            )
        )


        # The most recent observation (t) is highlighted.

        if (
            index
            == SEQUENCE_LENGTH - 1
        ):

            sequence_card_class = (
                "sequence-card sequence-card-active"
            )


        else:

            sequence_card_class = (
                "sequence-card"
            )


        sequence_html = (

            f'<div class="{sequence_card_class}">'

            '<div class="sequence-label">'

            f"{sequence_label_text}"

            "</div>"

            '<div class="sequence-date">'

            f"{sequence_date_text}"

            "</div>"

            '<div class="sequence-market-value">'

            f"{sequence_value_text}"

            "</div>"

            '<div class="sequence-detail">'

            "Age: "

            f"{sequence_age_text}"

            "<br>"

            "Minutes: "

            f"{sequence_minutes_text}"

            "<br>"

            "Goals: "

            f"{sequence_goals_text}"

            "</div>"

            "</div>"
        )


        with sequence_columns[
            index
        ]:

            render_html(
                sequence_html
            )


    # --------------------------------------------------------
    # Display temporal gaps
    # --------------------------------------------------------

    gaps = (

        prediction_result[
            "gaps"
        ]
    )


    if (

        gaps
        is not None

        and

        len(gaps) == 2
    ):

        gap_1_text = str(
            int(
                gaps[0]
            )
        )


        gap_2_text = str(
            int(
                gaps[1]
            )
        )


        render_html(
            '<div class="info-panel">'
            "<strong>Sequence validation</strong><br>"
            "Gap t-2 → t-1: "
            f"{gap_1_text} days"
            " &nbsp; • &nbsp; "
            "Gap t-1 → t: "
            f"{gap_2_text} days"
            "<br>"
            "Both gaps satisfy the maximum "
            f"{MAX_GAP_DAYS}-day requirement."
            "</div>"
        )


    # ========================================================

# ============================================================
# 46. SEQUENCE UNAVAILABLE
# ============================================================

else:

    render_section(
        "Sequence Availability",
    )


    fallback_reason_text = (

        safe_text(

            prediction_result[
                "fallback_reason"
            ]
        )
    )


    render_html(
        '<div class="routing-warning">'
        "A valid LSTM sequence could not be constructed "
        "for this player."
        "<br><br>"
        "<strong>Reason:</strong> "
        f"{fallback_reason_text}"
        "<br><br>"
        "The displayed forecast therefore uses the "
        "Gradient Boosting fallback model."
        "</div>"
    )



# ============================================================
# 45c. RECENT FORM
# ============================================================

render_section(
    "Recent Form",
    "Playing time and output over the six months "
    "preceding the latest snapshot.",
)


form_1, form_2, form_3, form_4 = (

    st.columns(
        4,
        gap="medium",
    )
)


with form_1:

    st.metric(

        "Games",

        format_integer(

            selected_player.get(
                "games_last_6m"
            )
        ),
    )


with form_2:

    st.metric(

        "Minutes",

        format_integer(

            selected_player.get(
                "minutes_last_6m"
            )
        ),
    )


with form_3:

    st.metric(

        "Goals",

        format_integer(

            selected_player.get(
                "goals_last_6m"
            )
        ),
    )


with form_4:

    st.metric(

        "Assists",

        format_integer(

            selected_player.get(
                "assists_last_6m"
            )
        ),
    )


# ------------------------------------------------------------
# RECENT-FORM SOURCE-DATA COVERAGE NOTICE
# ------------------------------------------------------------

games_last_6m_value = to_float(
    selected_player.get(
        "games_last_6m"
    )
)

minutes_last_6m_value = to_float(
    selected_player.get(
        "minutes_last_6m"
    )
)

no_recorded_appearances = (
    games_last_6m_value is not None
    and minutes_last_6m_value is not None
    and games_last_6m_value == 0
    and minutes_last_6m_value == 0
)

if no_recorded_appearances:

    render_html(
        '<div class="routing-warning">'
        "<strong>Source-data coverage notice.</strong> "
        "No appearances were recorded for this player "
        "within the six-month performance window in the "
        "source dataset. The zero-valued recent-performance "
        "features may therefore represent genuine inactivity "
        "or limited competition coverage. They should not "
        "automatically be interpreted as confirmed zero "
        "real-world performance."
        "</div>"
    )

# ============================================================
# 45d. CURRENT MODEL SIGNALS
# ============================================================

render_section(
    "Current Model Signals",
    "The four features consumed by the final models "
    "at the latest observation.",
)


signal_1, signal_2, signal_3, signal_4 = (

    st.columns(
        4,
        gap="medium",
    )
)


with signal_1:

    st.metric(

        "Market Value",

        format_eur(

            selected_player[
                "current_market_value"
            ]
        ),
    )


with signal_2:

    age_signal_text = (

        format_decimal(

            selected_player[
                "age_at_snapshot"
            ],

            decimals=1,
        )
    )


    if (
        age_signal_text
        != "N/A"
    ):

        age_signal_text = (

            age_signal_text
            + " years"
        )


    st.metric(

        "Age",

        age_signal_text,
    )


with signal_3:

    st.metric(

        "Minutes / 6m",

        format_integer(

            selected_player[
                "minutes_last_6m"
            ]
        ),
    )


with signal_4:

    st.metric(

        "Goals / 6m",

        format_integer(

            selected_player[
                "goals_last_6m"
            ]
        ),
    )

if no_recorded_appearances:

    st.caption(
        "Modelling note: the deployed model retains "
        "zero-valued Minutes / 6m and Goals / 6m inputs "
        "for this observation, consistent with the "
        "representation used during model development."
    )

# ============================================================
# 47. FINAL AI FORECAST
# ============================================================

render_section(
    "Final AI Forecast",
    "Headline output for the selected player.",
)


model_display_text = (

    safe_text(

        prediction_result[
            "model_display"
        ]
    )
)


latest_date_text = (

    format_date(
        snapshot_date
    )
)


render_html(
    '<div class="forecast-card">'
    '<div class="forecast-label">'
    "Predicted Future Market Value (~6 months)"
    "</div>"
    '<div class="forecast-number">'
    f"{predicted_value_text}"
    "</div>"
    '<div class="forecast-change '
    f"{change_class}"
    '">'
    f"{change_arrow} "
    f"{absolute_change_text}"
    " &nbsp;("
    f"{percentage_change_text}"
    ")"
    "</div>"
    '<div class="forecast-grid">'
    '<div class="forecast-grid-item">'
    '<div class="forecast-grid-label">Model Used</div>'
    '<div class="forecast-grid-value">'
    f"{model_display_text}"
    "</div>"
    "</div>"
    '<div class="forecast-grid-item">'
    '<div class="forecast-grid-label">Latest Observation</div>'
    '<div class="forecast-grid-value">'
    f"{latest_date_text}"
    "</div>"
    "</div>"
    '<div class="forecast-grid-item">'
    '<div class="forecast-grid-label">Approximate Forecast Date</div>'
    '<div class="forecast-grid-value">'
    f"{forecast_date_text}"
    "</div>"
    "</div>"
    "</div>"
    "</div>"
)


# ============================================================
# 55. FOOTER
# ============================================================

st.divider()


render_html(
    '<div class="footer">'
    "<strong>Final Year Project</strong>"
    " &nbsp; • &nbsp; "
    "AI-Based Football Transfer Market Value Prediction"
    " &nbsp; • &nbsp; "
    "Research Prototype"
    "</div>"
)