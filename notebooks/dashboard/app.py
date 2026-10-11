# ============================================================
# Football Market Value Prediction Dashboard
# Final Year Project
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Football Market Value Prediction",
    page_icon="⚽",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "experiment_2"
    / "gradient_boosting_top4_future_value.joblib"
)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    temporal_df = pd.read_csv(
        TEMPORAL_DATA_PATH
    )

    players_df = pd.read_csv(
        PLAYERS_DATA_PATH
    )

    valuations_df = pd.read_csv(
        VALUATIONS_DATA_PATH
    )

    temporal_df["snapshot_date"] = pd.to_datetime(
        temporal_df["snapshot_date"],
        errors="coerce"
    )

    valuations_df["date"] = pd.to_datetime(
        valuations_df["date"],
        errors="coerce"
    )

    return (
        temporal_df,
        players_df,
        valuations_df
    )


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# 5. LOAD PROJECT RESOURCES
# ============================================================

temporal_df, players_df, valuations_df = load_data()

final_model = load_model()


# ============================================================
# 6. FINAL MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "current_market_value",
    "age_at_snapshot",
    "minutes_last_6m",
    "goals_last_6m"
]


FEATURE_LABELS = {
    "current_market_value": "Current Market Value",
    "age_at_snapshot": "Age at Snapshot",
    "minutes_last_6m": "Minutes - Last 6 Months",
    "goals_last_6m": "Goals - Last 6 Months"
}


# ============================================================
# 7. BUILD LATEST PLAYER SNAPSHOTS
# ============================================================

temporal_sorted = temporal_df.sort_values(
    [
        "player_id",
        "snapshot_date"
    ]
)

latest_snapshots = (
    temporal_sorted
    .groupby(
        "player_id",
        as_index=False
    )
    .tail(1)
    .copy()
)


# ============================================================
# 8. PLAYER PROFILE DATA
# ============================================================

profile_columns = [
    "player_id",
    "name",
    "image_url",
    "position",
    "sub_position",
    "current_club_name",
    "country_of_citizenship",
    "date_of_birth"
]

player_profiles = (
    players_df[
        profile_columns
    ]
    .drop_duplicates(
        subset="player_id"
    )
    .copy()
)

player_data = latest_snapshots.merge(
    player_profiles,
    on="player_id",
    how="left",
    suffixes=("", "_profile")
)


# ============================================================
# 9. PREDICTION-READY PLAYERS
# ============================================================

player_data["prediction_ready"] = (
    player_data[
        MODEL_FEATURES
    ]
    .notna()
    .all(axis=1)
)

prediction_players = (
    player_data[
        player_data["prediction_ready"]
    ]
    .copy()
)

prediction_players = (
    prediction_players[
        prediction_players["name"].notna()
    ]
    .copy()
)


# ============================================================
# 10. SHAP COMPONENTS
# ============================================================

preprocessor = (
    final_model.named_steps[
        "preprocessor"
    ]
)

regressor = (
    final_model.named_steps[
        "regressor"
    ]
)


@st.cache_resource
def create_shap_explainer():

    return shap.TreeExplainer(
        regressor
    )


shap_explainer = create_shap_explainer()


# ============================================================
# 11. DASHBOARD HEADER
# ============================================================

st.title(
    "⚽ Football Market Value Prediction"
)

st.write(
    """
    AI-based dashboard for predicting a football player's
    future market value approximately six months ahead.
    """
)

st.caption(
    "Research prototype based on historical football data. "
    "Predictions use the latest available snapshot for each "
    "player in the project dataset and should not be "
    "interpreted as live market valuations."
)

st.divider()


# ============================================================
# 12. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🤖 Model Information"
    )

    st.write(
        "**Experiment:** Future Market Value Prediction"
    )

    st.write(
        "**Method:** Supervised Regression"
    )

    st.write(
        "**Algorithm:** Gradient Boosting Regressor"
    )

    st.write(
        "**Forecast Horizon:** Approximately 6 Months"
    )

    st.write(
        "**Features:** 4"
    )

    st.divider()

    st.subheader(
        "Final Features"
    )

    st.write(
        "1. Current Market Value"
    )

    st.write(
        "2. Age at Snapshot"
    )

    st.write(
        "3. Minutes - Last 6 Months"
    )

    st.write(
        "4. Goals - Last 6 Months"
    )

    st.divider()

    st.subheader(
        "Test Performance"
    )

    st.metric(
        "Test R²",
        "0.9175"
    )

    st.metric(
        "Test MAE",
        "€1.42M"
    )

    st.metric(
        "Test RMSE",
        "€3.57M"
    )

    st.divider()

    st.caption(
        "The final model was selected using temporal "
        "validation and evaluated on an unseen future-period "
        "test set."
    )


# ============================================================
# 13. PLAYER SEARCH
# ============================================================

st.subheader(
    "🔎 Player Search"
)

player_names = (
    prediction_players["name"]
    .dropna()
    .sort_values()
    .unique()
)

selected_name = st.selectbox(
    "Search for a player",
    options=player_names,
    index=None,
    placeholder="Type a player name..."
)


# ============================================================
# 14. SELECTED PLAYER
# ============================================================

if selected_name is not None:

    selected_player = (
        prediction_players[
            prediction_players["name"]
            == selected_name
        ]
        .sort_values(
            "snapshot_date"
        )
        .iloc[-1]
    )

    selected_player_id = (
        selected_player["player_id"]
    )

    snapshot_date = (
        selected_player["snapshot_date"]
    )

    st.divider()


    # ========================================================
    # 15. PLAYER PROFILE
    # ========================================================

    st.subheader(
        "👤 Player Profile"
    )

    image_col, info_col = st.columns(
        [1, 3]
    )

    with image_col:

        player_image = (
            selected_player["image_url"]
        )

        if pd.notna(player_image):

            st.image(
                player_image,
                width=220
            )

        else:

            st.info(
                "Player image not available."
            )


    with info_col:

        st.header(
            selected_player["name"]
        )

        club_name = (
            selected_player[
                "current_club_name"
            ]
        )

        position = (
            selected_player[
                "position"
            ]
        )

        sub_position = (
            selected_player[
                "sub_position"
            ]
        )

        nationality = (
            selected_player[
                "country_of_citizenship"
            ]
        )

        st.write(
            f"**Club:** {club_name}"
        )

        st.write(
            f"**Position:** {position}"
        )

        st.write(
            f"**Sub-position:** {sub_position}"
        )

        st.write(
            f"**Nationality:** {nationality}"
        )

        if pd.notna(snapshot_date):

            st.write(
                f"**Snapshot Date:** "
                f"{snapshot_date.date()}"
            )


    st.divider()


    # ========================================================
    # 16. MARKET VALUE HISTORY
    # ========================================================

    st.subheader(
        "📈 Market Value History"
    )

    player_history_all = (
        valuations_df[
            valuations_df["player_id"]
            == selected_player_id
        ]
        .copy()
    )

    player_history_all = (
        player_history_all
        .dropna(
            subset=[
                "date",
                "market_value_in_eur"
            ]
        )
        .sort_values(
            "date"
        )
    )


    # IMPORTANT:
    # Only show information available up to the player's
    # selected prediction snapshot.
    player_history = (
        player_history_all[
            player_history_all["date"]
            <= snapshot_date
        ]
        .copy()
    )


    if not player_history.empty:

        history_chart = (
            player_history[
                [
                    "date",
                    "market_value_in_eur"
                ]
            ]
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

        st.line_chart(
            history_chart,
            y="Market Value (€)",
            use_container_width=True
        )

        first_valuation = (
            player_history.iloc[0][
                "market_value_in_eur"
            ]
        )

        latest_observed_valuation = (
            player_history.iloc[-1][
                "market_value_in_eur"
            ]
        )

        history_col1, history_col2, history_col3 = (
            st.columns(3)
        )

        with history_col1:

            st.metric(
                "Historical Records",
                f"{len(player_history):,}"
            )

        with history_col2:

            st.metric(
                "First Recorded Value",
                f"€{first_valuation / 1_000_000:.2f}M"
            )

        with history_col3:

            st.metric(
                "Latest Value Before Snapshot",
                f"€{latest_observed_valuation / 1_000_000:.2f}M"
            )

        st.caption(
            "Only valuations recorded on or before the "
            "selected prediction snapshot are displayed."
        )

    else:

        st.info(
            "No historical market valuation data is "
            "available before this player's snapshot."
        )


    st.divider()


    # ========================================================
    # 17. RECENT PERFORMANCE
    # ========================================================

    st.subheader(
        "⚽ Recent Performance — Last 6 Months"
    )

    st.write(
        "Performance statistics recorded during the six "
        "months before the selected player's latest snapshot."
    )

    games_6m = selected_player.get(
        "games_last_6m",
        np.nan
    )

    goals_6m = selected_player.get(
        "goals_last_6m",
        np.nan
    )

    assists_6m = selected_player.get(
        "assists_last_6m",
        np.nan
    )

    minutes_6m = selected_player.get(
        "minutes_last_6m",
        np.nan
    )

    goals_per90 = selected_player.get(
        "goals_per_90_last_6m",
        np.nan
    )

    assists_per90 = selected_player.get(
        "assists_per_90_last_6m",
        np.nan
    )

    contribution_per90 = selected_player.get(
        "goal_contribution_per_90_last_6m",
        np.nan
    )


    perf_col1, perf_col2, perf_col3, perf_col4 = (
        st.columns(4)
    )

    with perf_col1:

        st.metric(
            "Games",
            (
                f"{games_6m:,.0f}"
                if pd.notna(games_6m)
                else "N/A"
            )
        )

    with perf_col2:

        st.metric(
            "Goals",
            (
                f"{goals_6m:,.0f}"
                if pd.notna(goals_6m)
                else "N/A"
            )
        )

    with perf_col3:

        st.metric(
            "Assists",
            (
                f"{assists_6m:,.0f}"
                if pd.notna(assists_6m)
                else "N/A"
            )
        )

    with perf_col4:

        st.metric(
            "Minutes Played",
            (
                f"{minutes_6m:,.0f}"
                if pd.notna(minutes_6m)
                else "N/A"
            )
        )


    rate_col1, rate_col2, rate_col3 = (
        st.columns(3)
    )

    with rate_col1:

        st.metric(
            "Goals / 90",
            (
                f"{goals_per90:.2f}"
                if pd.notna(goals_per90)
                else "N/A"
            )
        )

    with rate_col2:

        st.metric(
            "Assists / 90",
            (
                f"{assists_per90:.2f}"
                if pd.notna(assists_per90)
                else "N/A"
            )
        )

    with rate_col3:

        st.metric(
            "Goal Contributions / 90",
            (
                f"{contribution_per90:.2f}"
                if pd.notna(contribution_per90)
                else "N/A"
            )
        )

    st.caption(
        "These statistics provide additional context. "
        "The final model uses only Current Market Value, "
        "Age, Minutes Played and Goals."
    )


    st.divider()


    # ========================================================
    # 18. FINAL MODEL INPUTS
    # ========================================================

    st.subheader(
        "📊 Final Model Inputs"
    )

    st.write(
        "The final Gradient Boosting model uses exactly "
        "the following four features."
    )

    current_value = (
        selected_player[
            "current_market_value"
        ]
    )

    age = (
        selected_player[
            "age_at_snapshot"
        ]
    )

    minutes = (
        selected_player[
            "minutes_last_6m"
        ]
    )

    goals = (
        selected_player[
            "goals_last_6m"
        ]
    )


    input_col1, input_col2, input_col3, input_col4 = (
        st.columns(4)
    )

    with input_col1:

        st.metric(
            "Current Market Value",
            f"€{current_value / 1_000_000:.2f}M"
        )

    with input_col2:

        st.metric(
            "Age at Snapshot",
            f"{age:.1f}"
        )

    with input_col3:

        st.metric(
            "Minutes - Last 6 Months",
            f"{minutes:,.0f}"
        )

    with input_col4:

        st.metric(
            "Goals - Last 6 Months",
            f"{goals:,.0f}"
        )

    st.caption(
        "Only these four variables are passed to the final "
        "trained prediction pipeline."
    )


    st.divider()


    # ========================================================
    # 19. SIX-MONTH FORECAST
    # ========================================================

    st.subheader(
        "🔮 Six-Month Market Value Forecast"
    )

    st.write(
        "Generate an estimate of the player's market value "
        "approximately six months after the selected snapshot."
    )


    if st.button(
        "Predict Future Market Value",
        type="primary",
        use_container_width=True
    ):

        # ====================================================
        # 20. PREPARE MODEL INPUT
        # ====================================================

        player_features = pd.DataFrame(
            [
                {
                    "current_market_value":
                        current_value,

                    "age_at_snapshot":
                        age,

                    "minutes_last_6m":
                        minutes,

                    "goals_last_6m":
                        goals
                }
            ]
        )


        # ====================================================
        # 21. GENERATE PREDICTION
        # ====================================================

        predicted_future_value = (
            final_model.predict(
                player_features
            )[0]
        )

        predicted_change = (
            predicted_future_value
            - current_value
        )

        if current_value != 0:

            predicted_change_pct = (
                predicted_change
                / current_value
            ) * 100

        else:

            predicted_change_pct = 0.0


        forecast_date = (
            snapshot_date
            + pd.DateOffset(
                months=6
            )
        )


        # ====================================================
        # 22. PREDICTION RESULTS
        # ====================================================

        st.success(
            "Prediction completed successfully."
        )

        result_col1, result_col2, result_col3 = (
            st.columns(3)
        )

        with result_col1:

            st.metric(
                "Current Market Value",
                f"€{current_value / 1_000_000:.2f}M"
            )

        with result_col2:

            st.metric(
                "Predicted Value (~6 Months)",
                f"€{predicted_future_value / 1_000_000:.2f}M"
            )

        with result_col3:

            st.metric(
                "Predicted Change",
                f"€{predicted_change / 1_000_000:+.2f}M",
                delta=f"{predicted_change_pct:+.2f}%"
            )


        st.info(
            f"""
**Player:** {selected_player['name']}

**Latest Snapshot:** {snapshot_date.date()}

**Approximate Forecast Date:** {forecast_date.date()}

**Current Market Value:** €{current_value / 1_000_000:.2f}M

**Predicted Market Value:** €{predicted_future_value / 1_000_000:.2f}M

**Predicted Change:** €{predicted_change / 1_000_000:+.2f}M ({predicted_change_pct:+.2f}%)
            """
        )


        # ====================================================
        # 23. HISTORICAL + AI FORECAST
        # ====================================================

        st.subheader(
            "📈 Historical Value and AI Forecast"
        )


        forecast_chart = pd.DataFrame(
            columns=[
                "date",
                "Market Value (€)"
            ]
        )


        if not player_history.empty:

            forecast_chart = (
                player_history[
                    [
                        "date",
                        "market_value_in_eur"
                    ]
                ]
                .rename(
                    columns={
                        "market_value_in_eur":
                            "Market Value (€)"
                    }
                )
                .copy()
            )


        # Current prediction snapshot
        current_row = pd.DataFrame(
            {
                "date": [
                    snapshot_date
                ],

                "Market Value (€)": [
                    current_value
                ]
            }
        )


        # Future model prediction
        forecast_row = pd.DataFrame(
            {
                "date": [
                    forecast_date
                ],

                "Market Value (€)": [
                    predicted_future_value
                ]
            }
        )


        forecast_chart = pd.concat(
            [
                forecast_chart,
                current_row,
                forecast_row
            ],
            ignore_index=True
        )


        forecast_chart = (
            forecast_chart
            .drop_duplicates(
                subset="date",
                keep="last"
            )
            .sort_values(
                "date"
            )
            .set_index(
                "date"
            )
        )


        st.line_chart(
            forecast_chart,
            y="Market Value (€)",
            use_container_width=True
        )


        st.caption(
            "Historical observations are restricted to dates "
            "on or before the prediction snapshot. The final "
            "point is the AI-generated approximately six-month "
            "forecast and is not an observed valuation."
        )


        # ====================================================
        # 24. PREDICTION SUMMARY
        # ====================================================

        st.subheader(
            "📝 Prediction Summary"
        )


        if predicted_change > 0:

            st.write(
                f"The model predicts an **increase** in "
                f"market value for "
                f"**{selected_player['name']}**, from "
                f"approximately "
                f"**€{current_value / 1_000_000:.2f}M** "
                f"to "
                f"**€{predicted_future_value / 1_000_000:.2f}M** "
                f"over the approximately six-month "
                f"forecast horizon."
            )

        elif predicted_change < 0:

            st.write(
                f"The model predicts a **decrease** in "
                f"market value for "
                f"**{selected_player['name']}**, from "
                f"approximately "
                f"**€{current_value / 1_000_000:.2f}M** "
                f"to "
                f"**€{predicted_future_value / 1_000_000:.2f}M** "
                f"over the approximately six-month "
                f"forecast horizon."
            )

        else:

            st.write(
                f"The model predicts that the market value "
                f"of **{selected_player['name']}** will remain "
                f"approximately stable over the six-month "
                f"forecast horizon."
            )


        # ====================================================
        # 25. LOCAL SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.subheader(
            "🧠 AI Prediction Explanation"
        )

        st.write(
            "SHAP is used to explain how each of the four "
            "model inputs contributes to this specific "
            "player prediction relative to the model's "
            "baseline prediction."
        )


        # ----------------------------------------------------
        # Transform exactly as the pipeline does
        # ----------------------------------------------------

        transformed_features = (
            preprocessor.transform(
                player_features
            )
        )


        # ----------------------------------------------------
        # Calculate local SHAP values
        # ----------------------------------------------------

        local_shap_values = (
            shap_explainer.shap_values(
                transformed_features
            )
        )

        local_shap_values = (
            np.asarray(
                local_shap_values
            )[0]
        )


        base_value = (
            np.asarray(
                shap_explainer.expected_value
            )
            .reshape(-1)[0]
        )


        # ----------------------------------------------------
        # Verify SHAP decomposition
        # ----------------------------------------------------

        shap_prediction = (
            base_value
            + local_shap_values.sum()
        )

        shap_matches = np.isclose(
            shap_prediction,
            predicted_future_value,
            atol=0.01
        )


        # ----------------------------------------------------
        # Build explanation table
        # ----------------------------------------------------

        explanation_df = pd.DataFrame(
            {
                "Feature": [
                    FEATURE_LABELS[
                        feature
                    ]
                    for feature
                    in MODEL_FEATURES
                ],

                "Player Value": [
                    current_value,
                    age,
                    minutes,
                    goals
                ],

                "SHAP Contribution (€)":
                    local_shap_values
            }
        )


        explanation_df[
            "Absolute Contribution (€)"
        ] = (
            explanation_df[
                "SHAP Contribution (€)"
            ]
            .abs()
        )


        explanation_df = (
            explanation_df
            .sort_values(
                "Absolute Contribution (€)",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        # ----------------------------------------------------
        # XAI summary metrics
        # ----------------------------------------------------

        xai_col1, xai_col2, xai_col3 = (
            st.columns(3)
        )


        with xai_col1:

            st.metric(
                "SHAP Baseline Value",
                f"€{base_value / 1_000_000:.2f}M"
            )


        with xai_col2:

            st.metric(
                "Total Feature Contribution",
                (
                    f"€"
                    f"{local_shap_values.sum() / 1_000_000:+.2f}M"
                )
            )


        with xai_col3:

            st.metric(
                "Final Prediction",
                (
                    f"€"
                    f"{predicted_future_value / 1_000_000:.2f}M"
                )
            )


        # ----------------------------------------------------
        # Human-readable contribution cards
        # ----------------------------------------------------

        st.markdown(
            "#### Feature Contributions"
        )


        for _, row in explanation_df.iterrows():

            feature_name = row[
                "Feature"
            ]

            contribution = row[
                "SHAP Contribution (€)"
            ]


            if contribution > 0:

                direction = (
                    "Positive contribution"
                )

                symbol = "↑"

            elif contribution < 0:

                direction = (
                    "Negative contribution"
                )

                symbol = "↓"

            else:

                direction = (
                    "Neutral contribution"
                )

                symbol = "→"


            st.write(
                f"**{feature_name}** — "
                f"{symbol} "
                f"€{contribution / 1_000_000:+.2f}M "
                f"({direction})"
            )


        # ----------------------------------------------------
        # Contribution table
        # ----------------------------------------------------

        display_explanation = (
            explanation_df[
                [
                    "Feature",
                    "SHAP Contribution (€)"
                ]
            ]
            .copy()
        )


        display_explanation[
            "Contribution (€M)"
        ] = (
            display_explanation[
                "SHAP Contribution (€)"
            ]
            / 1_000_000
        )


        display_explanation = (
            display_explanation[
                [
                    "Feature",
                    "Contribution (€M)"
                ]
            ]
        )


        st.dataframe(
            display_explanation,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Contribution (€M)":
                    st.column_config.NumberColumn(
                        "SHAP Contribution (€M)",
                        format="%+.2f"
                    )
            }
        )


        # ----------------------------------------------------
        # Mathematical decomposition
        # ----------------------------------------------------

        st.markdown(
            "#### Prediction Decomposition"
        )


        st.write(
            f"**Baseline:** "
            f"€{base_value / 1_000_000:.2f}M"
        )


        for _, row in explanation_df.iterrows():

            st.write(
                f"{row['Feature']}: "
                f"€{row['SHAP Contribution (€)'] / 1_000_000:+.2f}M"
            )


        st.write(
            f"**Final prediction:** "
            f"€{predicted_future_value / 1_000_000:.2f}M"
        )


        # ----------------------------------------------------
        # Technical validation
        # ----------------------------------------------------

        if shap_matches:

            st.success(
                "SHAP decomposition successfully reconstructs "
                "the model prediction."
            )

        else:

            st.warning(
                "The SHAP decomposition does not exactly "
                "reconstruct the model prediction."
            )


        # ----------------------------------------------------
        # Identify dominant local contribution
        # ----------------------------------------------------

        dominant_feature = (
            explanation_df.iloc[0][
                "Feature"
            ]
        )

        dominant_contribution = (
            explanation_df.iloc[0][
                "SHAP Contribution (€)"
            ]
        )


        st.info(
            f"For this prediction, "
            f"**{dominant_feature}** has the largest absolute "
            f"SHAP contribution "
            f"({dominant_contribution / 1_000_000:+.2f}M)."
        )


        st.caption(
            "SHAP values explain the model's prediction "
            "relative to its baseline. Positive values push "
            "the prediction above the baseline and negative "
            "values push it below the baseline. These values "
            "describe model behaviour and should not be "
            "interpreted as causal effects."
        )


# ============================================================
# 26. NO PLAYER SELECTED
# ============================================================

else:

    st.info(
        "Search for and select a player to view the player's "
        "profile, historical market value, recent performance, "
        "model inputs and six-month prediction."
    )


# ============================================================
# 27. FOOTER
# ============================================================

st.divider()

st.caption(
    "Final Year Project — AI-Based Football "
    "Transfer Market Value Prediction"
)