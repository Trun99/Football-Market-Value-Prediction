# ============================================================
# UI CORE PACKAGE
# ============================================================
#
# Support code for the LSTM + Gradient Boosting deployment
# dashboard (app_lstm.py).
#
# Contains only dependency-free helpers that were extracted
# verbatim from app_lstm.py:
#
#   theme.py       -> THEME_CSS, render_html, render_section
#   formatting.py  -> to_float, format_eur, format_integer,
#                     format_decimal, safe_text, format_date
#
# No module in this package imports app_lstm.py, and no model,
# data, prediction or Streamlit layout logic lives here.
# ============================================================
