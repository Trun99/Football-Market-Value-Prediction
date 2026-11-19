# ============================================================
# VALUE FORMATTING HELPERS
# ============================================================
#
# Extracted verbatim from app_lstm.py (sections 7-12).
# Function bodies, names and behaviour are unchanged.
# ============================================================

import numpy as np
import pandas as pd


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
