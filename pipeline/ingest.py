import pandas as pd
from datetime import datetime

def normalize_any_csv(
    df: pd.DataFrame,
    text_col: str,
    date_col: str | None = None,
    rating_col: str | None = None,
    source_col: str | None = None,
) -> pd.DataFrame:
    """
    Convert ANY CSV into the app's standard schema:
    id, date, source, rating, text
    """
    out = pd.DataFrame()

    # REQUIRED: feedback text
    out["text"] = df[text_col].astype(str)

    # ALWAYS create an id
    out["id"] = range(1, len(df) + 1)

    # date: optional
    if date_col and date_col in df.columns:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        out["date"] = pd.Timestamp(datetime.today().date())

    # source: optional
    if source_col and source_col in df.columns:
        out["source"] = df[source_col].astype(str).fillna("unknown")
    else:
        out["source"] = "unknown"

    # rating: optional (tries to extract a number if text like "5 stars")
    if rating_col and rating_col in df.columns:
        raw = df[rating_col].astype(str)
        out["rating"] = pd.to_numeric(raw.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
    else:
        out["rating"] = pd.NA

    return out
