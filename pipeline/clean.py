import pandas as pd

def clean_feedback(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Only ensure required columns exist
    out["text"] = out["text"].astype(str)

    # Drop rows ONLY if text is empty
    out = out[out["text"].str.strip().ne("")].copy()

    # Date handling (keep default dates if auto-generated)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Rating is OPTIONAL
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")

    # Source fallback
    out["source"] = out["source"].fillna("unknown").astype(str)

    return out
