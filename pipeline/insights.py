import os
import pandas as pd
from openai import OpenAI

def add_week_bucket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["week"] = out["date"].dt.to_period("W").astype(str)
    return out

def theme_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes theme-level metrics.
    """
    # Define "negative" using rating (simple MVP rule)
    out = df.copy()
    out["is_negative"] = out["rating"].apply(lambda r: 1 if r <= 2 else 0)

    agg = (
        out.groupby("theme", as_index=False)
        .agg(
            feedback_count=("id", "count"),
            avg_rating=("rating", "mean"),
            negative_rate=("is_negative", "mean"),
        )
    )
    # score: prioritize high volume + high negative rate + low rating
    agg["priority_score"] = (
        agg["feedback_count"] * (1 + agg["negative_rate"]) * (6 - agg["avg_rating"])
    )
    return agg.sort_values("priority_score", ascending=False)

def exec_summary_and_actions(df: pd.DataFrame, top_n: int = 5, model: str = "gpt-4o-mini") -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    metrics = theme_metrics(df).head(top_n)
    bullets = []
    for _, row in metrics.iterrows():
        bullets.append(
            f"- Theme: {row['theme']} | count={int(row['feedback_count'])} | "
            f"avg_rating={row['avg_rating']:.2f} | negative_rate={row['negative_rate']:.0%}"
        )

    # evidence: a few representative quotes per top theme
    evidence_lines = []
    for theme in metrics["theme"].tolist():
        samples = df[df["theme"] == theme]["text"].head(3).tolist()
        evidence_lines.append(f"\nTheme: {theme}\n" + "\n".join([f"  • {s}" for s in samples]))

    prompt = (
        "Create an executive-ready weekly summary of customer feedback.\n"
        "Include:\n"
        "1) Key insights (3–5 bullets)\n"
        "2) Top risks/issues (2–4 bullets)\n"
        "3) Recommended actions (3–6 bullets) with clear owners (Product, Eng, Support)\n"
        "Keep it concise and business-friendly.\n\n"
        "Theme metrics:\n" + "\n".join(bullets) + "\n\n"
        "Evidence:\n" + "\n".join(evidence_lines)
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write concise executive summaries for product teams."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
