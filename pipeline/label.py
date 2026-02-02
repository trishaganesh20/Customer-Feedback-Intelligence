import os
import pandas as pd
from openai import OpenAI

def label_clusters_with_llm(df: pd.DataFrame, model: str = "gpt-4o-mini") -> dict[int, str]:
    """
    For each cluster_id, take representative examples and ask LLM for a short theme label.
    Returns {cluster_id: theme_name}
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mapping: dict[int, str] = {}

    for cid, group in df.groupby("cluster_id"):
        examples = group["text"].head(10).tolist()

        prompt = (
            "You are labeling customer feedback themes.\n"
            "Given these user comments, return a SHORT theme label (2–4 words). "
            "No quotes, no extra text.\n\n"
            "Comments:\n" + "\n".join([f"- {e}" for e in examples])
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return only the theme label."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        theme = resp.choices[0].message.content.strip()
        # safety: keep label short
        theme = " ".join(theme.split()[:5])
        mapping[int(cid)] = theme

    return mapping
