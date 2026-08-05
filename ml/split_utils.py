from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_intent_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    columns = [c for c in ("text", "intent", "base_text") if c in df.columns]
    df = df[columns]
    df["text"] = df["text"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]
    df = df.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No usable intent rows found in {data_path}")
    return df


def split_intent_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_attempts: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Group by the augmentation source sentence when available so trivial
    # variants of one utterance can never straddle the split; exact-text
    # grouping is only a weak fallback for datasets without base_text.
    if "base_text" in df.columns:
        groups = df["base_text"].astype(str).str.lower()
    else:
        groups = df["text"].str.lower()
    all_intents = set(df["intent"].tolist())

    for offset in range(max_attempts):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + offset,
        )
        train_idx, test_idx = next(splitter.split(df["text"], df["intent"], groups=groups))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        if set(train_df["intent"].tolist()) == all_intents and set(test_df["intent"].tolist()) == all_intents:
            return train_df, test_df

    raise RuntimeError(
        "Unable to create a group-aware train/test split that contains all intents in both sets. "
        "Increase dataset diversity or reduce test_size."
    )
