"""The train/test split must hold out base sentences BEFORE augmentation.

Splitting augmented rows scattered trivial variants of the same sentence
across both sides, which is what produced the old ~100% "accuracy".
"""

import pandas as pd
import pytest

from ml.dataset import INTENT_MIN_COUNTS, DatasetBuilder


@pytest.fixture(scope="module")
def split():
    builder = DatasetBuilder(seed=42)
    return builder.split_bases(test_fraction=0.2)


def test_no_base_sentence_appears_in_both_sides(split):
    train_pools, test_pools = split
    for intent in train_pools:
        overlap = {t.lower() for t in train_pools[intent]} & {t.lower() for t in test_pools[intent]}
        assert not overlap, f"{intent} leaks base sentences: {sorted(overlap)[:3]}"


def test_every_intent_present_on_both_sides(split):
    train_pools, test_pools = split
    for intent in INTENT_MIN_COUNTS:
        assert train_pools.get(intent), f"{intent} missing from train"
        assert test_pools.get(intent), f"{intent} missing from test"


def test_built_csvs_share_no_text_or_base(tmp_path, monkeypatch):
    import ml.dataset as dataset

    monkeypatch.setattr(dataset, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dataset, "TRAIN_PATH", tmp_path / "intents_train.csv")
    monkeypatch.setattr(dataset, "TEST_PATH", tmp_path / "intents_test.csv")

    train_df, test_df = dataset.DatasetBuilder(seed=7).build_datasets(test_fraction=0.2)

    assert not set(train_df["base_text"].str.lower()) & set(test_df["base_text"].str.lower())
    assert not set(train_df["text"].str.lower()) & set(test_df["text"].str.lower())
    assert (tmp_path / "intents_train.csv").exists()
    assert (tmp_path / "intents_test.csv").exists()


def test_test_set_is_unaugmented(tmp_path, monkeypatch):
    import ml.dataset as dataset

    monkeypatch.setattr(dataset, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dataset, "TRAIN_PATH", tmp_path / "train.csv")
    monkeypatch.setattr(dataset, "TEST_PATH", tmp_path / "test.csv")

    _, test_df = dataset.DatasetBuilder(seed=7).build_datasets(test_fraction=0.2)
    # Every test row is its own base sentence: no generated variants.
    assert (test_df["text"] == test_df["base_text"]).all()


def test_grouped_split_keeps_variants_together():
    from ml.split_utils import split_intent_dataframe

    rows = []
    for intent in ("launch_app", "close_app"):
        for base_index in range(10):
            base = f"{intent} base {base_index}"
            for variant_index in range(3):
                rows.append({"text": f"{base} v{variant_index}", "intent": intent, "base_text": base})
    df = pd.DataFrame(rows)

    train_df, test_df = split_intent_dataframe(df, test_size=0.3, random_state=1)
    assert not set(train_df["base_text"]) & set(test_df["base_text"])
