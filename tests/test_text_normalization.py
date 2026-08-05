"""The pipeline must normalize conversational phrasing BEFORE stripping punctuation.

Cleaning first destroyed the apostrophes that conversation_normalizer's
patterns depend on, so polite wrappers survived into the classifier input.
"""

from nlp import preprocessor
from nlp.conversation_normalizer import normalize_command_text


def prepare(text: str) -> str:
    """Mirror core.pipeline.process_text's preprocessing order."""
    normalized = normalize_command_text(text) or str(text or "").strip().lower()
    return preprocessor.clean(normalized)


def test_apostrophe_pattern_survives_normalization():
    assert prepare("i'd like you to open chrome") == "open chrome"


def test_polite_wrapper_stripped():
    assert prepare("Could you please open Spotify for me?") == "open spotify"


def test_assistant_prefix_and_filler_stripped():
    assert prepare("hey Jarvis, um, launch firefox") == "launch firefox"


def test_wrong_order_would_regress():
    # Cleaning first turns "i'd" into "i d", so the normalizer's pattern misses
    # and the polite wrapper leaks through. This is the bug the order fix cures.
    clean_first = normalize_command_text(preprocessor.clean("i'd like you to open chrome"))
    assert clean_first != "open chrome"


def test_clean_output_is_lowercase_without_punctuation():
    out = prepare("Open Chrome!!!")
    assert out == "open chrome"
