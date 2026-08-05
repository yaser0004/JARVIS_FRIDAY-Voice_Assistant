"""Router keyword recovery: the ladder that runs when the classifier is unsure.

These cases pin the keyword collisions that used to misroute questions into
system actions ("how does sound travel" -> volume, "how much sleep do I need"
-> suspend the machine).
"""

import pytest

from nlp.router import Router


@pytest.fixture(scope="module")
def router():
    return Router(llm=None)


# Utterances that must NOT be treated as device actions.
QUESTIONS = [
    "how does sound travel through water",
    "how much sleep do i need every night",
    "what is the best audio format for archiving",
    "who invented the pdf document format",
]

# Utterances that must still reach their action intent.
COMMANDS = [
    ("turn the volume down", "system_volume"),
    ("set volume to 40%", "system_volume"),
    ("mute the speakers", "system_volume"),
    ("put the pc to sleep", "power_control"),
    ("go to sleep", "power_control"),
    ("shutdown the computer", "power_control"),
    ("open my resume pdf", "file_control"),
    ("find the budget document", "file_control"),
    ("increase brightness", "system_brightness"),
    ("close chrome", "close_app"),
    ("launch spotify", "launch_app"),
    ("switch to firefox", "switch_app"),
    ("cancel that", "stop_cancel"),
    ("what is the weather in tokyo", "weather_query"),
]


@pytest.mark.parametrize("text", QUESTIONS)
def test_questions_do_not_become_device_actions(router, text):
    intent = router._recover_intent_from_text(text, {}, "general_qa", 0.0)
    assert intent not in {"system_volume", "power_control", "file_control", "system_brightness"}


@pytest.mark.parametrize("text,expected", COMMANDS)
def test_commands_reach_their_action(router, text, expected):
    intent = router._recover_intent_from_text(text, {}, "general_qa", 0.0)
    assert intent == expected


def test_confident_classifier_intent_is_respected(router):
    assert router._recover_intent_from_text("play some jazz", {}, "play_media", 0.9) == "play_media"


def test_confident_general_qa_skips_keyword_ladder(router):
    # A genuinely confident general_qa must not be dragged into an action by a
    # stray keyword; below the threshold the ladder still gets its say.
    assert router._recover_intent_from_text("tell me about audio compression", {}, "general_qa", 0.95) == "general_qa"


def test_weather_misclassification_is_corrected(router):
    # Classifier says weather but nothing in the text supports it.
    assert router._recover_intent_from_text("search for python tutorials", {}, "weather_query", 0.9) == "web_search"
