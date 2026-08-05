from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "intents_train.csv"
TEST_PATH = DATA_DIR / "intents_test.csv"
DEFAULT_TEST_FRACTION = 0.2

INTENT_MIN_COUNTS = {
    "launch_app": 180,
    "close_app": 120,
    "switch_app": 120,
    "web_search": 180,
    "weather_query": 160,
    "open_website": 120,
    "play_media": 150,
    "system_volume": 100,
    "system_brightness": 80,
    "power_control": 100,
    "system_settings": 80,
    "general_qa": 320,
    "vision_query": 100,
    "file_control": 100,
    "clipboard_action": 60,
    "stop_cancel": 80,
}

TEMPLATES = {
    "launch_app": [
        "open chrome",
        "launch spotify",
        "start vscode",
        "run notepad",
        "open calculator",
    ],
    "close_app": [
        "close chrome",
        "quit spotify",
        "exit vscode",
        "close notepad",
        "terminate discord",
    ],
    "switch_app": [
        "switch to chrome",
        "switch to vscode",
        "focus on spotify",
        "bring discord to front",
        "switch to file explorer",
    ],
    "web_search": [
        "search for machine learning roadmap",
        "google weather in delhi",
        "find best gpu under 30000",
        "search youtube for python tutorial",
        "look up latest ai news",
        "show me tech news",
        "give me a web overview of ai safety",
        "get me an online overview of quantum computing",
        "what are people saying online about electric cars",
    ],
    "weather_query": [
        "weather in london",
        "forecast for dubai",
        "temperature in tokyo",
        "what is the weather in new york",
        "weather in berlin in fahrenheit",
    ],
    "open_website": [
        "open github.com",
        "go to youtube",
        "open wikipedia",
        "visit reddit",
        "open stackoverflow",
    ],
    "play_media": [
        "play blinding lights on spotify",
        "play lofi mix on youtube",
        "play shape of you",
        "start my local playlist",
        "play relaxing piano music",
        "pause music playback",
        "resume the song",
        "next track",
        "previous track",
    ],
    "system_volume": [
        "set volume to 50 percent",
        "increase volume",
        "decrease volume",
        "mute volume",
        "set volume to max",
        "what is the current volume",
        "set volume to 30",
        "increase volume by 5",
    ],
    "system_brightness": [
        "set brightness to 40 percent",
        "increase brightness",
        "decrease brightness",
        "brightness max",
        "brightness half",
        "what is current brightness",
        "set brightness to 60",
        "decrease brightness by 10",
    ],
    "power_control": [
        "shutdown system",
        "restart computer",
        "put pc to sleep",
        "hibernate now",
        "lock my computer",
        "turn off monitor",
        "enable wifi",
        "disable wifi",
        "turn bluetooth on",
        "turn bluetooth off",
        "turn on airplane mode",
        "turn off airplane mode",
        "enable battery saver",
        "disable battery saver",
    ],
    "system_settings": [
        "open wifi settings",
        "open bluetooth settings",
        "show display settings",
        "open sound settings",
        "open privacy settings",
        "open airplane mode settings",
        "open battery saver settings",
    ],
    "general_qa": [
        "what is reinforcement learning",
        "how does photosynthesis work",
        "why is the sky blue",
        "what causes earthquakes",
        "how much is 15 percent of 200",
        "what is the difference between ram and storage",
        "why does my laptop fan run loudly",
        "what is two factor authentication",
        "who wrote pride and prejudice",
        "what is the capital of japan",
        "how many continents are there",
        "what is the speed of light",
        "why do we need sleep",
        "what does airplane mode do",
        "how does gps work",
        "what is cloud computing",
        "tell me about black holes",
        "tell me about tcp and udp",
        "tell me about insulin resistance",
        "tell me about convolutional neural networks",
    ],
    "vision_query": [
        "what is on my screen",
        "describe this image",
        "what do you see",
        "analyze the webcam view",
        "read the text on screen",
        "analyze image from file",
        "analyze camera snapshot",
    ],
    "file_control": [
        "open my project report",
        "find assignment pdf",
        "read notes file",
        "open downloads folder",
        "search for resume docx",
    ],
    "clipboard_action": [
        "read clipboard",
        "copy this text",
        "paste from clipboard",
        "what is in clipboard",
        "save this to clipboard",
    ],
    "stop_cancel": [
        "stop",
        "cancel that",
        "never mind",
        "be quiet",
        "forget it",
    ],
}

MANUAL_PARAPHRASES = {
    "launch_app": ["bring up chrome", "fire up spotify", "boot vscode", "open file explorer"],
    "close_app": ["shut chrome", "kill spotify", "close this app", "stop notepad"],
    "switch_app": ["bring chrome to front", "focus on vscode", "jump to discord", "switch over to spotify"],
    "web_search": [
        "search reddit for ai",
        "find internships in ai",
        "google pytorch docs",
        "look up cuda setup",
        "show me latest startup news",
        "give me a web overview of electric vehicles",
        "what does the web say about intermittent fasting",
    ],
    "weather_query": [
        "tell me weather in paris",
        "show weather for mumbai",
        "forecast in singapore",
        "temperature in cairo",
    ],
    "open_website": ["open github", "visit huggingface", "go to kaggle", "open wikipedia page"],
    "play_media": [
        "play chill beats",
        "play top hits",
        "start some music",
        "play coding soundtrack",
        "pause the current song",
        "skip this track",
    ],
    "system_volume": [
        "volume up",
        "volume down",
        "set sound to zero",
        "unmute speakers",
        "what volume are we on",
    ],
    "system_brightness": [
        "screen brighter",
        "dim the screen",
        "set display to 70 percent",
        "screen brightness low",
        "how bright is the screen",
    ],
    "power_control": [
        "turn off pc",
        "reboot machine",
        "put computer to sleep",
        "lock the workstation",
        "switch monitor off",
        "wifi off now",
        "wifi on now",
        "enable battery saver mode",
    ],
    "system_settings": [
        "open update settings",
        "take me to app settings",
        "open battery settings",
        "open display panel",
        "open airplane settings",
    ],
    "general_qa": [
        "explain gradient descent",
        "what is f1 score",
        "define precision recall",
        "what is overfitting",
        "how do i improve focus while studying",
        "what are signs of cpu overheating",
        "why is my internet unstable",
        "what is the purpose of a firewall",
        "how does bluetooth pairing work",
        "what is a secure password",
        "is seven a prime number",
        "what is the distance between earth and moon",
        "tell me about photosynthesis",
        "tell me about cloud computing",
        "tell me about the human immune system",
        "tell me about data structures",
    ],
    "vision_query": [
        "scan my display",
        "describe what is visible",
        "read the current screen",
        "analyze this picture",
        "inspect this image file",
        "recognize objects from camera",
    ],
    "file_control": ["open my documents", "find the budget sheet", "locate project file", "read the markdown file"],
    "clipboard_action": ["show clipboard text", "copy this sentence", "paste clipboard content", "save to clipboard"],
    "stop_cancel": ["abort", "drop it", "stop listening", "cancel operation"],
}

INTENT_EDGE_CASES = {
    "launch_app": [
        "open youtube app",
        "launch settings app",
        "start spotify desktop app",
        "open camera app on windows",
        "run edge browser app",
    ],
    "web_search": [
        "search web for chrome release notes",
        "find spotify premium price online",
        "look up youtube algorithm updates",
        "google vscode python extensions",
        "search internet for battery saver tips",
        "give me a web overview of nvidia earnings",
        "summarize online coverage about climate tech",
        "show me the latest news on fusion energy",
        "what are current web updates on ai regulation",
        "get a web overview for india gdp outlook",
    ],
    "weather_query": [
        "what is the weather in london right now",
        "is it raining in seattle",
        "forecast for paris in celsius",
        "temperature at dubai in fahrenheit",
        "weather in san francisco today",
    ],
    "open_website": [
        "open youtube dot com",
        "go to spotify website",
        "visit microsoft support page",
        "open wikipedia website",
        "open stackoverflow website",
    ],
    "play_media": [
        "play lofi on youtube",
        "play my spotify discover weekly",
        "start youtube music playlist",
        "play the song believer",
        "pause spotify playback",
    ],
    "system_settings": [
        "open windows settings",
        "open update settings panel",
        "show network settings page",
        "open sound settings window",
        "open bluetooth settings page",
    ],
    "file_control": [
        "search this pc for resume",
        "find report file in downloads",
        "open local budget spreadsheet",
        "look for assignment in documents",
        "open the latest local invoice file",
    ],
    "general_qa": [
        "is chrome better than edge for privacy",
        "which is better wifi or ethernet",
        "what is the difference between cpu and gpu",
        "how much ram is enough for programming",
        "what does low disk space affect",
        "tell me about world war one causes",
        "tell me about linux kernel",
        "tell me about quantum entanglement",
        "tell me about kidney stone symptoms",
        "tell me about object oriented programming",
    ],
    "switch_app": [
        "switch to the currently open chrome window",
        "bring my spotify window to front",
        "focus on vscode app",
        "switch over to file explorer",
        "jump to discord window",
    ],
    "stop_cancel": [
        "cancel the app launch",
        "stop that search now",
        "never mind dont play anything",
        "drop this operation",
        "abort current task",
    ],
}

STT_ERROR_REPLACEMENTS = {
    "spotify": ["spotifai", "spotifiy", "spotyfy"],
    "chrome": ["krome", "crome"],
    "youtube": ["you tube", "utube"],
    "vscode": ["vs code", "v s code"],
    "notepad": ["note pad"],
    "calculator": ["calc ulator", "calculater"],
    "bluetooth": ["blue tooth", "bluetoth"],
    "wifi": ["wi fi", "why fi"],
    "clipboard": ["clip board", "clibboard"],
    "brightness": ["briteness", "bright ness"],
    "volume": ["volyum", "valume"],
    "cancel": ["cansel", "kan sel"],
    "launch": ["lanch", "lonch"],
    "close": ["cloze", "closs"],
    "switch": ["swich", "switsh"],
    "search": ["serch", "surch"],
    "weather": ["wether", "wedher"],
    "forecast": ["forcast", "four cast"],
    "temperature": ["temprature", "temp rature"],
    "website": ["web site", "webside"],
}

STT_ERROR_EXAMPLES = {
    "launch_app": [
        "open krome",
        "lanch spotifai",
        "start v s code",
        "open note pad",
        "run calculater",
    ],
    "close_app": [
        "cansel chrome and cloze it",
        "close spotifiy",
        "quit v s code app",
        "close note pad now",
    ],
    "switch_app": [
        "swich to krome",
        "focus on v s code",
        "bring spotifai to front",
        "switch to discorde window",
    ],
    "web_search": [
        "serch web for gpu prices",
        "google why fi speed test",
        "look up utube python tutorial",
        "find webside for weather updates",
    ],
    "weather_query": [
        "wether in delhi",
        "forcast for london",
        "temprature in dubai",
        "what is wedher in tokyo",
    ],
    "open_website": [
        "open you tube dot com",
        "visit github webside",
        "open stack over flow",
    ],
    "play_media": [
        "play spotifai songs",
        "play you tube music",
        "next trak",
        "pause the song plz",
    ],
    "system_volume": [
        "increase valume",
        "mute volyum",
        "set valume to fifty",
    ],
    "system_brightness": [
        "increase briteness",
        "lower bright ness",
        "set briteness to sixty",
    ],
    "system_settings": [
        "open blue tooth settings",
        "show why fi settings",
        "open display setting page",
    ],
    "power_control": [
        "shut down the computor",
        "restar the pc",
        "turn off moniter",
        "enable air plane mode",
    ],
    "vision_query": [
        "what is on my skreen",
        "analyze this imaj",
        "read text from scren",
        "camera snap shot analysis",
    ],
    "file_control": [
        "serch this pc for file",
        "open down loads folder",
        "find resumay doc",
        "open lokal project report",
    ],
    "clipboard_action": [
        "show clip board",
        "copy this to clibboard",
        "paste from clip board",
    ],
    "stop_cancel": [
        "cansel that",
        "kan sel current task",
        "stop it now plz",
    ],
}

POLITE_PREFIXES = [
    "can you ",
    "could you ",
    "would you ",
    "please ",
    "do me a favor and ",
    "hey assistant ",
]

POLITE_SUFFIXES = [
    " please",
    " for me",
    " if you can",
    " right now",
]


class DatasetBuilder:
    def __init__(self, seed: int = 42) -> None:
        random.seed(seed)
        self.syn_aug = None
        self.del_aug = None
        self.typo_aug = None
        self._init_augmenters()

    def _base_pools(self) -> Dict[str, List[str]]:
        pools: Dict[str, List[str]] = {}
        for intent in INTENT_MIN_COUNTS:
            bases = list(
                dict.fromkeys(
                    TEMPLATES.get(intent, [])
                    + MANUAL_PARAPHRASES.get(intent, [])
                    + INTENT_EDGE_CASES.get(intent, [])
                    + STT_ERROR_EXAMPLES.get(intent, [])
                )
            )
            pools[intent] = [str(b).strip() for b in bases if str(b).strip()]
        return pools

    def split_bases(
        self, test_fraction: float = DEFAULT_TEST_FRACTION
    ) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Hold out whole base sentences per intent BEFORE any augmentation.

        Splitting augmented rows leaks trivial variants of the same sentence
        across train/test, which is how this project once reported 100%
        accuracy. Base-level holdout is the honest unit of generalization.
        """
        train_pools: Dict[str, List[str]] = {}
        test_pools: Dict[str, List[str]] = {}
        for intent, bases in self._base_pools().items():
            if len(bases) < 5:
                raise ValueError(f"intent '{intent}' needs at least 5 base sentences, has {len(bases)}")
            shuffled = bases[:]
            random.shuffle(shuffled)
            n_test = max(2, round(len(shuffled) * test_fraction))
            n_test = min(n_test, len(shuffled) - 3)
            test_pools[intent] = shuffled[:n_test]
            train_pools[intent] = shuffled[n_test:]
        return train_pools, test_pools

    def build_datasets(
        self, test_fraction: float = DEFAULT_TEST_FRACTION
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build and write intents_train.csv (augmented) and intents_test.csv (raw holdout)."""
        train_pools, test_pools = self.split_bases(test_fraction)

        rows: List[Dict[str, str]] = []
        for intent, bases in train_pools.items():
            for base in bases:
                rows.append({"text": base, "intent": intent, "base_text": base})
                variants = (
                    self._augment_text(base)
                    + self._stt_noise_variants(base)
                    + self._rule_based_variants(base)
                )
                for variant in variants:
                    value = str(variant).strip()
                    if value and value != base:
                        rows.append({"text": value, "intent": intent, "base_text": base})

        train_df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["text", "intent"])
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )

        test_df = pd.DataFrame(
            [
                {"text": base, "intent": intent, "base_text": base}
                for intent, bases in test_pools.items()
                for base in bases
            ]
        ).reset_index(drop=True)

        base_overlap = set(train_df["base_text"].str.lower()) & set(test_df["base_text"].str.lower())
        if base_overlap:
            raise RuntimeError(f"train/test base overlap: {sorted(base_overlap)[:5]}")

        # An augmented train variant can coincidentally reproduce a held-out base
        # (e.g. "open chrome" -> "launch chrome"); drop those train rows.
        test_texts = set(test_df["text"].str.lower())
        train_df = train_df[~train_df["text"].str.lower().isin(test_texts)].reset_index(drop=True)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)
        return train_df, test_df

    def _augment_text(self, text: str) -> List[str]:
        out = {text}
        for aug in (self.syn_aug, self.del_aug, self.typo_aug):
            if aug is None:
                continue
            try:
                augmented = aug.augment(text)
                if isinstance(augmented, list):
                    out.update([x for x in augmented if isinstance(x, str) and x.strip()])
                elif isinstance(augmented, str) and augmented.strip():
                    out.add(augmented)
            except Exception:
                continue

        if len(out) <= 1:
            out.update(self._rule_based_variants(text))
        return list(out)

    def _init_augmenters(self) -> None:
        try:
            import nlpaug.augmenter.char as nac
            import nlpaug.augmenter.word as naw

            self.syn_aug = naw.SynonymAug(aug_src="wordnet")
            self.del_aug = naw.RandomWordAug(action="delete")
            self.typo_aug = nac.KeyboardAug()
        except Exception:
            self.syn_aug = None
            self.del_aug = None
            self.typo_aug = None

    @staticmethod
    def _rule_based_variants(text: str) -> List[str]:
        value = str(text or "").strip()
        if not value:
            return []

        variants = {
            value,
            value.lower(),
            re.sub(r"\s+", " ", value).strip(),
        }

        polite_suffixes = [" please", " now", " quickly", " right away", " for me"]
        for suffix in polite_suffixes:
            variants.add(f"{value}{suffix}")

        for prefix in POLITE_PREFIXES:
            variants.add(f"{prefix}{value}")
            for suffix in POLITE_SUFFIXES:
                variants.add(f"{prefix}{value}{suffix}")

        replacements = {
            "open": "launch",
            "launch": "open",
            "start": "run",
            "close": "quit",
            "increase": "raise",
            "decrease": "lower",
            "turn on": "enable",
            "turn off": "disable",
        }
        lower = value.lower()
        for src, dst in replacements.items():
            if src in lower:
                variants.add(lower.replace(src, dst))

        return [item for item in variants if isinstance(item, str) and item.strip()]

    @staticmethod
    def _stt_noise_variants(text: str) -> List[str]:
        value = str(text or "").strip().lower()
        if not value:
            return []

        variants = set()
        for src, alternatives in STT_ERROR_REPLACEMENTS.items():
            if not alternatives:
                continue
            pattern = rf"\b{re.escape(src)}\b"
            if re.search(pattern, value):
                for alt in alternatives[:2]:
                    variants.add(re.sub(pattern, alt, value))

        compact = re.sub(r"\s+", " ", value).strip()
        compact = compact.replace(" please", "").replace(" right now", "")
        if compact and compact != value:
            variants.add(compact)

        return [item for item in variants if item and item != value]


if __name__ == "__main__":
    builder = DatasetBuilder(seed=42)
    train_df, test_df = builder.build_datasets()
    print(f"Train dataset size: {len(train_df)} ({train_df['base_text'].nunique()} base sentences)")
    print(f"Test dataset size: {len(test_df)} raw held-out base sentences")
    print(f"Train CSV path: {TRAIN_PATH}")
    print(f"Test CSV path: {TEST_PATH}")
    print("Train class distribution:")
    print(train_df["intent"].value_counts())
    print("Test class distribution:")
    print(test_df["intent"].value_counts())
