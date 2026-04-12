from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "intents_raw.csv"
AUG_PATH = DATA_DIR / "intents_augmented.csv"
DEFAULT_TARGET_SIZE = 4200

INTENT_MIN_COUNTS = {
    "launch_app": 180,
    "close_app": 120,
    "web_search": 180,
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
    "web_search": [
        "search for machine learning roadmap",
        "google weather in delhi",
        "find best gpu under 30000",
        "search youtube for python tutorial",
        "look up latest ai news",
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
    "web_search": ["search reddit for ai", "find internships in ai", "google pytorch docs", "look up cuda setup"],
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
    "search": ["serch", "surch"],
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
    "web_search": [
        "serch web for gpu prices",
        "google why fi speed test",
        "look up utube python tutorial",
        "find webside for weather updates",
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

    def bootstrap_raw_dataset(self) -> pd.DataFrame:
        rows: List[Dict[str, str]] = []
        for intent, min_count in INTENT_MIN_COUNTS.items():
            templates = TEMPLATES[intent]
            paraphrases = MANUAL_PARAPHRASES[intent]
            pool = templates + paraphrases
            while len([r for r in rows if r["intent"] == intent]) < min_count:
                text = random.choice(pool)
                roll = random.random()
                if roll < 0.25:
                    text = random.choice(POLITE_PREFIXES) + text
                elif roll < 0.5:
                    text = text + random.choice(POLITE_SUFFIXES)
                elif roll < 0.65:
                    text = random.choice(POLITE_PREFIXES) + text + random.choice(POLITE_SUFFIXES)
                rows.append({"text": text, "intent": intent})

        df = pd.DataFrame(rows)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_PATH, index=False)
        return df

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

    @staticmethod
    def _cap_to_target_size(aug_df: pd.DataFrame, target_size: int) -> pd.DataFrame:
        min_total = sum(INTENT_MIN_COUNTS.values())
        if target_size < min_total:
            raise ValueError(
                f"target_size ({target_size}) is below required intent minimum total ({min_total})"
            )

        selected_parts: List[pd.DataFrame] = []
        selected_indices = set()

        for intent, min_count in INTENT_MIN_COUNTS.items():
            intent_rows = aug_df[aug_df["intent"] == intent]
            if intent_rows.empty:
                continue
            take_n = min(min_count, len(intent_rows))
            sampled = intent_rows.sample(n=take_n, random_state=42)
            selected_parts.append(sampled)
            selected_indices.update(sampled.index.tolist())

        selected_df = (
            pd.concat(selected_parts, ignore_index=False)
            if selected_parts
            else pd.DataFrame(columns=aug_df.columns)
        )

        remaining_needed = target_size - len(selected_df)
        if remaining_needed > 0:
            remaining_pool = aug_df.drop(index=list(selected_indices), errors="ignore")
            if not remaining_pool.empty:
                extra = remaining_pool.sample(
                    n=min(remaining_needed, len(remaining_pool)),
                    random_state=42,
                    replace=False,
                )
                selected_df = pd.concat([selected_df, extra], ignore_index=False)

        remaining_needed = target_size - len(selected_df)
        if remaining_needed > 0:
            extra = aug_df.sample(n=remaining_needed, random_state=42, replace=True)
            selected_df = pd.concat([selected_df, extra], ignore_index=False)

        return selected_df.reset_index(drop=True)

    def augment(self, target_size: int = DEFAULT_TARGET_SIZE) -> pd.DataFrame:
        if RAW_PATH.exists():
            df = pd.read_csv(RAW_PATH)
        else:
            df = self.bootstrap_raw_dataset()

        rows: List[Dict[str, str]] = []
        for _, row in df.iterrows():
            text = str(row["text"])
            intent = str(row["intent"])
            rows.append({"text": text, "intent": intent})
            for aug_text in self._augment_text(text):
                rows.append({"text": aug_text, "intent": intent})
            for stt_text in self._stt_noise_variants(text):
                rows.append({"text": stt_text, "intent": intent})

        for intent, variants in MANUAL_PARAPHRASES.items():
            for variant in variants:
                rows.append({"text": variant, "intent": intent})

        for intent, variants in INTENT_EDGE_CASES.items():
            for variant in variants:
                rows.append({"text": variant, "intent": intent})

        for intent, variants in STT_ERROR_EXAMPLES.items():
            for variant in variants:
                rows.append({"text": variant, "intent": intent})

        aug_df = pd.DataFrame(rows).drop_duplicates().sample(frac=1.0, random_state=42)

        counts = Counter(aug_df["intent"].tolist())
        for intent, min_count in INTENT_MIN_COUNTS.items():
            while counts.get(intent, 0) < max(min_count, 80):
                base = random.choice(TEMPLATES[intent] + MANUAL_PARAPHRASES[intent])
                aug_df.loc[len(aug_df)] = [base, intent]
                counts[intent] += 1

        while len(aug_df) < target_size:
            sampled = aug_df.sample(1, random_state=random.randint(1, 100000)).iloc[0]
            rows_for_intent = TEMPLATES[str(sampled["intent"])] + MANUAL_PARAPHRASES[str(sampled["intent"])]
            aug_df.loc[len(aug_df)] = [random.choice(rows_for_intent), sampled["intent"]]

        aug_df = aug_df.drop_duplicates().reset_index(drop=True)
        if len(aug_df) > target_size:
            aug_df = self._cap_to_target_size(aug_df, target_size)
        elif len(aug_df) < target_size:
            needed = target_size - len(aug_df)
            extras = aug_df.sample(min(needed, len(aug_df)), replace=True, random_state=42)
            aug_df = pd.concat([aug_df, extras], ignore_index=True)

        aug_df = aug_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        aug_df.to_csv(AUG_PATH, index=False)
        return aug_df


if __name__ == "__main__":
    builder = DatasetBuilder(seed=42)
    raw = builder.bootstrap_raw_dataset()
    aug = builder.augment(target_size=DEFAULT_TARGET_SIZE)
    print(f"Raw dataset size: {len(raw)}")
    print(f"Augmented dataset size: {len(aug)}")
    print(f"Raw CSV path: {RAW_PATH}")
    print(f"Augmented CSV path: {AUG_PATH}")
    print("Class distribution:")
    print(aug["intent"].value_counts())
