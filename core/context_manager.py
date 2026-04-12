from __future__ import annotations

import hashlib
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from core.config import CONTEXT_WINDOW_SIZE


@dataclass
class ConversationTurn:
    role: str
    text: str
    timestamp: float
    embedding: Optional[np.ndarray] = None


class ContextManager:
    def __init__(self, window_size: int = CONTEXT_WINDOW_SIZE) -> None:
        self.window_size = window_size
        self._turns: Deque[ConversationTurn] = deque(maxlen=window_size)
        self._embedder = None
        self._embedder_ready = False
        self._lock = threading.Lock()

    def _ensure_embedder(self) -> None:
        if self._embedder_ready:
            return
        if sys.version_info >= (3, 13):
            self._embedder = None
            self._embedder_ready = True
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._embedder = None
        self._embedder_ready = True

    @staticmethod
    def _fallback_embedding(text: str, dim: int = 384) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        self._ensure_embedder()
        if self._embedder is None:
            return self._fallback_embedding(text)
        try:
            vector = self._embedder.encode([text], normalize_embeddings=True)
            return np.asarray(vector[0], dtype=np.float32)
        except Exception:
            return self._fallback_embedding(text)

    def add_turn(self, role: str, text: str, *, embed: bool = True) -> None:
        embedding = self._embed_text(text) if embed else None
        turn = ConversationTurn(
            role=role,
            text=text,
            timestamp=time.time(),
            embedding=embedding,
        )
        with self._lock:
            self._turns.append(turn)

    def embed_recent_missing(self, limit: int = 4) -> int:
        pending: List[ConversationTurn] = []
        with self._lock:
            for turn in reversed(self._turns):
                if turn.embedding is None:
                    pending.append(turn)
                    if len(pending) >= max(1, int(limit)):
                        break

        updated = 0
        for turn in pending:
            try:
                turn.embedding = self._embed_text(turn.text)
                updated += 1
            except Exception:
                pass
        return updated

    def get_window(self) -> List[Dict[str, str]]:
        with self._lock:
            turns = list(self._turns)
        return [{"role": t.role, "text": t.text} for t in turns]

    def resolve_reference(self, query: str, top_k: int = 1) -> List[Dict[str, str]]:
        embedding = self._embed_text(query)
        if embedding is None:
            return self.get_window()[-top_k:]

        scored = []
        with self._lock:
            turns = list(self._turns)

        for turn in turns:
            if turn.embedding is None:
                continue
            similarity = float(np.dot(embedding, turn.embedding))
            scored.append((similarity, turn))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"role": turn.role, "text": turn.text}
            for _, turn in scored[: max(top_k, 1)]
        ]

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
