from __future__ import annotations

import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.config import APPDATA_DIR


class VectorStore:
    def __init__(self, persist_dir: Path | None = None) -> None:
        self.persist_dir = persist_dir or (APPDATA_DIR / "chroma")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Chroma native backend can be unstable on some Windows environments.
        # Enable explicitly when validated on the current machine.
        self._disabled = os.getenv("JARVIS_ENABLE_VECTOR_STORE", "0") != "1"
        self.client = None
        self.collection = None
        if not self._disabled:
            try:
                import chromadb
                from chromadb.config import Settings

                self.client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
                self.collection = self.client.get_or_create_collection("jarvis_conversations")
            except Exception:
                self._disabled = True
                self.client = None
                self.collection = None
        self.embedder = None
        self._embedder_ready = False

    def _ensure_embedder(self) -> None:
        if self._embedder_ready:
            return
        # sentence-transformers does not reliably support Python 3.13 yet.
        if sys.version_info >= (3, 13):
            self.embedder = None
            self._embedder_ready = True
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.embedder = None
        self._embedder_ready = True

    @staticmethod
    def _fallback_embedding(text: str, dim: int = 384) -> List[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _embed(self, text: str) -> List[float]:
        self._ensure_embedder()
        if self.embedder is None:
            return self._fallback_embedding(text)
        try:
            return self.embedder.encode([text], normalize_embeddings=True)[0].tolist()
        except Exception:
            return self._fallback_embedding(text)

    def add_memory(self, text: str, metadata: Dict[str, Any]) -> None:
        if self._disabled or self.collection is None:
            return
        embedding = self._embed(text)
        payload = {"timestamp": time.time(), **metadata}
        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[payload],
            embeddings=[embedding],
        )

    def search_similar(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        if self._disabled or self.collection is None:
            return []
        embedding = self._embed(query)
        results = self.collection.query(query_embeddings=[embedding], n_results=n)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        out: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            out.append({"text": doc, "metadata": meta, "distance": dist})
        return out

