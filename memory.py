# memory.py - Long-term memory with vector search

import os
import pickle
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import faiss
from sentence_transformers import SentenceTransformer

from config import config


@dataclass
class MemoryEntry:
    """A single memory entry."""
    text: str
    embedding: List[float] = field(default_factory=list)
    access_count: int = 0
    created_at: float = field(default_factory=lambda: __import__('time').time())


class LocalVectorStore:
    """Vector store for memory using FAISS."""

    def __init__(self):
        self.model = SentenceTransformer(config.models.embedding_model)
        self.dim = self.model.get_sentence_embedding_dimension()

        os.makedirs(config.memory.memory_dir, exist_ok=True)

        index_path = os.path.join(config.memory.memory_dir, "index.faiss")
        docs_path = os.path.join(config.memory.memory_dir, "documents.pkl")

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents: List[MemoryEntry] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.documents = []

    def add(self, text: str) -> int:
        """Add a new memory."""
        emb = self.model.encode([text], normalize_embeddings=True)
        self.index.add(emb)
        entry = MemoryEntry(text=text, embedding=emb[0].tolist())
        self.documents.append(entry)
        return len(self.documents) - 1

    def search(self, query: str, k: int = 5) -> List[Tuple[float, str]]:
        """Search for similar memories."""
        if len(self.documents) == 0:
            return []

        emb = self.model.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(emb, min(k, len(self.documents)))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx >= 0 and idx < len(self.documents):
                entry = self.documents[idx]
                entry.access_count += 1
                results.append((float(score), entry.text))

        return results

    def get_all(self) -> List[MemoryEntry]:
        """Get all memories."""
        return self.documents

    def save(self):
        """Save index and documents."""
        index_path = os.path.join(config.memory.memory_dir, "index.faiss")
        docs_path = os.path.join(config.memory.memory_dir, "documents.pkl")

        faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def clear(self):
        """Clear all memories."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.documents = []
        self.save()

    def __len__(self) -> int:
        return len(self.documents)


class MemoryManager:
    """High-level memory management with deduplication and merging."""

    def __init__(self):
        self.store = LocalVectorStore()

    def process(self, text: str) -> str:
        """
        Process new text: deduplicate, merge related, or add new.
        Returns: 'duplicate', 'update', or 'new'
        """
        if len(self.store) == 0:
            self.store.add(text)
            return "new"

        matches = self.store.search(text, k=1)
        if not matches:
            self.store.add(text)
            return "new"

        score, memory = matches[0]

        print(f"\nBest Match ({score:.4f})")
        print(memory[:200] + ("..." if len(memory) > 200 else ""))

        # Exact duplicate
        if score >= config.memory.duplicate_threshold:
            return "duplicate"

        # Related memory - merge
        if score >= config.memory.related_threshold:
            merged = self._merge(memory, text)
            idx = self._find_memory_index(memory)
            if idx >= 0:
                self.store.documents[idx].text = merged
                self._rebuild_index()
            return "update"

        # New topic
        self.store.add(text)
        return "new"

    def _find_memory_index(self, text: str) -> int:
        """Find index of a memory by text."""
        for i, entry in enumerate(self.store.documents):
            if entry.text == text:
                return i
        return -1

    def _merge(self, old: str, new: str) -> str:
        """Simple merge strategy - append with separator."""
        return f"{old}\n---\n{new}"

    def _rebuild_index(self):
        """Rebuild FAISS index from documents."""
        self.store.index = faiss.IndexFlatIP(self.store.dim)
        if self.store.documents:
            texts = [doc.text for doc in self.store.documents]
            embs = self.store.model.encode(texts, normalize_embeddings=True)
            self.store.index.add(embs)

    def search(self, query: str, k: int = None) -> List[Tuple[float, str]]:
        """Search memories."""
        k = k or config.memory.search_k
        return self.store.search(query, k)

    def get_context(self, query: str) -> str:
        """Get formatted memory context for a query."""
        results = self.search(query, k = config.memory.max_memories_in_context
        if not results:
            return ""

        lines = ["Relevant memories from previous conversations:"]
        for i, (score, text) in enumerate(results, 1):
            if score >= config.memory.search_score_threshold:
                lines.append(f"\n[{i}] (relevance: {score:.2f})")
                lines.append(text)
        return "\n".join(lines)

    def get_all_memories(self) -> List[MemoryEntry]:
        """Get all memory entries."""
        return self.store.get_all()

    def save(self):
        """Save memory to disk."""
        self.store.save()

    def clear(self):
        """Clear all memories."""
        self.store.clear()

    def __len__(self) -> int:
        return len(self.store)