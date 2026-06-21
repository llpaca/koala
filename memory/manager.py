# manager.py

from .memory import LocalVectorStore


class MemoryManager:
    def __init__(
        self,
        duplicate_threshold=0.97,
        related_threshold=0.80,
    ):
        self.store = LocalVectorStore()

        self.duplicate_threshold = duplicate_threshold
        self.related_threshold = related_threshold

    def process(self, text: str):
        """
        Returns:
            duplicate
            update
            new
        """

        if len(self.store.documents) == 0:
            self.store.add(text)
            return "new"

        matches = self.store.search(text, k=1)

        if not matches:
            self.store.add(text)
            return "new"

        score, memory = matches[0]

        print(f"\nBest Match ({score:.4f})")
        print(memory)

        # Exact duplicate
        if score >= self.duplicate_threshold:
            return "duplicate"

        # Related memory
        if score >= self.related_threshold:

            merged = self.merge(memory, text)

            idx = self.store.documents.index(memory)

            self.store.documents[idx] = merged

            emb = self.store.model.encode(
                [merged],
                normalize_embeddings=True
            )

            self.store.index.reconstruct(idx)

            self.rebuild_index()

            return "update"

        # Brand new topic
        self.store.add(text)
        return "new"

    def merge(self, old_memory, new_memory):
        """
        Simple merge strategy.
        Replace later with LLM summarization.
        """

        return f"{old_memory}\n{new_memory}"

    def rebuild_index(self):

        dim = self.store.model.get_sentence_embedding_dimension()

        import faiss

        self.store.index = faiss.IndexFlatIP(dim)

        embeddings = self.store.model.encode(
            self.store.documents,
            normalize_embeddings=True
        )

        self.store.index.add(embeddings)

    def save(self):
        self.store.save()