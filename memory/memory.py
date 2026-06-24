from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class LocalVectorStore:
    def __init__(self):
        # self.model = SentenceTransformer("BAAI/bge-m3")
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        os.makedirs("agent_mem", exist_ok=True)
        if os.path.exists("agent_mem/index.faiss"):
            self.index = faiss.read_index("agent_mem/index.faiss")

            with open("agent_mem/documents.pkl", "rb") as f:
                self.documents = pickle.load(f)
        else:
            dim = self.model.get_sentence_embedding_dimension()

            self.index = faiss.IndexFlatIP(dim)
            self.documents = []

    def add(self, text):
        emb = self.model.encode(
            [text],
            normalize_embeddings=True
        )

        self.index.add(emb)
        self.documents.append(text)

    def search(self, query, k=5):
        emb = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        scores, ids = self.index.search(emb, k)

        return [
            (float(score), self.documents[idx])
            for score, idx in zip(scores[0], ids[0])
        ]

    def save(self):
        faiss.write_index(self.index, "agent_mem/index.faiss")

        with open("agent_mem/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)