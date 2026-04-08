import re
import pickle
import textwrap
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi

from src.embeddings import get_embedder, BaseEmbedder
from src.utils import split_into_sentences, extract_title

@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    doc_title: str
    text: str
    start_sentence: int
    end_sentence: int


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    query: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]



class SentenceWindowChunker:
    def __init__(self, window_size=5, step_size=3, min_chars=100):
        self.window_size = window_size
        self.step_size = step_size
        self.min_chars = min_chars

    def chunk_document(self, doc_id, text, start_chunk_id=0):
        title = extract_title(text)
        sentences = split_into_sentences(text)

        chunks = []
        cid = start_chunk_id
        i = 0

        while i < len(sentences):
            window = sentences[i:i+self.window_size]
            chunk_text = " ".join(window)

            if len(chunk_text) >= self.min_chars:
                chunks.append(Chunk(
                    cid, doc_id, title, chunk_text,
                    i, i + len(window) - 1
                ))
                cid += 1

            i += self.step_size

        return chunks


class FAISSVectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def add_chunks(self, chunks, embeddings):
        embeddings = self._normalize(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_emb, k):
        query_emb = self._normalize(query_emb.reshape(1, -1))
        scores, indices = self.index.search(query_emb, k)

        results = []
        for s, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append(RetrievedChunk(self.chunks[idx], float(s)))

        return results

    def _normalize(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-8)

    def save(self, path):
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".chunks", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path, dim):
        store = cls(dim)
        store.index = faiss.read_index(path + ".faiss")
        with open(path + ".chunks", "rb") as f:
            store.chunks = pickle.load(f)
        return store



class HybridRetriever:

    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder

        self.tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

        texts = [c.text for c in chunks]
        self.embeddings = embedder.encode(texts, convert_to_numpy=True)

    def _normalize(self, x):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-8)

    def search(self, query, k=5):

    
        bm25_scores = self.bm25.get_scores(query.lower().split())

       
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        dense_scores = np.dot(self.embeddings, q_emb)


        bm25_scores /= (np.max(bm25_scores) + 1e-8)
        dense_scores /= (np.max(dense_scores) + 1e-8)

       
        alpha = 0.7 if len(query.split()) <= 3 else 0.4

        final_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

        idx = np.argsort(final_scores)[-k:][::-1]

        return [
            RetrievedChunk(self.chunks[i], float(final_scores[i]))
            for i in idx
        ]



class QuantumRAGPipeline:

    INDEX_PATH = "index/quantum_rag"

    def __init__(self, data_dir="data", top_k=4):
        self.data_dir = Path(data_dir)
        self.top_k = top_k

        print("⚙ Loading embeddings...")
        self.embedder: BaseEmbedder = get_embedder(prefer_sbert=True)

        print("⚙ Loading LLM (Flan-T5)...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        self.vector_store = None
        self.retriever = None


   

    def ingest(self, force_rebuild=False):
        index_base = Path(self.INDEX_PATH)
        index_base.parent.mkdir(parents=True, exist_ok=True)

        if not force_rebuild and (index_base.parent / (index_base.name + ".faiss")).exists():
            print(" Loading index...")
            self.vector_store = FAISSVectorStore.load(str(index_base), 384)

            if hasattr(self.embedder, "fit"):
                corpus = [c.text for c in self.vector_store.chunks]
                self.embedder.fit(corpus)

        else:
            print(" Building index...")
            chunker = SentenceWindowChunker()
            all_chunks = []

            for doc in sorted(self.data_dir.glob("*.txt")):
                text = doc.read_text(encoding="utf-8")
                chunks = chunker.chunk_document(doc.stem, text, len(all_chunks))
                all_chunks.extend(chunks)

            texts = [c.text for c in all_chunks]

            if hasattr(self.embedder, "fit"):
                self.embedder.fit(texts)

            embeddings = self.embedder.encode(texts, convert_to_numpy=True)

            self.vector_store = FAISSVectorStore(embeddings.shape[1])
            self.vector_store.add_chunks(all_chunks, embeddings)
            self.vector_store.save(str(index_base))

            print(f" {len(all_chunks)} chunks indexed")

       
        self.retriever = HybridRetriever(self.vector_store.chunks, self.embedder)


    

    def _build_prompt(self, query, retrieved):
         context = "\n\n".join([
        r.chunk.text for r in retrieved
    ])
         return f"""
You are a question-answering system.

Answer the question using ONLY the context below.

Select ONE sentence from the context that best answers the question.

Rules:
- Copy the sentence exactly
- Do NOT modify wording
- Do NOT combine multiple sentences
- Do NOT add extra information
Context:
{context}

Question: {query}

Full Answer:
"""



    def generate(self, query):
        retrieved = self.retriever.search(query, self.top_k)
        prompt = self._build_prompt(query, retrieved)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                min_length=40
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return RAGResponse(query, answer, retrieved)

    def ask(self, q):
        return self.generate(q).answer




if __name__ == "__main__":
    rag = QuantumRAGPipeline()
    rag.ingest()

    while True:
        q = input("Ask: ")
        if q == "exit":
            break
        print("\n", rag.ask(q), "\n")