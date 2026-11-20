"""
RAG Knowledge Agent
Document ingestion → vector store → retrieval → grounded answer generation.
Demonstrates core AI Agent Engineer skills: RAG pipelines, embeddings,
retrieval, citation, and agentic querying.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

# ---------------------------------------------------------------------------
# Lightweight local RAG (no external API keys required)
# Uses TF-IDF for retrieval so the demo runs with minimal dependencies.
# Swap the retriever for Chroma + sentence-transformers in production.
# ---------------------------------------------------------------------------

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class Document:
    id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    text: str


class SimpleVectorStore:
    """Minimal TF-IDF backed store for demo purposes."""

    def __init__(self):
        self.documents: List[Document] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def add(self, documents: List[Document]) -> None:
        self.documents.extend(documents)
        self._rebuild()

    def _rebuild(self) -> None:
        if not self.documents or not HAS_SKLEARN:
            return
        texts = [d.content for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        if not self.documents:
            return []

        if not HAS_SKLEARN or self.vectorizer is None:
            # Fallback: simple keyword overlap
            query_terms = set(query.lower().split())
            scored = []
            for doc in self.documents:
                doc_terms = set(doc.content.lower().split())
                overlap = len(query_terms & doc_terms)
                if overlap > 0:
                    scored.append((overlap, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                RetrievedChunk(document=doc, score=float(score), text=doc.content[:500])
                for score, doc in scored[:top_k]
            ]

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            doc = self.documents[idx]
            results.append(
                RetrievedChunk(
                    document=doc,
                    score=float(scores[idx]),
                    text=doc.content[:800],
                )
            )
        return results


class KnowledgeAgent:
    """
    Simple RAG agent.
    1. Retrieves relevant chunks
    2. Builds a grounded context
    3. Produces an answer with citations
    """

    def __init__(self, store: Optional[SimpleVectorStore] = None):
        self.store = store or SimpleVectorStore()
        self.run_id = str(uuid4())[:8]

    def ingest_text(self, text: str, source: str = "manual") -> str:
        doc_id = str(uuid4())[:8]
        doc = Document(id=doc_id, content=text.strip(), source=source)
        self.store.add([doc])
        return doc_id

    def ingest_file(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = p.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(text, source=str(p.name))

    def query(self, question: str, top_k: int = 3) -> dict:
        print(f"[{self.run_id}] Query: {question}")

        chunks = self.store.search(question, top_k=top_k)
        print(f"[{self.run_id}] Retrieved {len(chunks)} chunks")

        if not chunks:
            return {
                "answer": "I could not find relevant information in the knowledge base.",
                "citations": [],
                "retrieved": [],
            }

        # Build context
        context_parts = []
        citations = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] (source: {chunk.document.source})\n{chunk.text}")
            citations.append({
                "id": i,
                "source": chunk.document.source,
                "score": round(chunk.score, 4),
                "doc_id": chunk.document.id,
            })

        context = "\n\n".join(context_parts)

        # Simple extractive-style answer (replace with LLM in production)
        answer = self._generate_answer(question, context, chunks)

        return {
            "answer": answer,
            "citations": citations,
            "retrieved": [
                {
                    "source": c.document.source,
                    "score": round(c.score, 4),
                    "snippet": c.text[:200] + ("..." if len(c.text) > 200 else ""),
                }
                for c in chunks
            ],
        }

    def _generate_answer(self, question: str, context: str, chunks: List[RetrievedChunk]) -> str:
        """
        Placeholder for LLM generation.
        In production: call OpenAI / Anthropic / local model with a grounded prompt.
        Here we return a structured summary of the top evidence.
        """
        if not chunks:
            return "No relevant documents found."

        top = chunks[0]
        answer = (
            f"Based on the knowledge base, here is the most relevant information "
            f"(source: {top.document.source}, relevance: {top.score:.2f}):\n\n"
            f"{top.text[:600]}"
        )
        if len(chunks) > 1:
            answer += f"\n\n(Additional supporting sources: {', '.join(c.document.source for c in chunks[1:])})"
        return answer


# ---------------------------------------------------------------------------
# Sample knowledge + CLI
# ---------------------------------------------------------------------------
SAMPLE_DOCS = [
    {
        "source": "agent_engineering.md",
        "content": """
AI Agent Engineering focuses on building systems that can reason, use tools, maintain memory,
and take actions toward goals. Core patterns include ReAct (Reason + Act), Plan-and-Execute,
and multi-agent collaboration. Evaluation of agents requires both final-answer accuracy and
process metrics such as tool-use correctness and trajectory quality.
        """.strip(),
    },
    {
        "source": "rag_best_practices.md",
        "content": """
Retrieval-Augmented Generation (RAG) improves LLM factuality by grounding answers in external documents.
Best practices include: high-quality chunking (200-500 tokens), hybrid search (keyword + dense),
reranking, citation generation, and continuous evaluation of retrieval precision/recall.
Chunk overlap of 10-20% often improves continuity for multi-sentence answers.
        """.strip(),
    },
    {
        "source": "tool_calling.md",
        "content": """
Tool calling (function calling) allows models to invoke external functions with structured arguments.
Modern APIs support parallel tool calls and strict JSON schema validation. Agents should handle
tool failures gracefully, support retries, and log every tool invocation for observability.
        """.strip(),
    },
]


def main():
    print("=" * 60)
    print("RAG Knowledge Agent Demo")
    print("=" * 60)

    agent = KnowledgeAgent()

    # Ingest sample knowledge
    print("\nIngesting sample documents...")
    for doc in SAMPLE_DOCS:
        doc_id = agent.ingest_text(doc["content"], source=doc["source"])
        print(f"  + {doc['source']} (id={doc_id})")

    print("\nReady. Ask questions about the knowledge base.")
    print("Commands: 'quit', 'ingest <file>', or just type a question.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        if user_input.lower().startswith("ingest "):
            path = user_input[7:].strip()
            try:
                doc_id = agent.ingest_file(path)
                print(f"Ingested {path} → id={doc_id}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        result = agent.query(user_input)
        print("\n--- Answer ---")
        print(result["answer"])
        print("\n--- Citations ---")
        for c in result["citations"]:
            print(f"  [{c['id']}] {c['source']} (score={c['score']})")
        print()


if __name__ == "__main__":
    main()
