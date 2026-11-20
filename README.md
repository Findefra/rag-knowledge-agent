# RAG Knowledge Agent

A clean **Retrieval-Augmented Generation (RAG)** agent that ingests documents, retrieves relevant context, and produces grounded answers with citations.

This project demonstrates core skills of an **AI Agent Engineer**:

- Document ingestion pipeline
- Vector / sparse retrieval
- Grounded answer generation with citations
- Clear separation between retrieval and generation
- Easy upgrade path to production stacks (Chroma, sentence-transformers, LLM)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python agent.py
```

The demo comes with three sample documents about agent engineering, RAG best practices, and tool calling.

### Example questions

```
What are the core patterns in AI agent engineering?
Explain RAG best practices for chunking.
How should agents handle tool failures?
```

You can also ingest your own files:

```
ingest path/to/document.md
```

## Architecture

```
Documents
    ↓
Ingestion (text → Document objects)
    ↓
SimpleVectorStore (TF-IDF for demo)
    ↓
Query → Retrieval (top-k chunks)
    ↓
Grounded Answer + Citations
```

## Design Choices

| Layer              | Demo Implementation          | Production Upgrade                          |
|--------------------|------------------------------|---------------------------------------------|
| Embeddings / Index | TF-IDF (scikit-learn)        | sentence-transformers + Chroma / Pinecone   |
| Generation         | Extractive summary           | LLM with grounded prompt + structured output|
| Chunking           | Whole document               | Recursive / semantic chunking + overlap     |
| Evaluation         | Manual                       | RAGAS, custom retrieval metrics             |

## Skills Demonstrated

- Building end-to-end RAG pipelines
- Retrieval interfaces that can be swapped
- Citation / source tracking
- Agent-style query interface on top of retrieval
- Designing for testability and future LLM integration

## Next Steps (Production)

1. Replace `SimpleVectorStore` with Chroma + embedding model
2. Add proper chunking (e.g. 300–500 tokens with overlap)
3. Swap `_generate_answer` for an LLM call with a system prompt that forces citations
4. Add evaluation (retrieval precision, answer faithfulness)
5. Add hybrid search and reranking

## License

MIT
