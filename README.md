# Finance RAG Assistant

> LLM-powered Q&A system for banking products.

---

## What it does

Answers customer questions about banking products (deposits, bonds, derivatives, loans, insurance) by retrieving relevant passages from a 350-article knowledge base and generating a grounded response via an LLM.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│  Stage 1 · Preprocessing        │
│  Sentence-level chunking        │
│  (350 docs → ~N chunks)         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 2 · Dense Retrieval      │
│  multilingual-e5-base           │
│  Cosine similarity → top-100    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 3 · Reranking            │
│  Qwen3-Reranker-4B              │
│  Scores candidates → top-5      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Stage 4 · Generation           │
│  Mistral-Small-3.2-24B          │
│  Context-grounded prompt        │
│  Fallback for low-context cases │
└────────────────┬────────────────┘
                 │
                 ▼
             Answer
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| **Sentence-level chunking** | Preserves semantic units; avoids splitting mid-fact |
| **Two-stage retrieval (embedder → reranker)** | Embedder casts a wide net (top-100); reranker picks the 5 most relevant — reduces hallucination vs. naive top-k |
| **Dual prompt strategy** | If retrieved context < 150 chars (low-signal query), switches to general banking knowledge mode without fabricating specific numbers |
| **Embedding cache** | `embeddings.pkl` computed once, reused across runs — avoids re-embedding 350 docs on every question |
| **All config via `.env`** | No hardcoded endpoints or model names; swap models without touching code |

---

## Stack

| Component | Model / Tool |
|---|---|
| Embedder | `intfloat/multilingual-e5-base` |
| Reranker | `Qwen/Qwen3-Reranker-4B` |
| Generator | `mistralai/mistral-small-3.2-24b-instruct` |
| Language | Python 3.12 |
| Core libs | PyTorch, HuggingFace Transformers, OpenAI SDK |

---

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/finance-rag-assistant.git
cd finance-rag-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add your LLM_API_KEY and EMBEDDER_API_KEY

# 4. Add data files (not included in repo)
# Place train_data.csv and questions.csv in the project root

# 5. Run
python main.py
# Outputs: submission.csv
```

---

## Project structure

```
finance-rag-assistant/
├── main.py            # Full pipeline: preprocess → embed → retrieve → rerank → generate
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template (copy to .env)
├── .gitignore         # Excludes secrets, data files, cache
└── README.md
```

---

## Notes

- `train_data.csv` and `questions.csv` are data files — not included in this repo
- `embeddings.pkl` is auto-generated on first run and cached locally
- This project uses external LLM/reranker APIs; configure endpoints in `.env`
