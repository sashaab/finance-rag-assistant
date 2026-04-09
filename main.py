"""
Finance RAG Assistant
=====================
A retrieval-augmented generation (RAG) pipeline for answering
banking product questions using a knowledge base of financial articles.

Pipeline:
  1. Preprocess knowledge base (sentence-level chunking)
  2. Embed chunks with multilingual-e5-base
  3. Retrieve top-N candidates via cosine similarity
  4. Rerank with Qwen3-Reranker-4B
  5. Generate grounded answer with Mistral-Small-3.2-24B
"""

import os
import re
import pickle
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModel
import requests
import time

load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
RERANKER_URL = os.getenv("RERANKER_URL", "https://api.openai.com/v1/rerank")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-small-3.2-24b-instruct")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B")
EMBEDDER_MODEL = os.getenv("EMBEDDER_MODEL", "intfloat/multilingual-e5-base")

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "train_data.csv")
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "questions.csv")
EMBEDDINGS_CACHE = os.getenv("EMBEDDINGS_CACHE", "embeddings.pkl")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "submission.csv")

CHUNK_MAX_LEN = int(os.getenv("CHUNK_MAX_LEN", 400))
CHUNK_MIN_LEN = int(os.getenv("CHUNK_MIN_LEN", 50))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 100))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
CONTEXT_MIN_LEN = int(os.getenv("CONTEXT_MIN_LEN", 150))
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", 0.4))


# =============================================================================
# Stage 1: Preprocessing
# =============================================================================

def preprocess_train_data(path: str = TRAIN_DATA_PATH, max_len: int = CHUNK_MAX_LEN) -> pd.DataFrame:
    """
    Load knowledge base CSV and split long documents into sentence-level chunks.
    Short documents are kept as-is.

    Args:
        path: Path to CSV with columns [id, text].
        max_len: Character threshold above which a document is chunked.

    Returns:
        DataFrame with one chunk per row under column 'text'.
    """
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df)} documents")

    if "text" not in df.columns:
        raise ValueError("Expected column 'text' not found in input CSV")

    long_texts = df[df["text"].str.len() > max_len]
    if len(long_texts) == 0:
        print("All documents within length limit — skipping chunking.")
        return df

    print(f"Found {len(long_texts)} long documents — chunking by sentence...")
    rows = []
    for text in df["text"]:
        parts = re.split(r'(?<=[.!?])\s+', str(text))
        for part in parts:
            if len(part.strip()) > CHUNK_MIN_LEN:
                rows.append({"text": part.strip()})

    result = pd.DataFrame(rows)
    print(f"Chunked corpus: {len(result)} chunks")
    return result


# =============================================================================
# Stage 2: Embedding
# =============================================================================

def load_embedder(model_name: str = EMBEDDER_MODEL):
    """Load HuggingFace tokenizer and model for dense retrieval."""
    print(f"Loading embedder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def get_embeddings(texts: list, tokenizer, model, batch_size: int = 16) -> torch.Tensor:
    """
    Compute mean-pooled embeddings for a list of texts.

    Args:
        texts: List of strings to embed.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.
        batch_size: Number of texts per forward pass.

    Returns:
        Tensor of shape (len(texts), hidden_size).
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding corpus"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb)
    if embeddings:
        return torch.cat(embeddings, dim=0)
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 768
    return torch.empty((0, hidden_size))


def cosine_similarity_torch(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between two sets of vectors."""
    v1 = vec1 / (vec1.norm(dim=1, keepdim=True) + 1e-8)
    v2 = vec2 / (vec2.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mm(v1, v2.T)


def retrieve_top_docs(query: str, df: pd.DataFrame, embeddings: torch.Tensor,
                      tokenizer, model, top_k: int = RETRIEVAL_TOP_K) -> list:
    """
    Retrieve the most similar documents to the query using cosine similarity.

    Returns:
        List of up to top_k document strings.
    """
    if embeddings.numel() == 0:
        return []
    with torch.no_grad():
        inputs = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        query_emb = outputs.last_hidden_state.mean(dim=1)
    sims = cosine_similarity_torch(query_emb, embeddings)[0]
    top_k = min(top_k, sims.size(0))
    top_indices = sims.topk(top_k).indices.tolist()
    return df.iloc[top_indices]["text"].tolist()


# =============================================================================
# Stage 3: Reranking
# =============================================================================

def rerank_docs(query: str, documents: list, key: str, retries: int = 3) -> dict:
    """
    Rerank candidate documents using an external reranker API.

    Truncates documents to 1200 chars before sending.
    Falls back gracefully if the reranker is unavailable.

    Returns:
        Dict with key 'results' containing relevance scores.
    """
    documents = [d[:1200] for d in documents if len(d.strip()) > 20]
    if not documents:
        return {"results": []}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": RERANKER_MODEL,
        "query": query,
        "documents": documents
    }

    for attempt in range(retries):
        try:
            resp = requests.post(RERANKER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            print(f"Reranker returned {resp.status_code}, attempt {attempt + 1}/{retries}")
            time.sleep(2)
        except Exception as e:
            print(f"Reranker error: {e}, attempt {attempt + 1}/{retries}")
            time.sleep(2)

    print("Reranker unavailable — using fallback uniform scores.")
    return {"results": [{"relevance_score": 0.1} for _ in documents]}


# =============================================================================
# Stage 4: Answer Generation
# =============================================================================

PROMPT_WITH_CONTEXT = """
You are an AI assistant for a bank.
Answer only based on the provided context.
If the context does not contain sufficient information, respond exactly:
"Недостаточно данных для точного ответа."

Response format:
1. Clear, complete explanation in natural Russian.
2. Short paragraph — no bullet points or numbered lists.
3. Include examples from context if available.
4. Use all relevant facts from context.
5. Calm, professional tone like a bank consultant.
6. Avoid phrases like "according to the context".

Business context:
Customers ask about banking products — deposits, loans, investments, bonds, insurance.
Provide accurate, factual answers from the knowledge base.
""".strip()

PROMPT_FALLBACK = """
You are an AI assistant for a bank.
Answer customer questions clearly and professionally, avoiding fabrications.
If the knowledge base lacks specific information, use general knowledge about
banking products (deposits, loans, fees, investments, insurance),
but never invent specific numbers, conditions, or product names.
Give a complete, logical answer in natural Russian.
""".strip()


def build_context(candidates: list, rerank_result: dict, top_k: int = RERANK_TOP_K) -> str:
    """Select and join the top-k reranked documents into a context string."""
    results = rerank_result.get("results", [])
    if not results:
        return ""
    scored = list(zip(candidates, [r.get("relevance_score", 0) for r in results]))
    top_docs = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return "\n\n".join([doc for doc, _ in top_docs])


def answer_generation(question: str, df: pd.DataFrame = None,
                      embeddings: torch.Tensor = None,
                      tokenizer=None, model=None) -> str:
    """
    Full RAG pipeline: retrieve → rerank → generate.

    Embeddings are loaded from cache if available, otherwise computed and cached.
    Tokenizer/model can be passed in to avoid reloading between calls.
    """
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    # Load or build embedding index
    if df is None or embeddings is None:
        if os.path.exists(EMBEDDINGS_CACHE):
            with open(EMBEDDINGS_CACHE, "rb") as f:
                df, embeddings = pickle.load(f)
            print(f"Loaded embeddings cache: {len(df)} chunks")
        else:
            df = preprocess_train_data()
            tokenizer, model = load_embedder()
            embeddings = get_embeddings(df["text"].tolist(), tokenizer, model)
            with open(EMBEDDINGS_CACHE, "wb") as f:
                pickle.dump((df, embeddings), f)
            print(f"Embeddings cached to {EMBEDDINGS_CACHE}")

    if tokenizer is None or model is None:
        tokenizer, model = load_embedder()

    candidates = retrieve_top_docs(question, df, embeddings, tokenizer, model)
    rerank_result = rerank_docs(question, candidates, EMBEDDER_API_KEY)
    context = build_context(candidates, rerank_result)

    system_prompt = PROMPT_FALLBACK if len(context.strip()) < CONTEXT_MIN_LEN else PROMPT_WITH_CONTEXT

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Вопрос: {question}\n\nКонтекст:\n{context}"}
            ],
            temperature=GENERATION_TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Generation error: {e}"


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    questions = pd.read_csv(QUESTIONS_PATH)
    questions_list = questions["Вопрос"].tolist()

    # Pre-load embedder once to avoid reloading on every question
    print("Initialising embedding index...")
    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, "rb") as f:
            corpus_df, corpus_embeddings = pickle.load(f)
        print(f"Cache loaded: {len(corpus_df)} chunks")
    else:
        corpus_df = preprocess_train_data()
        tok, mdl = load_embedder()
        corpus_embeddings = get_embeddings(corpus_df["text"].tolist(), tok, mdl)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump((corpus_df, corpus_embeddings), f)

    tok, mdl = load_embedder()

    answer_list = []
    for question in tqdm(questions_list, desc="Generating answers"):
        answer = answer_generation(
            question=question,
            df=corpus_df,
            embeddings=corpus_embeddings,
            tokenizer=tok,
            model=mdl
        )
        answer_list.append(answer)

    questions["Ответы на вопрос"] = answer_list
    questions.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(answer_list)} answers to {OUTPUT_PATH}")
