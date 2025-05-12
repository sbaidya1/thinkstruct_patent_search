"""
Search logic for patent claim retrieval and ranking.

Includes:
- Semantic search using OpenAI embeddings + ChromaDB
- Reranking results with a cross-encoder
- Optional metadata filtering and preference boosting
- Combination of semantic, rerank, and filter signals
"""

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

# load the cross-encoder model globally for reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def run_search(query, persist_dir, classification, keyword, title,
               classification_mode, keyword_mode, title_mode):
  """
  Top-level function that runs semantic search, reranks with cross encoder,
  and applies metadata-based preference weighting.

  Returns: 
    A list of top-ranked search results (max 15), formatted for display
  """
  # phase 1: semantic vector search using OpenAI embeddings
  candidates = semantic_search(query, persist_dir)

  # phase 2: reranking using a cross-encoder
  cross_scores = rerank_with_cross_encoder(query, candidates)

  # apply metadata filters (required/preferred) 
  filter_weights = compute_filter_weights(
    candidates, classification, keyword, title,
    classification_mode, keyword_mode, title_mode
  )

  # combine cross-encoder scores and filter weights to get final score
  combined = combine_reranking(candidates, cross_scores, filter_weights)

  results = format_results(combined, limit=15)

  return results


def semantic_search(query, persist_dir):
  """
  Performs the initial semantic search using OpenAI embeddings + ChromaDB.
  """
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
  return vectordb.similarity_search(query, k=75)


def rerank_with_cross_encoder(query, docs):
  """
  Uses a cross-encoder to compute relevance scores between the query and each document.

  Returns:
  - A list of float scores (one per document)
  """
  pairs = [[query, doc.page_content] for doc in docs]
  return cross_encoder.predict(pairs)


def compute_filter_weights(docs, classification, keyword, title,
                          classification_mode, keyword_mode, title_mode):
  """
  Applies required filters (exclusion) and preferred filters (soft boosts).
  Returns:
  - A list of weights (floats) or None for excluded documents
  """
  weights = []
  for doc in docs:
    meta = doc.metadata
    title_match = meta.get("title", "").lower()
    abstract_match = meta.get("abstract", "").lower()
    classification_match = meta.get("classification", "").lower()

    # apply required filters; exclude document if it fails any (by setting weight to none)
    if classification_mode == "required" and classification and not classification_match.startswith(classification.lower()):
      weights.append(None)
      continue
    if title_mode == "required" and title and title.lower() != title_match:
      weights.append(None)
      continue
    if keyword_mode == "required" and keyword and keyword.lower() not in title_match and keyword.lower() not in abstract_match:
      weights.append(None)
      continue

    # preferred filters; soft boosts to relevance score
    weight = 1.0
    if classification_mode == "preferred" and classification and classification_match.startswith(classification.lower()):
      weight += 0.1
    if title_mode == "preferred" and title and title.lower() == title_match:
      weight += 0.3
    if keyword_mode == "preferred" and keyword and (keyword.lower() in title_match or keyword.lower() in abstract_match):
      weight += 0.15

    weights.append(weight)

  return weights


def combine_reranking(docs, cross_scores, filter_weights):
  """
  Multiplies cross-encoder scores by filter weights.
  Skips documents that fail required filters (weight is None).

  Returns:
  - A list of (score, document) tuples
  """
  combined = []
  for doc, score, weight in zip(docs, cross_scores, filter_weights):
    if weight is None:
      continue
    final_score = score * weight
    combined.append((final_score, doc))
  return combined


def format_results(scored_docs, limit=15):
  """
  Sorts combined scores and formats top results for template rendering.

  Returns:
  - List of dicts with metadata and claim text
  """
  scored_docs.sort(key=lambda x: x[0], reverse=True)
  return [{
      "document_number": doc.metadata.get("doc_number", ""),
      "title": doc.metadata.get("title", ""),
      "classification": doc.metadata.get("classification", ""),
      "claim": doc.page_content,
      "score": round(score, 3)
  } for score, doc in scored_docs[:limit]]

