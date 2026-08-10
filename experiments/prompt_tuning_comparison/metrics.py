"""
Evaluation metrics for GraphRAG experiment comparison.

All metrics work purely on text (no model access required at inference time).
Optional BERTScore requires: pip install bert-score
"""

import math
import re
from collections import Counter
from typing import Optional

# ─── Tokenisation helpers ─────────────────────────────────────────────────────

# Words that carry no semantic content
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
        "and", "or", "for", "with", "that", "this", "what", "how", "when",
        "where", "who", "which", "do", "does", "did", "have", "has", "had",
        "be", "been", "being", "at", "by", "from", "on", "as", "it", "its",
        "não", "é", "são", "de", "da", "do", "das", "dos", "em", "para",
        "por", "com", "se", "que", "um", "uma", "os", "as",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase word-level tokenisation, strips punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


def _content_tokens(text: str) -> list[str]:
    """Tokenise and remove stopwords."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 2]


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences."""
    return [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]


def _token_f1(pred: list[str], ref: list[str]) -> float:
    """Micro-averaged token F1 between two token lists."""
    if not pred or not ref:
        return 0.0
    pc, rc = Counter(pred), Counter(ref)
    common = sum((pc & rc).values())
    precision = common / len(pred)
    recall = common / len(ref)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ─── Named-entity extraction (regex approximation) ───────────────────────────

def _extract_entities(text: str) -> set[str]:
    """
    Lightweight entity extraction via regex heuristics:
    - Multi-word proper nouns (e.g. "Hepatite B")
    - Uppercase abbreviations (e.g. "MELD", "HCC", "ALT")
    - Numeric values with units (e.g. "3.5 mg", "120 mmHg")
    """
    proper_nouns = set(re.findall(r"\b[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ][a-záéíóúâêîôûãõç]+(?:\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÇ][a-záéíóúâêîôûãõç]+)*\b", text))
    abbreviations = set(re.findall(r"\b[A-Z]{2,6}\b", text))
    return proper_nouns | abbreviations


# ═══════════════════════════════════════════════════════════════════════════════
# OBRIGATÓRIAS
# ═══════════════════════════════════════════════════════════════════════════════

def fact_score(response: str, reference: Optional[str] = None) -> float:
    """
    Simulated FactScore.

    With reference: fraction of response sentences whose content has high
    Jaccard overlap (≥ 0.35) with at least one reference sentence.

    Without reference: heuristic based on entity and numeric density — text
    with many specific facts (numbers, named entities) scores higher.

    Returns a float in [0, 1].
    """
    if not response or response.startswith("ERROR:"):
        return 0.0

    resp_sentences = _split_sentences(response)
    if not resp_sentences:
        return 0.0

    if reference:
        ref_sentences = _split_sentences(reference)
        supported = 0
        for rs in resp_sentences:
            rs_tokens = set(_content_tokens(rs))
            if not rs_tokens:
                continue
            for ref_s in ref_sentences:
                ref_tokens = set(_content_tokens(ref_s))
                if _jaccard(rs_tokens, ref_tokens) >= 0.35:
                    supported += 1
                    break
        return supported / len(resp_sentences)

    # Heuristic: density of numbers + named entities per 100 words
    words = _tokenize(response)
    if not words:
        return 0.0
    numbers = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", response))
    entities = len(_extract_entities(response))
    density = (numbers + entities) / (len(words) / 100)
    # Calibrate so ~15 facts/100 words → score 1.0
    return min(density / 15.0, 1.0)


def faithfulness(response: str, query: str) -> float:
    """
    Faithfulness: measures how well the response addresses the query.

    Computed as the recall of meaningful query tokens found anywhere in
    the response. A high score means the response directly engages with
    the concepts asked about.

    Returns a float in [0, 1].
    """
    if not response or response.startswith("ERROR:") or not query:
        return 0.0

    query_tokens = set(_content_tokens(query))
    if not query_tokens:
        return 0.0

    resp_tokens = set(_content_tokens(response))
    return len(query_tokens & resp_tokens) / len(query_tokens)


def recall_at_k(
    response: str,
    reference_keywords: Optional[list[str]] = None,
    query: Optional[str] = None,
    k: int = 5,
) -> float:
    """
    Recall@k: fraction of reference keywords found in the first k sentences.

    Falls back to using content tokens from the query when no explicit
    keyword list is provided.

    Returns a float in [0, 1].
    """
    if not response or response.startswith("ERROR:"):
        return 0.0

    sentences = _split_sentences(response)
    top_k_text = " ".join(sentences[:k]).lower()

    if reference_keywords:
        keywords = [kw.lower() for kw in reference_keywords]
    elif query:
        keywords = _content_tokens(query)
    else:
        return 0.0

    if not keywords:
        return 0.0

    found = sum(1 for kw in keywords if kw in top_k_text)
    return found / len(keywords)


def radgraph_f1(
    response: str,
    reference: Optional[str] = None,
    query: Optional[str] = None,
) -> float:
    """
    Approximate RadGraph-F1 via named entity extraction.

    With reference: entity-level F1 comparing entities extracted from
    the response against those extracted from the reference.

    Without reference: entity-level overlap between query entities and
    response entities (proxy for clinical entity coverage).

    Returns a float in [0, 1].
    """
    if not response or response.startswith("ERROR:"):
        return 0.0

    resp_entities = _extract_entities(response)

    if reference:
        ref_entities = _extract_entities(reference)
    elif query:
        ref_entities = _extract_entities(query) | set(_content_tokens(query))
    else:
        # Self-contained: entity density relative to word count
        words = _tokenize(response)
        if not words:
            return 0.0
        return min(len(resp_entities) / max(len(words) * 0.08, 1), 1.0)

    if not ref_entities:
        return 0.0

    common = resp_entities & ref_entities
    precision = len(common) / len(resp_entities) if resp_entities else 0.0
    recall = len(common) / len(ref_entities) if ref_entities else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ═══════════════════════════════════════════════════════════════════════════════
# OPCIONAIS
# ═══════════════════════════════════════════════════════════════════════════════

def ndcg_at_k(
    response: str,
    reference_keywords: Optional[list[str]] = None,
    query: Optional[str] = None,
    k: int = 5,
) -> float:
    """
    nDCG@k: grades the top-k sentences of the response by relevance
    (keyword coverage) and computes normalised discounted cumulative gain.

    Returns a float in [0, 1].
    """
    if not response or response.startswith("ERROR:"):
        return 0.0

    sentences = _split_sentences(response)

    if reference_keywords:
        keywords = set(kw.lower() for kw in reference_keywords)
    elif query:
        keywords = set(_content_tokens(query))
    else:
        return 0.0

    if not keywords:
        return 0.0

    relevances: list[float] = []
    for s in sentences[:k]:
        s_tokens = set(_tokenize(s))
        rel = len(keywords & s_tokens) / len(keywords)
        relevances.append(rel)

    # Pad with zeros if fewer than k sentences
    while len(relevances) < k:
        relevances.append(0.0)

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0.0 else 0.0


def temporal_consistency(response: str) -> float:
    """
    Heuristic temporal consistency score.

    Extracts 4-digit years from the response and checks for implausible
    chronological jumps (> 50 years between consecutive mentions).

    Returns 1.0 when no years are present or all transitions are plausible.
    Returns 0.0 when every transition is a large jump.
    """
    if not response or response.startswith("ERROR:"):
        return 0.0

    years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", response)]

    if len(years) < 2:
        return 1.0

    diffs = [abs(years[i + 1] - years[i]) for i in range(len(years) - 1)]
    large_jumps = sum(1 for d in diffs if d > 50)
    return 1.0 - (large_jumps / len(diffs))


def bertscore_f1(
    response: str, reference: str, lang: str = "pt"
) -> Optional[float]:
    """
    BERTScore F1 between response and reference.

    Requires:  pip install bert-score
    Returns None when the package is not installed or computation fails.
    """
    if not response or not reference:
        return None
    try:
        from bert_score import score as _bscore  # type: ignore

        _, _, F1 = _bscore([response], [reference], lang=lang, verbose=False)
        return float(F1[0].item())
    except ImportError:
        return None
    except Exception:
        return None


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def compute_all(
    response: str,
    query: str,
    reference: Optional[str] = None,
    reference_keywords: Optional[list[str]] = None,
    k: int = 5,
    use_bertscore: bool = False,
) -> dict[str, Optional[float]]:
    """
    Compute all metrics for a single (query, response) pair and return
    them as a dict ready to merge into an experiment result record.
    """
    bert = (
        bertscore_f1(response, reference)
        if use_bertscore and reference
        else None
    )

    return {
        "fact_score": fact_score(response, reference),
        "faithfulness": faithfulness(response, query),
        "recall_at_k": recall_at_k(response, reference_keywords, query, k),
        "radgraph_f1": radgraph_f1(response, reference, query),
        "ndcg_at_k": ndcg_at_k(response, reference_keywords, query, k),
        "temporal_consistency": temporal_consistency(response),
        "bert_score_f1": bert,
    }
