import numpy as np
from typing import List
from indexer import InvertedIndex
from tokenizer import Tokenizer


def tokenize_query(query: str, tokenizer: Tokenizer) -> List[str]:
    if not query:
        return []
    return tokenizer.tokenize(query)

def boolean_and(index: InvertedIndex, query_tokens: List[str]) -> np.ndarray:
    if not query_tokens:
        return np.array([], dtype=np.uint32)

    postings = []
    for token in query_tokens:
        doc_ids = index.get(token)
        if len(doc_ids) == 0:
            return np.array([], dtype=np.uint32)
        postings.append(doc_ids)

    # Сортируем по длине, чтобы минимизировать количество операций пересечения
    postings.sort(key=len)

    result = postings[0]
    for other in postings[1:]:
        result = np.intersect1d(result, other, assume_unique=True)
        if len(result) == 0:
            break

    return result

def search(index: InvertedIndex, query_text: str, tokenizer: Tokenizer) -> np.ndarray:
    tokens = tokenize_query(query_text, tokenizer)
    return boolean_and(index, tokens)
