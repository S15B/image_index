import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'index'))

import pytest
import numpy as np
from indexer import InvertedIndex
from tokenizer import Tokenizer
from query_processor import search


class TestSearch:
    
    def test_search_returns_results(self):
        idx = InvertedIndex(compression='none')
        # Используем стеммированную форму (как в реальном индексе)
        idx.add("книг", 1)   # ← "книг" вместо "книга"
        idx.add("книг", 2)
        idx.add("стол", 3)
        idx.finalize()

        tokenizer = Tokenizer()
        results = search(idx, "книга", tokenizer)  # запрос "книга" превратится в "книг"

        assert isinstance(results, np.ndarray)
        assert len(results) > 0
    
    def test_search_correct_documents(self):
        idx = InvertedIndex(compression='none')
        # Используем стеммированную форму
        idx.add("книг", 1)
        idx.add("книг", 2)
        idx.add("стол", 3)
        idx.finalize()

        tokenizer = Tokenizer()
        results = search(idx, "книга", tokenizer)
        
        result_ids = results.tolist() if isinstance(results, np.ndarray) else results

        assert 1 in result_ids
        assert 2 in result_ids
        assert 3 not in result_ids
    
    def test_empty_query_returns_empty(self):
        idx = InvertedIndex(compression='none')
        idx.add("книг", 1)
        idx.finalize()

        tokenizer = Tokenizer()
        results = search(idx, "", tokenizer)

        assert isinstance(results, np.ndarray)
        assert results.size == 0
    
    def test_search_nonexistent_word(self):
        idx = InvertedIndex(compression='none')
        idx.add("книг", 1)
        idx.finalize()

        tokenizer = Tokenizer()
        results = search(idx, "ывапывапвыапавып", tokenizer)

        assert isinstance(results, np.ndarray)
        assert results.size == 0
    
    def test_search_returns_unique_doc_ids(self):
        idx = InvertedIndex(compression='none')
        idx.add("книг", 1)
        idx.add("книг", 1)  # дубликат
        idx.add("книг", 2)
        idx.finalize()

        tokenizer = Tokenizer()
        results = search(idx, "книга", tokenizer)

        if results.size > 0:
            unique_ids = np.unique(results)
            assert len(unique_ids) == len(results)