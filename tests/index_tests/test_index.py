import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'index'))

import pytest
import numpy as np
from indexer import InvertedIndex


class TestInvertedIndex:
    def test_add_and_get(self):
        idx = InvertedIndex(compression='none')
        idx.add("книга", 1)
        idx.add("книга", 2)
        idx.add("стол", 3)
        idx.finalize()
        
        assert np.array_equal(idx.get("книга"), np.array([1, 2], dtype=np.uint32))
        assert np.array_equal(idx.get("стол"), np.array([3], dtype=np.uint32))
    
    def test_duplicate_doc_ids_removed(self):
        idx = InvertedIndex(compression='none')
        idx.add("книга", 1)
        idx.add("книга", 1)
        idx.add("книга", 2)
        idx.finalize()
        
        assert np.array_equal(idx.get("книга"), np.array([1, 2], dtype=np.uint32))
    
    def test_terms_list(self):
        idx = InvertedIndex(compression='none')
        idx.add("книга", 1)
        idx.add("стол", 2)
        idx.finalize()
        
        assert set(idx.terms()) == {"книга", "стол"}
    
    def test_cannot_add_after_finalize(self):
        idx = InvertedIndex(compression='none')
        idx.add("книга", 1)
        idx.finalize()
        
        with pytest.raises(RuntimeError):
            idx.add("стол", 2)


class TestIndexCompression:
    @pytest.mark.parametrize("method", ["gamma", "delta"])
    def test_compress_preserves_data(self, method):
        original = InvertedIndex(compression='none')
        original.add("книга", 1)
        original.add("книга", 5)
        original.add("книга", 10)
        original.add("стол", 3)
        original.finalize()
        
        compressed = original.compress(method)
        
        assert np.array_equal(compressed.get("книга"), original.get("книга"))
        assert np.array_equal(compressed.get("стол"), original.get("стол"))
    
    def test_compressed_smaller(self):
        original = InvertedIndex(compression='none')
        for i in range(1, 100):
            original.add("слово", i * 100)
        original.finalize()
        
        compressed = original.compress('gamma')
        assert compressed.size_bytes() < original.size_bytes()