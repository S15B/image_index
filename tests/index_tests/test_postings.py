import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'index'))

import pytest
import numpy as np
from indexer import UncompressedPostingsList, CompressedPostingsList


class TestUncompressedPostingsList:
    def test_add_and_finalize(self):
        pl = UncompressedPostingsList()
        pl.add(5)
        pl.add(3)
        pl.add(5)
        pl.add(7)
        pl.finalize()
        
        doc_ids = pl.doc_ids()
        expected = np.array([3, 5, 7], dtype=np.uint32)
        assert np.array_equal(doc_ids, expected)
    
    def test_empty_list(self):
        pl = UncompressedPostingsList()
        pl.finalize()
        assert len(pl.doc_ids()) == 0
        assert pl.size_bytes() == 0


class TestCompressedPostingsList:
    @pytest.mark.parametrize("method", ["gamma", "delta"])
    def test_compression_roundtrip(self, method):
        original = np.array([1, 3, 7, 15, 31], dtype=np.uint32)
        compressed = CompressedPostingsList(original, method)
        decompressed = compressed.doc_ids()
        assert np.array_equal(original, decompressed)
    
    @pytest.mark.parametrize("method", ["gamma", "delta"])
    def test_compressed_smaller(self, method):
        doc_ids = np.array([1, 2, 3, 1000, 1001, 1002], dtype=np.uint32)
        compressed = CompressedPostingsList(doc_ids, method)
        assert compressed.size_bytes() < doc_ids.nbytes