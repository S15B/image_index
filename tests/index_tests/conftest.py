import pytest
import tempfile
import json
import os


@pytest.fixture
def temp_index_dir():
    """Временная директория для тестов"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def real_documents():
    """Реальные документы (если есть)"""
    path = "data/unified_documents.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        return docs[:100]  # первые 100 для быстрых тестов
    return None