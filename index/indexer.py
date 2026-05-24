from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Literal

import numpy as np

from index.elias import gamma_encode_list, gamma_decode_list, delta_encode_list, delta_decode_list


class PostingsList(ABC):
    """Базовый класс для списка документов."""
    @abstractmethod
    def doc_ids(self) -> np.ndarray:
        pass

    @abstractmethod
    def size_bytes(self) -> int:
        pass


class UncompressedPostingsList(PostingsList):
    """Несжатый список doc_id."""
    def __init__(self):
        self._list: Optional[np.ndarray] = None

    def add(self, doc_id: int):
        """Добавляет doc_id во временный Python-список (до финализации)."""
        if not hasattr(self, '_tmp_list'):
            self._tmp_list = []
        self._tmp_list.append(doc_id)

    def finalize(self):
        """Сортирует, удаляет дубликаты и преобразует в np.array."""
        if hasattr(self, '_tmp_list') and self._tmp_list:
            self._list = np.unique(np.array(self._tmp_list, dtype=np.uint32))
            del self._tmp_list
        else:
            self._list = np.array([], dtype=np.uint32)

    def doc_ids(self) -> np.ndarray:
        return self._list

    def size_bytes(self) -> int:
        return self._list.nbytes


class CompressedPostingsList(PostingsList):
    """Сжатый (гамма или дельта) список doc_id."""
    def __init__(self, doc_ids: np.ndarray, method: Literal['gamma', 'delta']):
        """
        doc_ids: отсортированный массив уникальных doc_id (np.uint32).
        method: 'gamma' или 'delta'.
        """
        if len(doc_ids) == 0:
            self._size = 0
            self._encoded = np.array([], dtype=np.uint8)
            self._method = method
            return

        gaps = np.empty_like(doc_ids)
        gaps[0] = doc_ids[0]
        if len(doc_ids) > 1:
            gaps[1:] = np.diff(doc_ids)

        if method == 'gamma':
            self._encoded = gamma_encode_list(gaps)
        elif method == 'delta':
            self._encoded = delta_encode_list(gaps)
        else:
            raise ValueError(f"Unknown compression method: {method}")
        self._size = len(doc_ids)
        self._method = method

    def doc_ids(self) -> np.ndarray:
        if self._size == 0:
            return np.array([], dtype=np.uint32)

        if self._method == 'gamma':
            gaps = gamma_decode_list(self._encoded, self._size)
        else:
            gaps = delta_decode_list(self._encoded, self._size)

        return np.cumsum(gaps)

    def size_bytes(self) -> int:
        return self._encoded.nbytes


class InvertedIndex:
    def __init__(self, compression: Literal['none', 'gamma', 'delta'] = 'none'):
        """
        compression: 'none', 'gamma', 'delta'.
        Если 'none', списки хранятся как UncompressedPostingsList,
        иначе – как CompressedPostingsList (после вызова finalize()).
        """
        self.compression = compression
        self._index: Dict[str, PostingsList] = {}
        self._finalized = False

    def add(self, term: str, doc_id: int):
        """Добавляет doc_id для заданного терма."""
        if self._finalized:
            raise RuntimeError("Cannot add to a finalized index.")
        if term not in self._index:
            self._index[term] = UncompressedPostingsList()
        self._index[term].add(doc_id)

    def finalize(self):
        """
        Завершает построение индекса:
        – сортирует и удаляет дубликаты во всех списках;
        – если compression != 'none', сразу создаёт сжатое представление.
        """
        if self._finalized:
            return
        for term, pl in self._index.items():
            pl.finalize()
            if self.compression != 'none':
                self._index[term] = CompressedPostingsList(pl.doc_ids(), self.compression)
        self._finalized = True

    def compress(self, method: Literal['gamma', 'delta']) -> 'InvertedIndex':
        """
        Создаёт новый индекс, в котором все списки сжаты указанным методом.
        Исходный индекс не изменяется.
        """
        new_idx = InvertedIndex(compression=method)
        new_idx._index = {}
        for term, pl in self._index.items():
            doc_ids = pl.doc_ids()
            new_idx._index[term] = CompressedPostingsList(doc_ids, method)
        new_idx._finalized = True
        return new_idx

    def get(self, term: str) -> np.ndarray:
        """Возвращает массив doc_id для терма (или пустой массив)."""
        pl = self._index.get(term)
        if pl is None:
            return np.array([], dtype=np.uint32)
        return pl.doc_ids()

    def terms(self) -> List[str]:
        return list(self._index.keys())

    def size_bytes(self) -> int:
        """Суммарный размер всех списков в байтах."""
        return sum(pl.size_bytes() for pl in self._index.values())
