import re
from typing import List, Set
from nltk.stem.snowball import RussianStemmer, EnglishStemmer


class Tokenizer:
    """
    Токенизатор русско-английских текстов.
    Язык определяется по наличию кириллицы, иначе считается английским.
    Стоп-слова из NLTK подгружаются при инициализации; если корпуса нет,
    стоп-слова игнорируются.
    """
    def __init__(self, stop_ru: Set[str] = None, stop_en: Set[str] = None):
        self._stem_ru = RussianStemmer()
        self._stem_en = EnglishStemmer()
        self._stop_ru = stop_ru if stop_ru is not None else self._load_stop('russian')
        self._stop_en = stop_en if stop_en is not None else self._load_stop('english')

    @staticmethod
    def _load_stop(lang: str) -> Set[str]:
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words(lang))
        except (ImportError, LookupError):
            return set()

    def tokenize(self, text: str, lang: str = 'auto') -> List[str]:
        if not text:
            return []
        text = text.lower()
        if lang == 'auto':
            lang = 'ru' if self._has_cyrillic(text) else 'en'

        if lang == 'ru':
            tokens = self._extract_words_ru(text)
            stop_set = self._stop_ru
            stemmer = self._stem_ru
        else:
            tokens = self._extract_words_en(text)
            stop_set = self._stop_en
            stemmer = self._stem_en

        tokens = [t for t in tokens if t not in stop_set]
        return [stemmer.stem(t) for t in tokens]

    @staticmethod
    def _has_cyrillic(text: str) -> bool:
        return any('а' <= ch <= 'я' or ch == 'ё' for ch in text)

    @staticmethod
    def _extract_words_ru(text: str) -> List[str]:
        return re.findall(r'[а-яё]+', text)

    @staticmethod
    def _extract_words_en(text: str) -> List[str]:
        return re.findall(r'[a-z]+', text)
