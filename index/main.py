import json
import os
import time
import argparse
import pickle
import logging
import timeit
from typing import Dict, List, Literal

from tokenizer import Tokenizer
from indexer import InvertedIndex
from query_processor import search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_documents(filepath: str) -> List[dict]:
    """
    Загружает документы из unified_documents.json.
    Каждый документ – словарь с ключами:
      id, image_path, phash, source_url, context, description_en, description_ru
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    logger.info(f"Загружено документов: {len(documents)}")
    return documents


def build_index(documents: List[dict], tokenizer, text_fields: List[str],
                compression: Literal['none', 'gamma', 'delta'] = 'none') -> InvertedIndex:
    index = InvertedIndex(compression)
    for doc in documents:
        doc_id = doc['id']
        full_text = ' '.join(doc.get(field, '') for field in text_fields)
        tokens = tokenizer.tokenize(full_text)
        for token in tokens:
            index.add(token, doc_id)
    index.finalize()
    return index


def build_indexes(documents: List[dict], tokenizer: Tokenizer) -> Dict[str, InvertedIndex]:
    """
    Строит три несжатых индекса:
      - full        : context + description_ru
      - context     : только context
      - description : только description_ru
    Возвращает словарь с ключами 'full', 'context', 'description'.
    """
    logger.info("Строим индексы...")
    t0 = time.time()

    indexes = {
        'full':        build_index(documents, tokenizer, ['context', 'description_ru'], compression='none'),
        'context':     build_index(documents, tokenizer, ['context'], compression='none'),
        'description': build_index(documents, tokenizer, ['description_ru'], compression='none'),
    }

    logger.info(f"Индексы построены за {time.time() - t0:.3f} сек.")
    return indexes


def compress_all(indexes: Dict[str, InvertedIndex]) -> Dict[str, Dict[str, InvertedIndex]]:
    logger.info("Создаём сжатые индексы (гамма, дельта)...")
    compressed = {}
    for field, idx in indexes.items():
        t0 = time.time()
        idx_gamma = idx.compress('gamma')
        idx_delta = idx.compress('delta')
        compressed[field] = {'gamma': idx_gamma, 'delta': idx_delta}
        logger.info(f"  {field}: гамма-сжатие {time.time()-t0:.3f}с, дельта-сжатие выполнено.")
    return compressed


def measure_sizes(indexes: Dict[str, InvertedIndex],
                  compressed: Dict[str, Dict[str, InvertedIndex]]) -> Dict[str, dict]:
    sizes = {}
    for field in ('full', 'context', 'description'):
        sizes[field] = {
            'none': indexes[field].size_bytes(),
            'gamma': compressed[field]['gamma'].size_bytes(),
            'delta': compressed[field]['delta'].size_bytes()
        }
    return sizes


def benchmark_search(indexes: Dict[str, InvertedIndex],
                     compressed: Dict[str, Dict[str, InvertedIndex]],
                     tokenizer: Tokenizer,
                     queries: Dict[str, str],
                     n_runs: int = 10) -> Dict[str, dict]:
    all_indexes = {}
    for field in ('full', 'context', 'description'):
        all_indexes[(field, 'none')] = indexes[field]
        all_indexes[(field, 'gamma')] = compressed[field]['gamma']
        all_indexes[(field, 'delta')] = compressed[field]['delta']

    results = {}
    for qname, qtext in queries.items():
        tokens = tokenizer.tokenize(qtext)
        if not tokens:
            logger.warning(f"  запрос '{qname}' не дал токенов, пропускаем")
            continue

        results[qname] = {}
        for (field, method), idx in all_indexes.items():
            timer = timeit.Timer(lambda: search(idx, qtext, tokenizer))
            times = timer.repeat(repeat=n_runs, number=1)
            elapsed = sum(times) / len(times)

            res = search(idx, qtext, tokenizer)
            results[qname][f"{field}_{method}"] = {
                'time_ms': elapsed * 1000,
                'num_results': len(res)
            }
    return results


def save_indexes(indexes: Dict[str, InvertedIndex],
                 compressed: Dict[str, Dict[str, InvertedIndex]],
                 tokenizer: Tokenizer,
                 output_dir: str = "data/index"):
    """
    Сохраняет все индексы и токенизатор в указанную директорию.
    """
    os.makedirs(output_dir, exist_ok=True)
    for field, idx in indexes.items():
        with open(os.path.join(output_dir, f"index_{field}_none.pkl"), 'wb') as f:
            pickle.dump(idx, f)
    for field, comp_dict in compressed.items():
        for method in ('gamma', 'delta'):
            with open(os.path.join(output_dir, f"index_{field}_{method}.pkl"), 'wb') as f:
                pickle.dump(comp_dict[method], f)
    with open(os.path.join(output_dir, "tokenizer.pkl"), 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Индексы сохранены в {output_dir}")


def load_indexes(input_dir: str = "data/index") -> tuple:
    with open(os.path.join(input_dir, "tokenizer.pkl"), 'rb') as f:
        tokenizer = pickle.load(f)

    indexes = {}
    compressed = {}
    for field in ('full', 'context', 'description'):
        with open(os.path.join(input_dir, f"index_{field}_none.pkl"), 'rb') as f:
            indexes[field] = pickle.load(f)
        compressed[field] = {}
        for method in ('gamma', 'delta'):
            with open(os.path.join(input_dir, f"index_{field}_{method}.pkl"), 'rb') as f:
                compressed[field][method] = pickle.load(f)

    logger.info(f"Индексы загружены из {input_dir}")
    return indexes, compressed, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Построение и тестирование инвертированных индексов')
    parser.add_argument('command', nargs='?', default='all',
                        choices=['all', 'build', 'test'],
                        help='Действие: all (построить+тестировать), build (только построить и сохранить), test (загрузить и тестировать)')
    parser.add_argument('--data', default='data/unified_documents.json',
                        help='Путь к unified_documents.json')
    parser.add_argument('--index-dir', default='data/index',
                        help='Директория для сохранения/загрузки индексов')
    args = parser.parse_args()

    logger.info("Инициализация токенизатора...")
    tokenizer = Tokenizer()

    if args.command in ('all', 'build'):
        docs = load_documents(args.data)
        indexes = build_indexes(docs, tokenizer)
        compressed = compress_all(indexes)
        save_indexes(indexes, compressed, tokenizer, args.index_dir)

        if args.command == 'build':
            return

    if args.command == 'test':
        indexes, compressed, tokenizer = load_indexes(args.index_dir)

        test_queries = {
            "мужчина с книгой": "мужчина читает книгу",
            "блокчейн технологии": "блокчейн технологии",
            "нож": "нож",
            "пустой запрос": "",
        }

        sizes = measure_sizes(indexes, compressed)
        logger.info("\n" + "="*60)
        logger.info("Размеры индексов (байты):")
        for field in ('full', 'context', 'description'):
            logger.info(f"  {field}:")
            for method in ('none', 'gamma', 'delta'):
                logger.info(f"    {method:6s}: {sizes[field][method]:>10d}")

        logger.info("\n" + "="*60)
        logger.info("Тестирование скорости поиска (среднее время в мс, 100 повторов):")
        bench = benchmark_search(indexes, compressed, tokenizer, test_queries, 100)
        for qname, metrics in bench.items():
            logger.info(f"Запрос: '{qname}'")
            for combo, data in metrics.items():
                logger.info(f"    {combo:30s} время: {data['time_ms']:6.3f} мс, найдено: {data['num_results']:4d}")


if __name__ == '__main__':
    main()
