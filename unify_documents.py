import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

from deep_translator import GoogleTranslator

logger = logging.getLogger("prepare_data")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
console_handler.setFormatter(console_format)

file_handler = logging.FileHandler("logs/prepare_data.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_format)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

SRC_LANG = "en"
TGT_LANG = "ru"
SLEEP_BETWEEN_REQUESTS = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def translate_with_retry(text: str) -> Optional[str]:
    """Перевести текст с повторными попытками."""
    translator = GoogleTranslator(source=SRC_LANG, target=TGT_LANG)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = translator.translate(text)
            if result:
                return result
        except Exception as e:
            logger.warning("Попытка %d/%d не удалась: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    logger.error("Не удалось перевести: \"%s...\"", text[:70])
    return None


def load_json(path: str) -> Any:
    """Загрузить JSON-файл."""
    logger.debug("Загрузка %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Сохранить объект в JSON-файл."""
    logger.info("Сохранение %s", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_context(record: Dict[str, str]) -> str:
    """Собрать контекст из полей results.json (русский текст)."""
    parts = []
    for field in ("heading", "context", "alt", "text_before", "text_after", "block_text"):
        val = record.get(field, "").strip()
        if val:
            parts.append(val)
    return " ".join(parts)


def normalize_image_path(absolute_path: str, base_dir: str) -> str:
    """Преобразовать абсолютный путь в относительный вида 'extracted_images/page_X/image_Y.jpg'."""
    abs_path = os.path.normpath(absolute_path)
    base = os.path.normpath(base_dir)
    rel = os.path.relpath(abs_path, base)
    rel = rel.replace("\\", "/")
    return rel


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовка единого JSON с метаданными и переводом")
    parser.add_argument(
        "--descriptions",
        default="data/descriptions_index.json",
        help="Путь к файлу описаний нейросети (JSON с ключами model, completed)",
    )
    parser.add_argument(
        "--output",
        default="data/unified_documents.json",
        help="Путь к выходному JSON-файлу со всеми документами",
    )
    parser.add_argument(
        "--cache",
        default="data/translation_cache.json",
        help="Путь к файлу кэша переводов",
    )
    parser.add_argument(
        "--results-dir",
        default="data/extracted_images",
        help="Папка с подпапками page_*, содержащими results.json",
    )
    args = parser.parse_args()

    translation_cache: Dict[str, str] = {}
    if os.path.exists(args.cache):
        translation_cache = load_json(args.cache)
        logger.info("Загружен кэш переводов: %d записей", len(translation_cache))
    else:
        logger.info("Файл кэша переводов не найден, будет создан новый")

    existing_docs: List[Dict[str, Any]] = []
    existing_paths: Set[str] = set()
    max_id: int = 0
    if os.path.exists(args.output):
        existing_docs = load_json(args.output)
        for doc in existing_docs:
            existing_paths.add(doc["image_path"])
            if doc["id"] > max_id:
                max_id = doc["id"]
        logger.info("Загружено %d существующих документов, max_id=%d", len(existing_docs), max_id)

    logger.info("Загрузка описаний из %s", args.descriptions)
    desc_data = load_json(args.descriptions)

    if "completed" not in desc_data:
        logger.error("Ожидаемый ключ 'completed' отсутствует в файле описаний")
        sys.exit(1)

    completed_list = desc_data["completed"]
    logger.info("Получено %d записей из completed", len(completed_list))

    results_dir = args.results_dir
    descriptions_by_path: Dict[str, Dict[str, str]] = {}
    for idx, (img_path, item) in enumerate(completed_list.items()):
        img_path = img_path.replace("\\", "/")
        if item.get("is_logo", False):
            continue

        if img_path.startswith("extracted_images/"):
            img_path = img_path[len("extracted_images/"):]
        descriptions_by_path[img_path] = {
            "phash": item.get("phash", ""),
            "description_en": item.get("description", ""),
        }

    logger.info("Уникальных путей с описаниями: %d", len(descriptions_by_path))

    if not os.path.isdir(results_dir):
        logger.error("Папка с результатами не найдена: %s", results_dir)
        sys.exit(1)

    contexts_by_path: Dict[str, Dict[str, str]] = {}
    total_pages = 0
    total_images = 0

    for page_entry in os.scandir(results_dir):
        if not page_entry.is_dir() or not page_entry.name.startswith("page_"):
            continue
        page_dir = page_entry.name
        results_file = os.path.join(results_dir, page_dir, "results.json")
        if not os.path.isfile(results_file):
            logger.warning("Не найден results.json в %s", page_dir)
            continue

        try:
            page_data = load_json(results_file)
        except Exception as e:
            logger.error("Ошибка загрузки %s: %s", results_file, e)
            continue

        total_pages += 1
        if not isinstance(page_data, list):
            logger.warning("results.json в %s не является списком, пропущено", page_dir)
            continue

        for img_record in page_data:
            abs_path = img_record.get("image_path")
            if not abs_path:
                continue
            rel_path = normalize_image_path(abs_path, results_dir)
            context = build_context(img_record)
            source_url = img_record.get("page_url", "")
            contexts_by_path[rel_path] = {
                "context": context,
                "source_url": source_url,
            }
            total_images += 1

    logger.info("Обработано страниц: %d, изображений: %d", total_pages, total_images)

    common_paths = set(descriptions_by_path.keys()) & set(contexts_by_path.keys())
    logger.info(
        "Пересечение (описание+контекст): %d из %d описаний и %d контекстов",
        len(common_paths),
        len(descriptions_by_path),
        len(contexts_by_path),
    )

    new_paths = common_paths - existing_paths
    logger.info("Новых изображений для добавления: %d", len(new_paths))

    if not new_paths:
        logger.info("Нет новых записей, выходной файл актуален.")
        return

    new_documents: List[Dict[str, Any]] = []
    new_translations = 0
    cache_hits = 0

    for img_path in sorted(new_paths):
        en_text = descriptions_by_path[img_path]["description_en"]

        if not en_text.strip():
            ru_text = ""
        else:
            if en_text in translation_cache:
                ru_text = translation_cache[en_text]
                cache_hits += 1
            else:
                logger.info("Перевод: \"%s...\"", en_text[:70])
                ru_text = translate_with_retry(en_text)
                if ru_text is None:
                    logger.warning("Не удалось перевести, пропускаем %s", img_path)
                    continue
                translation_cache[en_text] = ru_text
                new_translations += 1

                time.sleep(SLEEP_BETWEEN_REQUESTS)

        max_id += 1
        new_documents.append(
            {
                "id": max_id,
                "image_path": img_path,
                "phash": descriptions_by_path[img_path]["phash"],
                "source_url": contexts_by_path[img_path]["source_url"],
                "context": contexts_by_path[img_path]["context"],
                "description_en": en_text,
                "description_ru": ru_text,
            }
        )

    all_documents = existing_docs + new_documents
    save_json(all_documents, args.output)
    save_json(translation_cache, args.cache)

    logger.info("=" * 50)
    logger.info("Итоговый размер выходного файла: %d документов", len(all_documents))
    logger.info("Новых переводов через API: %d", new_translations)
    logger.info("Взято из кэша: %d", cache_hits)
    logger.info("Готово.")


if __name__ == "__main__":
    main()
