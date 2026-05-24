def deduplicate_by_phash(results):
    seen = set()
    unique = []
    for item in results:
        phash = item.get('phash')
        if phash is not None:
            if phash not in seen:
                seen.add(phash)
                unique.append(item)
        else:
            unique.append(item)
    return unique


def deduplicate_by_url(results):
    """Удаляет дубликаты по source_url."""
    seen = set()
    unique = []
    for item in results:
        url = item.get('source_url')
        if url is not None:
            if url not in seen:
                seen.add(url)
                unique.append(item)
        else:
            unique.append(item)
    return unique


def deduplicate(results, by_phash=True, by_url=True):
    if by_phash:
        results = deduplicate_by_phash(results)
    if by_url:
        results = deduplicate_by_url(results)
    return results
