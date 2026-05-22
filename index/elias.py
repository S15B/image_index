import numpy as np
import numba as nb


@nb.njit(inline='always', cache=True)
def _bit_length(n: int) -> int:
    """Количество бит, необходимое для представления n (без знака)."""
    if n == 0:
        return 0
    l = 0
    while n > 0:
        n >>= 1
        l += 1
    return l

@nb.njit(inline='always', cache=True)
def _write_bits(buf: np.ndarray, bit_pos: int, value: int, num_bits: int) -> int:
    """
    Записывает младшие num_bits бит числа value в буфер начиная с позиции bit_pos.
    Возвращает новую позицию после записи.
    """
    for i in range(num_bits - 1, -1, -1):
        bit = (value >> i) & 1
        byte_idx = bit_pos >> 3
        if bit:
            buf[byte_idx] |= (1 << (bit_pos & 7))
        bit_pos += 1
    return bit_pos

@nb.njit(inline='always', cache=True)
def _read_bits(encoded: np.ndarray, bit_pos: int, num_bits: int) -> (int, int):
    """
    Читает num_bits бит из буфера начиная с bit_pos.
    Возвращает (значение, новая_позиция).
    """
    value = 0
    for i in range(num_bits - 1, -1, -1):
        byte_idx = bit_pos >> 3
        bit = (encoded[byte_idx] >> (bit_pos & 7)) & 1
        value |= (bit << i)
        bit_pos += 1
    return value, bit_pos


@nb.njit(cache=True)
def gamma_encode_list(gaps: np.ndarray) -> np.ndarray:
    """Сжимает массив gap'ов (>=1) гамма-кодом, возвращает массив байтов."""
    max_bits = len(gaps) * 64
    buf = np.zeros((max_bits + 7) // 8, dtype=np.uint8)
    bit_pos = 0

    for gap in gaps:
        if gap < 1:
            raise ValueError("Gamma code requires gap >= 1")
        m = _bit_length(gap) - 1

        bit_pos += m
        buf[bit_pos >> 3] |= (1 << (bit_pos & 7))
        bit_pos += 1
        if m > 0:
            bit_pos = _write_bits(buf, bit_pos, gap & ((1 << m) - 1), m)
    used_bytes = (bit_pos + 7) // 8
    return buf[:used_bytes]


@nb.njit(cache=True)
def gamma_decode_list(encoded: np.ndarray, size: int) -> np.ndarray:
    """Распаковывает гамма-код в массив исходных gap'ов."""
    result = np.zeros(size, dtype=np.int32)
    bit_pos = 0
    for idx in range(size):
        m = 0
        while True:
            bit = (encoded[bit_pos >> 3] >> (bit_pos & 7)) & 1
            bit_pos += 1
            if bit == 1:
                break
            m += 1
        if m == 0:
            value = 1
        else:
            low_bits, bit_pos = _read_bits(encoded, bit_pos, m)
            value = (1 << m) | low_bits
        result[idx] = value
    return result


@nb.njit(cache=True)
def delta_encode_list(gaps: np.ndarray) -> np.ndarray:
    """Сжатие дельта-кодом (использует гамма для длины)."""
    max_bits = len(gaps) * 96
    buf = np.zeros((max_bits + 7) // 8, dtype=np.uint8)
    bit_pos = 0

    for gap in gaps:
        if gap < 1:
            raise ValueError("Delta code requires gap >= 1")
        m = _bit_length(gap) - 1
        L = m + 1
        len_m = _bit_length(L) - 1
        bit_pos += len_m
        buf[bit_pos >> 3] |= (1 << (bit_pos & 7))
        bit_pos += 1
        if len_m > 0:
            bit_pos = _write_bits(buf, bit_pos, L & ((1 << len_m) - 1), len_m)
        if m > 0:
            bit_pos = _write_bits(buf, bit_pos, gap & ((1 << m) - 1), m)

    used_bytes = (bit_pos + 7) // 8
    return buf[:used_bytes]


@nb.njit(cache=True)
def delta_decode_list(encoded: np.ndarray, size: int) -> np.ndarray:
    """Распаковка дельта-кода."""
    result = np.zeros(size, dtype=np.int32)
    bit_pos = 0
    for idx in range(size):
        len_m = 0
        while True:
            bit = (encoded[bit_pos >> 3] >> (bit_pos & 7)) & 1
            bit_pos += 1
            if bit == 1:
                break
            len_m += 1
        if len_m == 0:
            L = 1
        else:
            low_bits, bit_pos = _read_bits(encoded, bit_pos, len_m)
            L = (1 << len_m) | low_bits
        m = L - 1
        if m == 0:
            value = 1
        else:
            low_bits, bit_pos = _read_bits(encoded, bit_pos, m)
            value = (1 << m) | low_bits
        result[idx] = value
    return result


def gamma_encode_single(n: int) -> bytes:
    arr = np.array([n], dtype=np.int32)
    return gamma_encode_list(arr).tobytes()

def gamma_decode_single(data: bytes) -> int:
    arr = np.frombuffer(data, dtype=np.uint8)
    return gamma_decode_list(arr, 1)[0]
