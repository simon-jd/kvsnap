"""Incremental Castagnoli CRC32C, native accelerated when available."""
from .native_io import get_io_backend


def _table():
    result = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        result.append(value)
    return tuple(result)


_TABLE = _table()


def crc32c(data: bytes, crc: int = 0) -> int:
    """Return CRC32C; passing a prior CRC permits incremental checksumming."""
    if type(crc) is not int or not 0 <= crc <= 0xFFFFFFFF:
        raise ValueError("prior CRC32C must fit uint32")
    backend = get_io_backend()
    if backend.name == "native":
        return backend.crc32c(data, crc)
    value = crc ^ 0xFFFFFFFF
    for byte in data:
        value = _TABLE[(value ^ byte) & 0xFF] ^ (value >> 8)
    return value ^ 0xFFFFFFFF
