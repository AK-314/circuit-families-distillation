"""Portable deterministic compression and atomic streaming helpers."""

from __future__ import annotations

import os
import secrets
import struct
import zlib
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

from .records import CodecProfile, Stage12P4Error

_GZIP_HEADER = bytes.fromhex("1f8b08000000000000ff")


class DeterministicGzipWriter:
    """A gzip writer with fixed header fields and no host/path leakage."""

    def __init__(self, handle: BinaryIO, *, level: int) -> None:
        self._handle = handle
        self._compressor = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
        self._crc = 0
        self._size = 0
        self._closed = False
        handle.write(_GZIP_HEADER)

    def write(self, data: bytes) -> None:
        if self._closed:
            raise Stage12P4Error("cannot write a finalized gzip stream")
        if not isinstance(data, bytes):
            raise Stage12P4Error("codec input must be bytes")
        self._crc = zlib.crc32(data, self._crc)
        self._size = (self._size + len(data)) & 0xFFFFFFFF
        self._handle.write(self._compressor.compress(data))

    def close(self) -> None:
        if self._closed:
            return
        self._handle.write(self._compressor.flush(zlib.Z_FINISH))
        self._handle.write(struct.pack("<II", self._crc & 0xFFFFFFFF, self._size))
        self._closed = True


def encode_chunks(chunks: Iterable[bytes], profile: CodecProfile, handle: BinaryIO) -> None:
    if profile.codec == "none":
        for chunk in chunks:
            handle.write(chunk)
        return
    writer = DeterministicGzipWriter(handle, level=int(profile.compression_level))
    for chunk in chunks:
        writer.write(chunk)
    writer.close()


def decode_bytes(data: bytes, profile: CodecProfile) -> bytes:
    if profile.codec == "none":
        return data
    try:
        return zlib.decompress(data, wbits=31)
    except zlib.error as exc:
        raise Stage12P4Error("compressed object is corrupt or truncated") from exc


def atomic_encode(path: Path, chunks: Iterable[bytes], profile: CodecProfile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            encode_chunks(chunks, profile, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_bytes() != temporary.read_bytes():
                raise Stage12P4Error("refusing to overwrite a conflicting compact object")
            temporary.unlink()
        else:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
