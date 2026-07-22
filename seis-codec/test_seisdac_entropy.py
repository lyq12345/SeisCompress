import unittest

import numpy as np
import zstandard as zstd

from evaluate_seisdac_entropy import (
    STREAM_HEADER,
    decode_code_stream,
    encode_code_stream,
    pack_unsigned,
    unpack_unsigned,
)


class BitPackingTest(unittest.TestCase):
    def test_round_trip_non_byte_aligned(self):
        values = np.array([0, 1, 7, 3, 5], dtype=np.int64)
        payload = pack_unsigned(values, bits=3)
        self.assertEqual(len(payload), 2)
        np.testing.assert_array_equal(unpack_unsigned(payload, bits=3, count=len(values)), values)

    def test_rejects_out_of_range_code(self):
        with self.assertRaises(ValueError):
            pack_unsigned(np.array([1024]), bits=10)


class CodeStreamTest(unittest.TestCase):
    def test_all_stream_codings_are_lossless(self):
        rng = np.random.default_rng(17)
        codes = rng.integers(0, 1024, size=(9, 376), dtype=np.int64)
        compressor = zstd.ZstdCompressor(level=3, write_checksum=True)
        decompressor = zstd.ZstdDecompressor()
        for coding in ("packed10", "zstd-packed10", "zstd-uint16"):
            with self.subTest(coding=coding):
                stream = encode_code_stream(
                    codes,
                    original_length=3001,
                    bits_per_code=10,
                    coding=coding,
                    compressor=compressor,
                )
                self.assertGreater(len(stream), STREAM_HEADER.size)
                decoded, original_length = decode_code_stream(stream, decompressor=decompressor)
                self.assertEqual(original_length, 3001)
                np.testing.assert_array_equal(decoded, codes)

    def test_rejects_bad_magic(self):
        rng = np.random.default_rng(3)
        codes = rng.integers(0, 8, size=(2, 5), dtype=np.int64)
        compressor = zstd.ZstdCompressor(level=1)
        stream = encode_code_stream(
            codes,
            original_length=41,
            bits_per_code=3,
            coding="packed10",
            compressor=compressor,
        )
        damaged = b"BAD!" + stream[4:]
        with self.assertRaises(ValueError):
            decode_code_stream(damaged, decompressor=zstd.ZstdDecompressor())


if __name__ == "__main__":
    unittest.main()
