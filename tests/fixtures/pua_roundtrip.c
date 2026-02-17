/*
 * pua_roundtrip.c — PUA (Private Use Area) Encoding Round-Trip Test Fixture
 *
 * Checkpoint 2 validation: Non-UTF-8 bytes (0x80–0xFF) must survive the
 * entire BCC compilation pipeline with byte-exact fidelity.
 *
 * The BCC compiler uses PUA code point mapping (U+E080–U+E0FF) to encode
 * non-UTF-8 bytes on source file read and decode them back to raw bytes
 * during code generation output. This test validates that mapping by
 * embedding hex escape sequences in string literals and verifying that the
 * exact bytes appear in the compiled binary's .rodata section and at runtime.
 *
 * Per Section 0.7.9: The Linux kernel contains binary data in string literals
 * and inline assembly operands; encoding failure corrupts generated machine
 * code. Validation protocol: compile this source, inspect .rodata with
 * `objdump -s`, and confirm exact bytes are present.
 *
 * Usage:
 *   ./bcc -o pua_roundtrip tests/fixtures/pua_roundtrip.c
 *   ./pua_roundtrip
 *   echo $?  # must be 0
 */

#include <stdio.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Test Case 1: Core PUA round-trip — bytes 0x80 and 0xFF
 * Per Section 0.7.9: compile source with \x80\xFF in string literal,
 * inspect .rodata with objdump -s, confirm exact bytes 80 ff.
 * -------------------------------------------------------------------------- */
static const char pua_core[] = "\x80\xFF";

/* --------------------------------------------------------------------------
 * Test Case 2: Full range sampling — representative bytes from 0x80 to 0xFF
 * Covers the first non-ASCII byte (0x80), mid-range values, and the maximum
 * byte value (0xFF) to exercise the entire PUA mapping range.
 * -------------------------------------------------------------------------- */
static const char pua_range[] = "\x80\x90\xA0\xB0\xC0\xD0\xE0\xF0\xFF";

/* --------------------------------------------------------------------------
 * Test Case 3: Mixed ASCII and non-UTF-8 bytes
 * Validates that PUA encoding does not corrupt adjacent ASCII characters
 * when non-UTF-8 bytes are interleaved with printable text.
 * -------------------------------------------------------------------------- */
static const char pua_mixed[] = "hello\x80world\xFF";

/* --------------------------------------------------------------------------
 * Test Case 4: All 128 non-ASCII bytes (0x80 through 0xFF)
 * Exhaustive coverage of the entire PUA mapping range to ensure no single
 * byte value is dropped, substituted, or corrupted during compilation.
 * -------------------------------------------------------------------------- */
static const char pua_all[128] = {
    '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87',
    '\x88', '\x89', '\x8A', '\x8B', '\x8C', '\x8D', '\x8E', '\x8F',
    '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97',
    '\x98', '\x99', '\x9A', '\x9B', '\x9C', '\x9D', '\x9E', '\x9F',
    '\xA0', '\xA1', '\xA2', '\xA3', '\xA4', '\xA5', '\xA6', '\xA7',
    '\xA8', '\xA9', '\xAA', '\xAB', '\xAC', '\xAD', '\xAE', '\xAF',
    '\xB0', '\xB1', '\xB2', '\xB3', '\xB4', '\xB5', '\xB6', '\xB7',
    '\xB8', '\xB9', '\xBA', '\xBB', '\xBC', '\xBD', '\xBE', '\xBF',
    '\xC0', '\xC1', '\xC2', '\xC3', '\xC4', '\xC5', '\xC6', '\xC7',
    '\xC8', '\xC9', '\xCA', '\xCB', '\xCC', '\xCD', '\xCE', '\xCF',
    '\xD0', '\xD1', '\xD2', '\xD3', '\xD4', '\xD5', '\xD6', '\xD7',
    '\xD8', '\xD9', '\xDA', '\xDB', '\xDC', '\xDD', '\xDE', '\xDF',
    '\xE0', '\xE1', '\xE2', '\xE3', '\xE4', '\xE5', '\xE6', '\xE7',
    '\xE8', '\xE9', '\xEA', '\xEB', '\xEC', '\xED', '\xEE', '\xEF',
    '\xF0', '\xF1', '\xF2', '\xF3', '\xF4', '\xF5', '\xF6', '\xF7',
    '\xF8', '\xF9', '\xFA', '\xFB', '\xFC', '\xFD', '\xFE', '\xFF'
};

/* --------------------------------------------------------------------------
 * Test Case 5: Boundary bytes in a string literal context
 * Specifically targets the two critical boundary values:
 *   0x80 — first byte above ASCII range (first PUA-mapped byte)
 *   0xFF — maximum byte value (last PUA-mapped byte)
 * Embedded within a string with a null terminator to test string handling.
 * -------------------------------------------------------------------------- */
static const char pua_boundary[] = "\x80\xFF\x00";

/* --------------------------------------------------------------------------
 * Test Case 6: Consecutive identical non-ASCII bytes
 * Verifies that repeated identical high bytes are not collapsed or deduplicated
 * by any stage of the pipeline.
 * -------------------------------------------------------------------------- */
static const char pua_repeated[] = "\xFF\xFF\xFF\x80\x80\x80";

/* --------------------------------------------------------------------------
 * Test Case 7: Non-ASCII bytes adjacent to escape sequences
 * Tests interaction between standard C escape sequences and PUA-encoded bytes.
 * -------------------------------------------------------------------------- */
static const char pua_with_escapes[] = "\n\x80\t\xFF\\\x90\0\xA0";

/* --------------------------------------------------------------------------
 * Helper: verify_bytes — compare a memory region against expected byte values
 * Returns 0 on match, 1 on mismatch (with diagnostic output).
 * -------------------------------------------------------------------------- */
static int verify_bytes(const char *name,
                        const unsigned char *actual,
                        const unsigned char *expected,
                        int length)
{
    int i;
    for (i = 0; i < length; i++) {
        if (actual[i] != expected[i]) {
            printf(
                    "FAIL [%s]: byte %d: expected 0x%02X, got 0x%02X\n",
                    name, i, expected[i], actual[i]);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    int failures = 0;

    /* ------------------------------------------------------------------
     * Test Case 1: Core PUA round-trip — \x80\xFF
     * Per Section 0.7.9 validation protocol.
     * ------------------------------------------------------------------ */
    {
        static const unsigned char expected[] = { 0x80, 0xFF };
        if (verify_bytes("pua_core",
                         (const unsigned char *)pua_core,
                         expected,
                         2) != 0) {
            failures++;
        }
        /* Also verify string length (should be 2 + null terminator) */
        if (strlen(pua_core) != 2) {
            printf(
                    "FAIL [pua_core]: strlen expected 2, got %d\n",
                    (int)strlen(pua_core));
            failures++;
        }
    }

    /* ------------------------------------------------------------------
     * Test Case 2: Full range sampling
     * ------------------------------------------------------------------ */
    {
        static const unsigned char expected[] = {
            0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0xFF
        };
        if (verify_bytes("pua_range",
                         (const unsigned char *)pua_range,
                         expected,
                         9) != 0) {
            failures++;
        }
        if (strlen(pua_range) != 9) {
            printf(
                    "FAIL [pua_range]: strlen expected 9, got %d\n",
                    (int)strlen(pua_range));
            failures++;
        }
    }

    /* ------------------------------------------------------------------
     * Test Case 3: Mixed ASCII and non-UTF-8 bytes
     * "hello\x80world\xFF" = 'h','e','l','l','o',0x80,'w','o','r','l','d',0xFF
     * ------------------------------------------------------------------ */
    {
        static const unsigned char expected[] = {
            'h', 'e', 'l', 'l', 'o', 0x80,
            'w', 'o', 'r', 'l', 'd', 0xFF
        };
        if (verify_bytes("pua_mixed",
                         (const unsigned char *)pua_mixed,
                         expected,
                         12) != 0) {
            failures++;
        }
        if (strlen(pua_mixed) != 12) {
            printf(
                    "FAIL [pua_mixed]: strlen expected 12, got %d\n",
                    (int)strlen(pua_mixed));
            failures++;
        }
    }

    /* ------------------------------------------------------------------
     * Test Case 4: All 128 non-ASCII bytes (0x80–0xFF)
     * Exhaustive verification of every byte in the PUA mapping range.
     * ------------------------------------------------------------------ */
    {
        int i;
        for (i = 0; i < 128; i++) {
            unsigned char expected_byte = (unsigned char)(0x80 + i);
            unsigned char actual_byte = (unsigned char)pua_all[i];
            if (actual_byte != expected_byte) {
                printf(
                        "FAIL [pua_all]: index %d: expected 0x%02X, got 0x%02X\n",
                        i, expected_byte, actual_byte);
                failures++;
            }
        }
    }

    /* ------------------------------------------------------------------
     * Test Case 5: Boundary bytes — explicit null terminator in literal
     * pua_boundary is "\x80\xFF\x00" — the \x00 is an explicit embedded null.
     * The array therefore has 4 bytes total: 0x80, 0xFF, 0x00, 0x00 (implicit).
     * We check that the first two bytes are correct.
     * ------------------------------------------------------------------ */
    {
        static const unsigned char expected[] = { 0x80, 0xFF, 0x00 };
        if (verify_bytes("pua_boundary",
                         (const unsigned char *)pua_boundary,
                         expected,
                         3) != 0) {
            failures++;
        }
    }

    /* ------------------------------------------------------------------
     * Test Case 6: Repeated identical non-ASCII bytes
     * ------------------------------------------------------------------ */
    {
        static const unsigned char expected[] = {
            0xFF, 0xFF, 0xFF, 0x80, 0x80, 0x80
        };
        if (verify_bytes("pua_repeated",
                         (const unsigned char *)pua_repeated,
                         expected,
                         6) != 0) {
            failures++;
        }
        if (strlen(pua_repeated) != 6) {
            printf(
                    "FAIL [pua_repeated]: strlen expected 6, got %d\n",
                    (int)strlen(pua_repeated));
            failures++;
        }
    }

    /* ------------------------------------------------------------------
     * Test Case 7: Non-ASCII bytes adjacent to standard escape sequences
     * "\n\x80\t\xFF\\\x90\0\xA0"
     * Bytes: 0x0A, 0x80, 0x09, 0xFF, 0x5C, 0x90, 0x00, 0xA0
     * Note: the embedded \0 makes strlen stop early, so we verify by
     * array size (sizeof includes the implicit trailing null = 9 bytes).
     * ------------------------------------------------------------------ */
    {
        static const unsigned char expected[] = {
            0x0A, 0x80, 0x09, 0xFF, 0x5C, 0x90, 0x00, 0xA0
        };
        if (verify_bytes("pua_with_escapes",
                         (const unsigned char *)pua_with_escapes,
                         expected,
                         8) != 0) {
            failures++;
        }
    }

    /* ------------------------------------------------------------------
     * Test Case 8: Runtime-constructed verification
     * Build the expected pattern in a local array and compare against
     * the string literal to guard against compiler constant-folding
     * optimizations hiding a PUA encoding bug.
     * ------------------------------------------------------------------ */
    {
        unsigned char runtime_expected[2];
        runtime_expected[0] = (unsigned char)0x80;
        runtime_expected[1] = (unsigned char)0xFF;

        if ((unsigned char)pua_core[0] != runtime_expected[0] ||
            (unsigned char)pua_core[1] != runtime_expected[1]) {
            printf(
                    "FAIL [runtime_verify]: pua_core bytes do not match "
                    "runtime-constructed expectation\n");
            failures++;
        }
    }

    /* ------------------------------------------------------------------
     * Summary
     * ------------------------------------------------------------------ */
    if (failures == 0) {
        printf("PUA round-trip: ALL TESTS PASSED\n");
    } else {
        printf(
                "PUA round-trip: %d TEST(S) FAILED\n", failures);
    }

    return failures == 0 ? 0 : 1;
}
