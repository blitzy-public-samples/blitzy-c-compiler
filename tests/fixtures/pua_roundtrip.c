/*
 * pua_roundtrip.c — Non-UTF-8 Byte Round-Trip Test (Checkpoint 2)
 *
 * Validates PUA (Private Use Area) encoding fidelity in the BCC compiler
 * pipeline. Non-UTF-8 bytes (0x80–0xFF) in C source files must survive the
 * entire compilation pipeline with byte-exact fidelity.
 *
 * The BCC preprocessor encodes non-UTF-8 bytes (0x80–0xFF) as Unicode
 * Private Use Area code points (U+E080–U+E0FF) on source file read, and
 * the code generator decodes them back to the original raw bytes on output.
 * This test verifies that round-trip is lossless.
 *
 * Validation protocol:
 *   1. Compile with BCC: ./bcc -o pua_roundtrip tests/fixtures/pua_roundtrip.c
 *   2. Inspect .rodata:  objdump -s -j .rodata pua_roundtrip
 *      → Confirm exact bytes 80 ff are present in the hex dump
 *   3. Run the binary:   ./pua_roundtrip
 *      → Output: "PUA round-trip OK"
 *      → Exit code: 0
 *
 * Per AAP §0.7.9: The Linux kernel contains binary data in string literals
 * and inline assembly operands; encoding failure corrupts generated machine
 * code.
 */

#include <stdio.h>
#include <string.h>

/*
 * Core test data: string literal containing the two critical boundary bytes.
 * These bytes (0x80 and 0xFF) are NOT valid UTF-8 lead or continuation bytes,
 * so they exercise the PUA encoding path in the BCC pipeline.
 */
static const char core_data[] = "\x80\xFF";

/*
 * Extended test data: a sweep of representative bytes across the full
 * non-UTF-8 range (0x80–0xFF). This exercises the complete PUA mapping
 * from U+E080 through U+E0FF.
 */
static const char extended_data[] =
    "\x80\x81\x8F\x90\x9F\xA0\xAF\xB0\xBF\xC0\xCF\xD0\xDF\xE0\xEF\xF0\xFE\xFF";

/*
 * Mixed content: non-UTF-8 bytes interleaved with normal ASCII. This tests
 * that the PUA encoding does not corrupt adjacent ASCII characters.
 */
static const char mixed_data[] = "A\x80" "B\xFF" "C\xA0" "D\xFE" "E";

/*
 * Consecutive identical high bytes: ensures no byte merging or deduplication
 * occurs during PUA encoding/decoding.
 */
static const char repeated_data[] = "\xFF\xFF\xFF\x80\x80\x80";

/*
 * Embedded within a longer ASCII context — simulates kernel-style string
 * literals that mix human-readable text with raw binary data.
 */
static const char kernel_style[] = "magic:\x89\x50\x4E\x47\x0D\x0A\x1A\x0A:end";

/* Helper: verify a single byte at a given index */
static int check_byte(const char *name, const char *data, int index,
                      unsigned char expected)
{
    unsigned char actual = (unsigned char)data[index];
    if (actual != expected) {
        printf("FAIL: %s[%d] = 0x%02x, expected 0x%02x\n",
               name, index, actual, expected);
        return 1;
    }
    return 0;
}

int main(void)
{
    int failures = 0;

    /* === Test 1: Core data — \x80\xFF === */

    /* Verify string length: 2 data bytes + 1 null terminator = 3 total */
    if (strlen(core_data) != 2) {
        printf("FAIL: core_data strlen = %d, expected 2\n", (int)strlen(core_data));
        failures++;
    }

    failures += check_byte("core_data", core_data, 0, 0x80);
    failures += check_byte("core_data", core_data, 1, 0xFF);

    /* Verify null terminator is intact */
    failures += check_byte("core_data", core_data, 2, 0x00);

    /* === Test 2: Extended range sweep === */

    {
        static const unsigned char expected_extended[] = {
            0x80, 0x81, 0x8F, 0x90, 0x9F, 0xA0, 0xAF, 0xB0,
            0xBF, 0xC0, 0xCF, 0xD0, 0xDF, 0xE0, 0xEF, 0xF0,
            0xFE, 0xFF
        };
        int extended_len = (int)(sizeof(expected_extended) / sizeof(expected_extended[0]));
        int i;

        if ((int)strlen(extended_data) != extended_len) {
            printf("FAIL: extended_data strlen = %d, expected %d\n",
                   (int)strlen(extended_data), extended_len);
            failures++;
        }

        for (i = 0; i < extended_len; i++) {
            failures += check_byte("extended_data", extended_data, i,
                                   expected_extended[i]);
        }
    }

    /* === Test 3: Mixed ASCII and non-UTF-8 bytes === */

    {
        /*
         * mixed_data = "A\x80" "B\xFF" "C\xA0" "D\xFE" "E"
         * Adjacent string literals are concatenated by the compiler.
         * Expected: 'A', 0x80, 'B', 0xFF, 'C', 0xA0, 'D', 0xFE, 'E'
         */
        static const unsigned char expected_mixed[] = {
            'A', 0x80, 'B', 0xFF, 'C', 0xA0, 'D', 0xFE, 'E'
        };
        int mixed_len = (int)(sizeof(expected_mixed) / sizeof(expected_mixed[0]));
        int i;

        if ((int)strlen(mixed_data) != mixed_len) {
            printf("FAIL: mixed_data strlen = %d, expected %d\n",
                   (int)strlen(mixed_data), mixed_len);
            failures++;
        }

        for (i = 0; i < mixed_len; i++) {
            failures += check_byte("mixed_data", mixed_data, i,
                                   expected_mixed[i]);
        }
    }

    /* === Test 4: Repeated identical high bytes === */

    {
        static const unsigned char expected_repeated[] = {
            0xFF, 0xFF, 0xFF, 0x80, 0x80, 0x80
        };
        int repeated_len = (int)(sizeof(expected_repeated) / sizeof(expected_repeated[0]));
        int i;

        if ((int)strlen(repeated_data) != repeated_len) {
            printf("FAIL: repeated_data strlen = %d, expected %d\n",
                   (int)strlen(repeated_data), repeated_len);
            failures++;
        }

        for (i = 0; i < repeated_len; i++) {
            failures += check_byte("repeated_data", repeated_data, i,
                                   expected_repeated[i]);
        }
    }

    /* === Test 5: Kernel-style mixed binary data === */

    {
        /*
         * kernel_style = "magic:\x89\x50\x4E\x47\x0D\x0A\x1A\x0A:end"
         *
         * This is a PNG file signature (89 50 4E 47 0D 0A 1A 0A) embedded
         * within an ASCII context. Bytes 0x89 and 0x1A are non-UTF-8 and
         * exercise the PUA path; 0x50/0x4E/0x47/0x0D/0x0A are normal ASCII
         * and must pass through unmodified.
         */
        static const unsigned char expected_kernel[] = {
            'm', 'a', 'g', 'i', 'c', ':',
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
            ':', 'e', 'n', 'd'
        };
        int kernel_len = (int)(sizeof(expected_kernel) / sizeof(expected_kernel[0]));
        int i;

        if ((int)strlen(kernel_style) != kernel_len) {
            printf("FAIL: kernel_style strlen = %d, expected %d\n",
                   (int)strlen(kernel_style), kernel_len);
            failures++;
        }

        for (i = 0; i < kernel_len; i++) {
            failures += check_byte("kernel_style", kernel_style, i,
                                   expected_kernel[i]);
        }
    }

    /* === Test 6: sizeof verification === */

    /*
     * sizeof(core_data) must be exactly 3: two data bytes + null terminator.
     * This confirms no byte expansion or contraction occurred during PUA
     * encoding/decoding.
     */
    if (sizeof(core_data) != 3) {
        printf("FAIL: sizeof(core_data) = %d, expected 3\n",
               (int)sizeof(core_data));
        failures++;
    }

    /* === Report results === */

    if (failures > 0) {
        printf("PUA round-trip FAILED: %d error(s)\n", failures);
        return 1;
    }

    printf("PUA round-trip OK\n");
    return 0;
}
