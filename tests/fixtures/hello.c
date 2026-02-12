/*
 * hello.c — Checkpoint 1 Hello World test fixture for BCC.
 *
 * Validates the fundamental end-to-end compilation pipeline:
 *   preprocessor (#include) → lexer → parser → semantic analysis →
 *   IR lowering → code generation → assembler → linker
 *
 * Expected invocation:
 *   ./bcc -o hello tests/fixtures/hello.c && ./hello
 *
 * Expected result:
 *   stdout: "Hello, World!\n"
 *   exit code: 0
 *
 * This program is intentionally minimal and standards-compliant C11.
 * No GCC extensions, no inline assembly — purely exercising the basic
 * compilation pipeline across all four target architectures:
 *   x86-64, i686, AArch64, RISC-V 64.
 */

#include <stdio.h>

int main(void) {
    printf("Hello, World!\n");
    return 0;
}
