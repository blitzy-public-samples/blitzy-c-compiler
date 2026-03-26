/*
 * tests/fixtures/static_assert.c — C11 _Static_assert Test (Checkpoint 2)
 *
 * Validates C11 _Static_assert support at file scope and block scope.
 * All assertions use true conditions so the program compiles and runs.
 * Exercises:
 *   - src/frontend/parser/declarations.rs (_Static_assert parsing)
 *   - src/frontend/sema/constant_eval.rs  (constant expression evaluation)
 *
 * Expected output: "static_assert OK\n"
 * Expected exit code: 0
 */

#include <stdio.h>
#include <limits.h>

/* ===== File-scope static assertions ===== */

/* Basic sizeof checks — guaranteed by the C standard */
_Static_assert(sizeof(int) >= 2, "int must be at least 2 bytes");
_Static_assert(sizeof(char) == 1, "char must be exactly 1 byte");
_Static_assert(sizeof(long long) >= 8, "long long must be at least 8 bytes");

/* Assertions with arithmetic expressions */
_Static_assert(1 + 1 == 2, "basic arithmetic failed");
_Static_assert(sizeof(void *) == 4 || sizeof(void *) == 8, "pointer must be 4 or 8 bytes");

/* Assertions using limits.h constants */
_Static_assert(CHAR_BIT == 8, "CHAR_BIT must be 8");
_Static_assert(INT_MAX >= 2147483647, "INT_MAX must be at least 2^31-1");

/* Struct for layout assertion */
struct layout_check {
    char a;
    int b;
};

/* Assertion about struct properties — padding makes this >= 5 bytes */
_Static_assert(sizeof(struct layout_check) >= 5, "struct must be at least 5 bytes");

int main(void) {
    /* ===== Block-scope static assertions ===== */

    /* Type size comparison */
    _Static_assert(sizeof(int) == sizeof(unsigned int), "int and unsigned int must be same size");

    /* Simple constant truth */
    _Static_assert(1, "constant 1 is truthy");

    /* Assertion with sizeof on a local fixed-size array */
    int arr[10];
    _Static_assert(sizeof(arr) == 10 * sizeof(int), "array size check");

    printf("static_assert OK\n");
    return 0;
}
