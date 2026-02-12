/*
 * static_assert.c — C11 _Static_assert test fixture for Checkpoint 2
 *
 * This file validates that the BCC compiler correctly parses and evaluates
 * _Static_assert declarations at file scope, block scope, and struct scope.
 * All assertions in this file are valid and must compile successfully.
 *
 * Invalid _Static_assert cases (that should cause compilation errors) are
 * intentionally commented out. The test runner (checkpoint2_language.rs) tests
 * invalid cases separately by compiling dedicated error-case snippets and
 * checking for the expected diagnostic message.
 *
 * Validates:
 *   - Parser declarations.rs: _Static_assert parsing
 *   - Sema constant_eval.rs: compile-time integer constant expression evaluation
 */

#include <stdio.h>

/* ========================================================================
 * 1. File-scope static assertions — basic type properties
 * ======================================================================== */

/* sizeof(int) must be at least 4 bytes on all supported targets */
_Static_assert(sizeof(int) >= 4, "int must be at least 4 bytes");

/* sizeof(char) is always exactly 1 by the C standard (§6.5.3.4) */
_Static_assert(sizeof(char) == 1, "char must be 1 byte");

/* Basic compile-time arithmetic in the constant expression */
_Static_assert(1 + 1 == 2, "basic arithmetic");

/* Pointer size is either 4 (i686) or 8 (x86-64, AArch64, RISC-V 64) */
_Static_assert(sizeof(void *) == 8 || sizeof(void *) == 4, "pointer size");

/* ========================================================================
 * 2. File-scope static assertions — additional sizeof checks
 * ======================================================================== */

/* short is at least 2 bytes per the C standard */
_Static_assert(sizeof(short) >= 2, "short must be at least 2 bytes");

/* long long is at least 8 bytes per the C standard */
_Static_assert(sizeof(long long) >= 8, "long long must be at least 8 bytes");

/* float and double sizes per IEEE 754 */
_Static_assert(sizeof(float) == 4, "float must be 4 bytes");
_Static_assert(sizeof(double) == 8, "double must be 8 bytes");

/* char signedness-agnostic size */
_Static_assert(sizeof(signed char) == 1, "signed char must be 1 byte");
_Static_assert(sizeof(unsigned char) == 1, "unsigned char must be 1 byte");

/* ========================================================================
 * 3. Preprocessor-defined constant in _Static_assert
 * ======================================================================== */

#define EXPECTED_CHAR_SIZE 1
#define MAX_ALIGN 16
#define COMPILE_TIME_VALUE (2 * 3 + 1)

_Static_assert(EXPECTED_CHAR_SIZE == sizeof(char),
               "preprocessor constant matches sizeof(char)");

_Static_assert(COMPILE_TIME_VALUE == 7,
               "preprocessor arithmetic expression");

_Static_assert(MAX_ALIGN >= 1, "MAX_ALIGN is positive");

/* ========================================================================
 * 4. Enum values in _Static_assert constant expressions
 * ======================================================================== */

enum limits {
    LIMIT_MIN = 0,
    LIMIT_LOW = 10,
    LIMIT_MID = 100,
    LIMIT_HIGH = 1000,
    LIMIT_MAX = 10000
};

_Static_assert(LIMIT_MIN == 0, "enum LIMIT_MIN is zero");
_Static_assert(LIMIT_LOW < LIMIT_MID, "enum ordering: LOW < MID");
_Static_assert(LIMIT_MID < LIMIT_HIGH, "enum ordering: MID < HIGH");
_Static_assert(LIMIT_HIGH < LIMIT_MAX, "enum ordering: HIGH < MAX");
_Static_assert(LIMIT_MAX == 10000, "enum LIMIT_MAX is 10000");

/* ========================================================================
 * 5. Complex constant expressions combining multiple operators
 * ======================================================================== */

_Static_assert((sizeof(int) * 8) >= 32,
               "int must have at least 32 bits");

_Static_assert((1 << 10) == 1024,
               "bit shift: 1 << 10 == 1024");

_Static_assert(((unsigned)(-1)) > 0,
               "unsigned -1 wraps to a positive value");

_Static_assert(sizeof(int[10]) == 10 * sizeof(int),
               "array sizeof equals element count times element size");

_Static_assert((5 > 3) && (2 < 4),
               "logical AND of comparisons");

_Static_assert((0 == 1) || (1 == 1),
               "logical OR with one true operand");

_Static_assert(!(0),
               "logical NOT of zero is true");

_Static_assert((10 % 3) == 1,
               "modulo operator in constant expression");

_Static_assert((0xFF & 0x0F) == 0x0F,
               "bitwise AND in constant expression");

_Static_assert((0xF0 | 0x0F) == 0xFF,
               "bitwise OR in constant expression");

_Static_assert((0xFF ^ 0x0F) == 0xF0,
               "bitwise XOR in constant expression");

_Static_assert((1 ? 42 : 0) == 42,
               "ternary operator in constant expression");

/* ========================================================================
 * 6. _Static_assert inside a struct (struct scope)
 * ======================================================================== */

struct validated {
    int x;
    _Static_assert(sizeof(int) == 4, "struct member size check");
    int y;
};

/* A more complex struct with multiple static assertions */
struct complex_validated {
    char tag;
    _Static_assert(sizeof(char) == 1, "tag field is 1 byte");
    int value;
    _Static_assert(sizeof(int) >= 4, "value field is at least 4 bytes");
    double weight;
    _Static_assert(sizeof(double) == 8, "weight field is 8 bytes");
};

/* ========================================================================
 * 7. _Static_assert inside a union
 * ======================================================================== */

union data_cell {
    int i;
    float f;
    _Static_assert(sizeof(float) == sizeof(int),
                   "float and int have equal size in this union");
    char bytes[4];
};

/* ========================================================================
 * 8. _Static_assert with typedef-related checks
 * ======================================================================== */

typedef unsigned long ulong_t;
_Static_assert(sizeof(ulong_t) >= 4,
               "ulong_t must be at least 4 bytes");

typedef int int32_checked_t;
_Static_assert(sizeof(int32_checked_t) == 4,
               "int32_checked_t must be exactly 4 bytes");

/* ========================================================================
 * 9. main function exercising the validated struct (block scope assertions)
 * ======================================================================== */

int main(void) {
    /* Block-scope _Static_assert declarations */
    _Static_assert(sizeof(int) >= 4, "block scope: int size");
    _Static_assert(sizeof(char) == 1, "block scope: char size");
    _Static_assert(1, "block scope: nonzero is true");

    /* Enum value in block scope static assert */
    _Static_assert(LIMIT_MAX > LIMIT_MIN,
                   "block scope: enum comparison");

    /* Preprocessor constant in block scope static assert */
    _Static_assert(COMPILE_TIME_VALUE == 7,
                   "block scope: preprocessor constant");

    /* Use the validated struct to confirm it was parsed correctly */
    struct validated v;
    v.x = 10;
    v.y = 20;

    /* Use the complex_validated struct */
    struct complex_validated cv;
    cv.tag = 'A';
    cv.value = 42;
    cv.weight = 3.14;

    /* Use the union */
    union data_cell dc;
    dc.i = 0x41424344;

    /* Access union member to suppress unused-but-set warning */
    if (dc.bytes[0] == 0 && dc.bytes[1] == 0 && dc.bytes[2] == 0 && dc.bytes[3] == 0) {
        printf("FAIL: union data_cell zeroed unexpectedly\n");
        return 1;
    }

    /* Verify the struct fields are accessible and correct */
    if (v.x != 10 || v.y != 20) {
        printf("FAIL: validated struct fields incorrect\n");
        return 1;
    }

    if (cv.tag != 'A' || cv.value != 42) {
        printf("FAIL: complex_validated struct fields incorrect\n");
        return 1;
    }

    printf("PASS: all _Static_assert tests compiled and ran successfully\n");
    return 0;
}

/* ========================================================================
 * 10. Commented-out invalid _Static_assert cases
 *
 *     These would cause compilation errors if uncommented.
 *     The test runner (checkpoint2_language.rs) tests invalid cases
 *     separately by compiling dedicated error-case snippets.
 *
 *     Example invalid cases:
 *
 *     // Assertion with a false condition — must produce a compile error
 *     // with the specified message string:
 *     // _Static_assert(sizeof(char) == 2, "this should fail");
 *
 *     // Assertion with a zero constant expression:
 *     // _Static_assert(0, "zero is false");
 *
 *     // Assertion with a negative result:
 *     // _Static_assert(1 - 2, "nonzero but this is actually -1 which is truthy");
 *     // Note: -1 is truthy (nonzero), so this would actually pass.
 *     // The real failing case is a zero expression:
 *     // _Static_assert(1 == 2, "one does not equal two");
 *
 *     // Non-constant expression — must be diagnosed:
 *     // int runtime_var = 5;
 *     // _Static_assert(runtime_var > 0, "runtime var not allowed");
 *
 * ======================================================================== */
