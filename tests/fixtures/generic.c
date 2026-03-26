/*
 * tests/fixtures/generic.c — C11 _Generic Selection Test (Checkpoint 2)
 *
 * Validates C11 _Generic selection expression support. _Generic provides
 * compile-time type-based dispatch, used in C11 type-generic math macros
 * and other type-safe macro patterns.
 *
 * _Generic syntax:
 *   _Generic(controlling_expr, type1: expr1, type2: expr2, ..., default: expr_default)
 *
 * The controlling expression is NOT evaluated — only its type matters.
 * The compiler must select the correct association based on the type of
 * the controlling expression at compile time.
 *
 * Expected output: "generic OK\n"
 * Expected exit code: 0
 */

#include <stdio.h>

/* Type name macro using _Generic — returns a string describing the type */
#define type_name(x) _Generic((x), \
    int: "int", \
    double: "double", \
    char *: "char *", \
    float: "float", \
    default: "other")

/* Type-specific operation macro — returns an integer ID per type */
#define type_id(x) _Generic((x), \
    int: 1, \
    double: 2, \
    char *: 3, \
    default: 0)

/* Nested _Generic: dispatches on type to produce a string, then dispatches
   on that result (always char *) to confirm it resolves to 3 */
#define nested_check(x) _Generic( \
    _Generic((x), int: "int_path", double: "dbl_path", default: "def_path"), \
    char *: 99, \
    default: -1)

int main(void) {
    /* ================================================================
     * Test 1: type_id macro — integer type selection
     * ================================================================ */
    int i = 42;
    int id_i = type_id(i);
    if (id_i != 1) { printf("FAIL: type_id(int)=%d\n", id_i); return 1; }

    /* ================================================================
     * Test 2: type_id macro — double type selection
     * ================================================================ */
    double d = 3.14;
    int id_d = type_id(d);
    if (id_d != 2) { printf("FAIL: type_id(double)=%d\n", id_d); return 1; }

    /* ================================================================
     * Test 3: type_id macro — char * type selection
     * ================================================================ */
    char *s = "hello";
    int id_s = type_id(s);
    if (id_s != 3) { printf("FAIL: type_id(char*)=%d\n", id_s); return 1; }

    /* ================================================================
     * Test 4: type_id macro — default fallback case (long has no
     *         explicit association, so it falls through to default: 0)
     * ================================================================ */
    long l = 100L;
    int id_l = type_id(l);
    if (id_l != 0) { printf("FAIL: type_id(long)=%d\n", id_l); return 1; }

    /* ================================================================
     * Test 5: type_name macro — string-based type dispatching
     * ================================================================ */
    const char *name_i = type_name(i);
    const char *name_d = type_name(d);
    /* Verify strings are correct (compare first chars) */
    if (name_i[0] != 'i') { printf("FAIL: type_name int\n"); return 1; }
    if (name_d[0] != 'd') { printf("FAIL: type_name double\n"); return 1; }

    /* ================================================================
     * Test 6: Direct _Generic expression (not through a macro)
     *         1.0f has type float, so the float association is selected.
     * ================================================================ */
    int result = _Generic(1.0f, float: 10, double: 20, default: 30);
    if (result != 10) { printf("FAIL: direct _Generic float\n"); return 1; }

    /* ================================================================
     * Test 7: Direct _Generic with double literal
     *         1.0 (without f suffix) has type double.
     * ================================================================ */
    int result2 = _Generic(1.0, float: 10, double: 20, default: 30);
    if (result2 != 20) { printf("FAIL: direct _Generic double\n"); return 1; }

    /* ================================================================
     * Test 8: Direct _Generic hitting the default case
     *         A short has no explicit association listed.
     * ================================================================ */
    short sh = 5;
    int result3 = _Generic(sh, int: 10, double: 20, default: 30);
    if (result3 != 30) { printf("FAIL: direct _Generic default\n"); return 1; }

    /* ================================================================
     * Test 9: Nested _Generic expressions
     *         The inner _Generic resolves to a char * string literal,
     *         and the outer _Generic dispatches on char * to yield 99.
     * ================================================================ */
    int nested_i = nested_check(i);
    if (nested_i != 99) { printf("FAIL: nested _Generic int=%d\n", nested_i); return 1; }

    int nested_d = nested_check(d);
    if (nested_d != 99) { printf("FAIL: nested _Generic double=%d\n", nested_d); return 1; }

    int nested_l = nested_check(l);
    if (nested_l != 99) { printf("FAIL: nested _Generic default=%d\n", nested_l); return 1; }

    /* ================================================================
     * Test 10: _Generic with unsigned types — unsigned int is distinct
     *          from int in the type system; must fall to default.
     * ================================================================ */
    unsigned int ui = 7u;
    int id_ui = type_id(ui);
    if (id_ui != 0) { printf("FAIL: type_id(unsigned int)=%d\n", id_ui); return 1; }

    /* ================================================================
     * All tests passed.
     * ================================================================ */
    printf("generic OK\n");
    return 0;
}
