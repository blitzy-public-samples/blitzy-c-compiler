/*
 * generic.c — C11 _Generic selection expression test fixture
 *
 * Validates that the BCC parser (expressions.rs) and semantic analyzer
 * correctly parse and resolve _Generic selection expressions based on
 * the controlling expression's type.
 *
 * Covers:
 *   - Basic _Generic with int, float, double, char*, default
 *   - Type-safe function dispatch via _Generic macro
 *   - Controlling expression from literals (int, float, string)
 *   - Controlling expression from declared variables of various types
 *   - Nested _Generic expressions
 *   - _Generic with default fallback
 *   - _Generic used in compile-time constant contexts
 *
 * Checkpoint 2 fixture — _Generic selection test source.
 * Removing this file breaks the _Generic validation in Checkpoint 2.
 */

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* 1. Basic _Generic type name dispatch macro                         */
/* ------------------------------------------------------------------ */
#define type_name(x) _Generic((x), \
    int: "int",                    \
    float: "float",                \
    double: "double",              \
    char *: "char*",               \
    default: "other")

/* ------------------------------------------------------------------ */
/* 2. Type-safe abs_val using _Generic for function dispatch          */
/* ------------------------------------------------------------------ */
static int abs_int(int v) {
    return v < 0 ? -v : v;
}

static float abs_float(float v) {
    return v < 0.0f ? -v : v;
}

static double abs_double(double v) {
    return v < 0.0 ? -v : v;
}

#define abs_val(x) _Generic((x), \
    int: abs_int,                \
    float: abs_float,            \
    double: abs_double)(x)

/* ------------------------------------------------------------------ */
/* 3. _Generic selecting a constant based on type                     */
/* ------------------------------------------------------------------ */
#define type_size(x) _Generic((x), \
    char: 1,                       \
    short: 2,                      \
    int: 4,                        \
    long: 8,                       \
    default: 0)

/* ------------------------------------------------------------------ */
/* 4. _Generic with pointer and qualified types                       */
/* ------------------------------------------------------------------ */
#define is_pointer(x) _Generic((x),       \
    int *: 1,                             \
    float *: 1,                           \
    double *: 1,                          \
    char *: 1,                            \
    const char *: 1,                      \
    void *: 1,                            \
    default: 0)

/* ------------------------------------------------------------------ */
/* 5. _Generic with unsigned integer variants                         */
/* ------------------------------------------------------------------ */
#define is_unsigned(x) _Generic((x),      \
    unsigned char: 1,                     \
    unsigned short: 1,                    \
    unsigned int: 1,                      \
    unsigned long: 1,                     \
    unsigned long long: 1,               \
    default: 0)

/* ------------------------------------------------------------------ */
/* Helper: simple check macro with failure reporting                  */
/* ------------------------------------------------------------------ */
static int test_failures = 0;

#define CHECK(cond, msg) do {                             \
    if (!(cond)) {                                        \
        printf("FAIL: %s (line %d)\n", (msg), __LINE__);  \
        test_failures++;                                  \
    }                                                     \
} while (0)

/* ------------------------------------------------------------------ */
/* main — Exercise all _Generic test cases                            */
/* ------------------------------------------------------------------ */
int main(void) {
    /* -------------------------------------------------------------- */
    /* Test 1: Basic type_name dispatch                               */
    /* -------------------------------------------------------------- */
    {
        int i = 0;
        float f = 0.0f;
        double d = 0.0;
        char *s = "hello";

        CHECK(strcmp(type_name(i), "int") == 0,
              "type_name(int) should be \"int\"");
        CHECK(strcmp(type_name(f), "float") == 0,
              "type_name(float) should be \"float\"");
        CHECK(strcmp(type_name(d), "double") == 0,
              "type_name(double) should be \"double\"");
        CHECK(strcmp(type_name(s), "char*") == 0,
              "type_name(char*) should be \"char*\"");
    }

    /* -------------------------------------------------------------- */
    /* Test 2: default fallback                                       */
    /* -------------------------------------------------------------- */
    {
        long long ll = 0;
        /* long long is not in the type_name association list,        */
        /* so it must hit the default case → "other"                  */
        CHECK(strcmp(type_name(ll), "other") == 0,
              "type_name(long long) should be \"other\" (default)");

        unsigned int u = 0;
        CHECK(strcmp(type_name(u), "other") == 0,
              "type_name(unsigned int) should be \"other\" (default)");
    }

    /* -------------------------------------------------------------- */
    /* Test 3: _Generic for type-safe abs_val dispatch                */
    /* -------------------------------------------------------------- */
    {
        CHECK(abs_val(-5) == 5,
              "abs_val(-5) should be 5");
        CHECK(abs_val(7) == 7,
              "abs_val(7) should be 7");

        float fv = -3.5f;
        CHECK(abs_val(fv) > 3.4f && abs_val(fv) < 3.6f,
              "abs_val(-3.5f) should be ~3.5f");

        double dv = -2.25;
        CHECK(abs_val(dv) > 2.24 && abs_val(dv) < 2.26,
              "abs_val(-2.25) should be ~2.25");
    }

    /* -------------------------------------------------------------- */
    /* Test 4: _Generic with literal controlling expressions          */
    /* -------------------------------------------------------------- */
    {
        /* Integer literal — type is int */
        int r_int = _Generic(0, int: 1, default: 0);
        CHECK(r_int == 1,
              "_Generic(0, int:1, default:0) should be 1");

        /* Float literal — type is float */
        int r_float = _Generic(0.0f, float: 1, default: 0);
        CHECK(r_float == 1,
              "_Generic(0.0f, float:1, default:0) should be 1");

        /* Double literal — type is double (unsuffixed) */
        int r_double = _Generic(0.0, double: 1, default: 0);
        CHECK(r_double == 1,
              "_Generic(0.0, double:1, default:0) should be 1");

        /* String literal — type is char* (or const char* depending on mode) */
        int r_str = _Generic("hello", char *: 1, default: 0);
        CHECK(r_str == 1,
              "_Generic(\"hello\", char*:1, default:0) should be 1");
    }

    /* -------------------------------------------------------------- */
    /* Test 5: _Generic with declared variable types                  */
    /* -------------------------------------------------------------- */
    {
        char c = 'A';
        short sh = 10;
        int i = 42;
        long l = 100L;
        float f = 1.0f;
        double d = 2.0;
        int *ip = &i;
        void *vp = (void *)0;

        /* type_size macro: char→1, short→2, int→4, long→8 */
        CHECK(type_size(c) == 1,  "type_size(char) should be 1");
        CHECK(type_size(sh) == 2, "type_size(short) should be 2");
        CHECK(type_size(i) == 4,  "type_size(int) should be 4");
        CHECK(type_size(l) == 8,  "type_size(long) should be 8");
        CHECK(type_size(f) == 0,  "type_size(float) should be 0 (default)");
        CHECK(type_size(d) == 0,  "type_size(double) should be 0 (default)");

        /* is_pointer macro */
        CHECK(is_pointer(ip) == 1, "is_pointer(int*) should be 1");
        CHECK(is_pointer(vp) == 1, "is_pointer(void*) should be 1");
        CHECK(is_pointer(i) == 0,  "is_pointer(int) should be 0");
    }

    /* -------------------------------------------------------------- */
    /* Test 6: _Generic with unsigned variants                        */
    /* -------------------------------------------------------------- */
    {
        unsigned char uc = 0;
        unsigned int ui = 0;
        unsigned long ul = 0;
        unsigned long long ull = 0;
        int si = 0;

        CHECK(is_unsigned(uc) == 1,   "is_unsigned(unsigned char) should be 1");
        CHECK(is_unsigned(ui) == 1,   "is_unsigned(unsigned int) should be 1");
        CHECK(is_unsigned(ul) == 1,   "is_unsigned(unsigned long) should be 1");
        CHECK(is_unsigned(ull) == 1,  "is_unsigned(unsigned long long) should be 1");
        CHECK(is_unsigned(si) == 0,   "is_unsigned(int) should be 0 (default)");
    }

    /* -------------------------------------------------------------- */
    /* Test 7: Nested _Generic expressions                            */
    /* -------------------------------------------------------------- */
    {
        /*
         * Outer _Generic selects on the type of the controlling
         * expression; inner _Generic selects within a branch.
         *
         *   nested_check(x):
         *     if x is int   → inner _Generic(1,   int:10, default:0)  → 10
         *     if x is float → inner _Generic(1.0f, float:20, default:0) → 20
         *     default       → 0
         */
        #define nested_check(x) _Generic((x),              \
            int:   _Generic(1, int: 10, default: 0),       \
            float: _Generic(1.0f, float: 20, default: 0),  \
            default: 0)

        int ni = 0;
        float nf = 0.0f;
        double nd = 0.0;

        CHECK(nested_check(ni) == 10,
              "nested _Generic(int) should resolve to 10");
        CHECK(nested_check(nf) == 20,
              "nested _Generic(float) should resolve to 20");
        CHECK(nested_check(nd) == 0,
              "nested _Generic(double) should resolve to 0 (outer default)");
    }

    /* -------------------------------------------------------------- */
    /* Test 8: _Generic as function argument                          */
    /* -------------------------------------------------------------- */
    {
        int val = 42;
        /*
         * _Generic used directly as a printf argument.
         * Selects the string "int" because val is int.
         */
        const char *name = _Generic(val,
            int: "int",
            float: "float",
            default: "other");
        CHECK(strcmp(name, "int") == 0,
              "_Generic as function arg should select \"int\"");
    }

    /* -------------------------------------------------------------- */
    /* Test 9: _Generic with compatible but distinct types             */
    /* -------------------------------------------------------------- */
    {
        /*
         * Verify that signed / unsigned distinction is respected.
         * _Generic(0u, unsigned int: 1, int: 2, default: 0)
         * 0u is unsigned int → should select 1.
         */
        int r = _Generic(0u, unsigned int: 1, int: 2, default: 0);
        CHECK(r == 1,
              "_Generic(0u) should select unsigned int branch (1)");
    }

    /* -------------------------------------------------------------- */
    /* Test 10: _Generic controlling expr is an expression, not type   */
    /* -------------------------------------------------------------- */
    {
        int a = 1;
        int b = 2;
        /*
         * The controlling expression is (a + b) which has type int.
         * Verify _Generic evaluates the type of the expression, not its value.
         */
        int r = _Generic(a + b, int: 100, double: 200, default: 0);
        CHECK(r == 100,
              "_Generic(a+b) should select int branch (100)");
    }

    /* -------------------------------------------------------------- */
    /* Test 11: _Generic with only default                            */
    /* -------------------------------------------------------------- */
    {
        int val = 0;
        int r = _Generic(val, default: 999);
        CHECK(r == 999,
              "_Generic with only default should select default (999)");
    }

    /* -------------------------------------------------------------- */
    /* Test 12: _Generic in initializer                               */
    /* -------------------------------------------------------------- */
    {
        int x = 5;
        int init_val = _Generic(x, int: 10, default: 20);
        CHECK(init_val == 10,
              "_Generic in initializer should yield 10 for int");
    }

    /* -------------------------------------------------------------- */
    /* Report results                                                 */
    /* -------------------------------------------------------------- */
    if (test_failures == 0) {
        printf("All _Generic tests passed.\n");
        return 0;
    } else {
        printf("%d _Generic test(s) FAILED.\n", test_failures);
        return 1;
    }
}
