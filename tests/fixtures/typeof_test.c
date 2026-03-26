/*
 * tests/fixtures/typeof_test.c
 *
 * Validates typeof/__typeof__ GCC extension support (Checkpoint 2).
 *
 * typeof infers types at compile time and is heavily used in Linux kernel
 * macros for type-safe generic programming. Both typeof(expression) and
 * typeof(type-name) forms must be supported, as well as the __typeof__
 * alternate spelling.
 *
 * Compile: bcc -o typeof_test typeof_test.c   (or gcc -std=gnu11)
 * Expected: prints "typeof OK\n" and exits 0
 */

#include <stdio.h>

/* Kernel-style type-safe max macro using typeof */
#define max(a, b) ({        \
    typeof(a) _a = (a);     \
    typeof(b) _b = (b);     \
    _a > _b ? _a : _b;      \
})

/* Kernel-style type-safe min macro using __typeof__ */
#define min(a, b) ({            \
    __typeof__(a) _a = (a);     \
    __typeof__(b) _b = (b);     \
    _a < _b ? _a : _b;          \
})

/* Kernel-style swap macro using typeof */
#define swap(a, b) do {         \
    typeof(a) _tmp = (a);       \
    (a) = (b);                  \
    (b) = _tmp;                 \
} while (0)

/* Simple struct for testing typeof with struct types */
struct point {
    int x;
    int y;
};

int main(void) {
    /* typeof with expression */
    int x = 42;
    typeof(x) y = x + 1;
    if (y != 43) { printf("FAIL: typeof(expr)\n"); return 1; }

    /* __typeof__ variant */
    double d = 3.14;
    __typeof__(d) e = d * 2.0;
    if (e < 6.27 || e > 6.29) { printf("FAIL: __typeof__\n"); return 1; }

    /* typeof with type name */
    typeof(int) z = 100;
    if (z != 100) { printf("FAIL: typeof(type)\n"); return 1; }

    /* typeof with pointer */
    int *p = &x;
    typeof(p) q = &y;
    if (*q != 43) { printf("FAIL: typeof(ptr)\n"); return 1; }

    /* typeof with array element */
    int arr[3] = {10, 20, 30};
    typeof(arr[0]) elem = arr[1];
    if (elem != 20) { printf("FAIL: typeof(arr elem)\n"); return 1; }

    /* typeof preserving qualifiers (const) */
    const int cx = 99;
    typeof(cx) cy = 99;
    if (cy != 99) { printf("FAIL: typeof(const)\n"); return 1; }

    /* typeof with struct type */
    struct point pt1 = {5, 10};
    typeof(pt1) pt2 = {15, 20};
    if (pt2.x != 15 || pt2.y != 20) { printf("FAIL: typeof(struct)\n"); return 1; }

    /* __typeof__ with struct member expression */
    __typeof__(pt1.x) sx = pt1.x + pt2.x;
    if (sx != 20) { printf("FAIL: __typeof__(struct member)\n"); return 1; }

    /* typeof in macro definitions — kernel-style max */
    int a = 7, b = 13;
    int m = max(a, b);
    if (m != 13) { printf("FAIL: typeof max macro\n"); return 1; }

    /* typeof in macro definitions — kernel-style min */
    int n = min(a, b);
    if (n != 7) { printf("FAIL: typeof min macro\n"); return 1; }

    /* typeof in macro definitions — kernel-style swap */
    swap(a, b);
    if (a != 13 || b != 7) { printf("FAIL: typeof swap macro\n"); return 1; }

    /* typeof with unsigned type */
    unsigned long ul = 1000000UL;
    typeof(ul) ul2 = ul + 1;
    if (ul2 != 1000001UL) { printf("FAIL: typeof(unsigned long)\n"); return 1; }

    /* typeof with char */
    char c = 'A';
    typeof(c) c2 = c + 1;
    if (c2 != 'B') { printf("FAIL: typeof(char)\n"); return 1; }

    /* typeof with pointer-to-pointer */
    int **pp = &p;
    typeof(pp) pp2 = &q;
    if (**pp2 != 43) { printf("FAIL: typeof(ptr-to-ptr)\n"); return 1; }

    /* typeof with function return value expression */
    typeof(1 + 2) sum = 1 + 2;
    if (sum != 3) { printf("FAIL: typeof(arith expr)\n"); return 1; }

    printf("typeof OK\n");
    return 0;
}
