/*
 * tests/fixtures/designated_init.c — Designated Initializer Test (Checkpoint 2)
 *
 * Validates designated initializer support including:
 *   - Out-of-order field designation
 *   - Implicit zero-initialization of unspecified members
 *   - Nested designation (GCC extension: .field.subfield)
 *   - Array index designation
 *   - Brace elision (initializing struct members without inner braces)
 *
 * Expected: compiles and runs with exit code 0, printing "designated_init OK\n"
 */

#include <stdio.h>

struct Point {
    int x;
    int y;
    int z;
};

struct Nested {
    struct Point origin;
    struct Point size;
    int flags;
};

int main(void) {
    /* Out-of-order field designation */
    struct Point p = { .z = 30, .x = 10, .y = 20 };
    if (p.x != 10 || p.y != 20 || p.z != 30) {
        printf("FAIL: out-of-order\n"); return 1;
    }

    /* Implicit zero-initialization of unspecified members */
    struct Point q = { .x = 5 };
    if (q.x != 5 || q.y != 0 || q.z != 0) {
        printf("FAIL: zero-init\n"); return 1;
    }

    /* Nested designation (GCC extension: .origin.x = value) */
    struct Nested n = {
        .origin.x = 1,
        .origin.y = 2,
        .size = { .x = 100, .y = 200, .z = 0 },
        .flags = 0xFF
    };
    if (n.origin.x != 1 || n.origin.y != 2 || n.origin.z != 0) {
        printf("FAIL: nested origin\n"); return 1;
    }
    if (n.size.x != 100 || n.size.y != 200) {
        printf("FAIL: nested size\n"); return 1;
    }
    if (n.flags != 0xFF) {
        printf("FAIL: nested flags\n"); return 1;
    }

    /* Array index designation */
    int arr[5] = { [2] = 20, [4] = 40, [0] = 0 };
    if (arr[0] != 0 || arr[1] != 0 || arr[2] != 20 || arr[3] != 0 || arr[4] != 40) {
        printf("FAIL: array index\n"); return 1;
    }

    /* Brace elision */
    struct Point pts[2] = { 1, 2, 3, 4, 5, 6 };
    if (pts[0].x != 1 || pts[0].y != 2 || pts[0].z != 3) {
        printf("FAIL: brace elision [0]\n"); return 1;
    }
    if (pts[1].x != 4 || pts[1].y != 5 || pts[1].z != 6) {
        printf("FAIL: brace elision [1]\n"); return 1;
    }

    printf("designated_init OK\n");
    return 0;
}
