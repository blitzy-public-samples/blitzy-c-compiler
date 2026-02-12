/*
 * designated_init.c — Designated initializer test fixture for BCC Checkpoint 2
 *
 * Exercises C11 designated initializers and GCC range designator extension:
 *   - Out-of-order field designation
 *   - Nested struct designation (.field.subfield)
 *   - Array index designation ([N] = value)
 *   - Brace elision with designated initializers
 *   - Implicit zero-initialization of unspecified members
 *   - GCC range designator ([lo ... hi] = value)
 *
 * Expected: compiles successfully and returns 0 when all assertions pass.
 */

#include <stdio.h>
#include <string.h>

/* ---------- Type Definitions ---------- */

/* Simple struct for out-of-order and zero-init tests */
struct point {
    int x, y, z;
};

/* Nested struct for .field.subfield designation */
struct nested {
    struct point origin;
    int weight;
};

/* Struct with mixed types for broader coverage */
struct config {
    int id;
    double value;
    char name[16];
    int flags[4];
};

/* Struct used for brace-elision test */
struct pair {
    int a;
    int b;
};

struct pair_array {
    struct pair items[3];
    int count;
};

/* Struct with deeply nested designation */
struct deep {
    struct nested inner;
    int tag;
};

/* ---------- Helper ---------- */

static int failures = 0;

static void check(int condition, const char *label) {
    if (!condition) {
        printf("FAIL: %s\n", label);
        failures++;
    }
}

/* ---------- 1. Out-of-order field designation ---------- */

struct point p_global = { .z = 3, .x = 1, .y = 2 };

/* ---------- 2. Nested struct designation ---------- */

struct nested n_global = { .origin.x = 10, .origin.y = 20, .weight = 5 };

/* ---------- 3. Array index designation ---------- */

int arr_global[10] = { [3] = 30, [7] = 70, [0] = 100 };

/* ---------- 4. Implicit zero-initialization ---------- */

struct point q_global = { .x = 1 };  /* y and z must be 0 */

/* ---------- 5. GCC range designator ---------- */

int range_global[10] = { [2 ... 5] = 42 };

/* ---------- 6. Mixed types with designated init ---------- */

struct config cfg_global = {
    .name = "test",
    .id = 99,
    .flags = { [0] = 1, [3] = 8 },
    .value = 3.14
};

/* ---------- 7. Deep nesting designation ---------- */

struct deep deep_global = {
    .inner.origin.x = 100,
    .inner.origin.z = 300,
    .inner.weight = 50,
    .tag = 7
};

/* ---------- main ---------- */

int main(void) {
    /* ===== 1. Out-of-order field designation ===== */
    check(p_global.x == 1,  "out-of-order: p.x == 1");
    check(p_global.y == 2,  "out-of-order: p.y == 2");
    check(p_global.z == 3,  "out-of-order: p.z == 3");

    /* Same test with a local variable */
    struct point p_local = { .z = 30, .x = 10, .y = 20 };
    check(p_local.x == 10, "out-of-order local: p.x == 10");
    check(p_local.y == 20, "out-of-order local: p.y == 20");
    check(p_local.z == 30, "out-of-order local: p.z == 30");

    /* ===== 2. Nested struct designation (.field.subfield) ===== */
    check(n_global.origin.x == 10, "nested: n.origin.x == 10");
    check(n_global.origin.y == 20, "nested: n.origin.y == 20");
    check(n_global.origin.z == 0,  "nested: n.origin.z == 0 (zero-init)");
    check(n_global.weight   == 5,  "nested: n.weight == 5");

    /* Local nested designation */
    struct nested n_local = { .origin.x = 11, .origin.z = 33, .weight = 7 };
    check(n_local.origin.x == 11, "nested local: n.origin.x == 11");
    check(n_local.origin.y == 0,  "nested local: n.origin.y == 0 (zero-init)");
    check(n_local.origin.z == 33, "nested local: n.origin.z == 33");
    check(n_local.weight   == 7,  "nested local: n.weight == 7");

    /* ===== 3. Array index designation ===== */
    check(arr_global[0] == 100, "array: arr[0] == 100");
    check(arr_global[1] == 0,   "array: arr[1] == 0 (zero-init)");
    check(arr_global[2] == 0,   "array: arr[2] == 0 (zero-init)");
    check(arr_global[3] == 30,  "array: arr[3] == 30");
    check(arr_global[4] == 0,   "array: arr[4] == 0 (zero-init)");
    check(arr_global[5] == 0,   "array: arr[5] == 0 (zero-init)");
    check(arr_global[6] == 0,   "array: arr[6] == 0 (zero-init)");
    check(arr_global[7] == 70,  "array: arr[7] == 70");
    check(arr_global[8] == 0,   "array: arr[8] == 0 (zero-init)");
    check(arr_global[9] == 0,   "array: arr[9] == 0 (zero-init)");

    /* Local array index designation */
    int arr_local[6] = { [5] = 500, [1] = 100 };
    check(arr_local[0] == 0,   "array local: arr[0] == 0 (zero-init)");
    check(arr_local[1] == 100, "array local: arr[1] == 100");
    check(arr_local[2] == 0,   "array local: arr[2] == 0 (zero-init)");
    check(arr_local[3] == 0,   "array local: arr[3] == 0 (zero-init)");
    check(arr_local[4] == 0,   "array local: arr[4] == 0 (zero-init)");
    check(arr_local[5] == 500, "array local: arr[5] == 500");

    /* ===== 4. Brace elision with designated initializers ===== */
    /*
     * The C standard allows eliding inner braces when the initializer
     * list can be unambiguously associated with nested aggregates.
     * Here we combine designated and positional initializers with
     * brace elision for an array of pairs.
     */
    struct pair_array pa = {
        .items[0] = { 1, 2 },
        .items[1] = { 3, 4 },
        .items[2] = { 5, 6 },
        .count = 3
    };
    check(pa.items[0].a == 1, "brace elision: items[0].a == 1");
    check(pa.items[0].b == 2, "brace elision: items[0].b == 2");
    check(pa.items[1].a == 3, "brace elision: items[1].a == 3");
    check(pa.items[1].b == 4, "brace elision: items[1].b == 4");
    check(pa.items[2].a == 5, "brace elision: items[2].a == 5");
    check(pa.items[2].b == 6, "brace elision: items[2].b == 6");
    check(pa.count == 3,      "brace elision: count == 3");

    /* Brace-elided flat initializer for an array of structs:
     * Inner braces for each struct element are omitted; values flow
     * sequentially through the nested aggregate members. */
    struct point pts[3] = { 10, 20, 30, 40, 50, 60 };
    check(pts[0].x == 10, "brace elision flat: pts[0].x == 10");
    check(pts[0].y == 20, "brace elision flat: pts[0].y == 20");
    check(pts[0].z == 30, "brace elision flat: pts[0].z == 30");
    check(pts[1].x == 40, "brace elision flat: pts[1].x == 40");
    check(pts[1].y == 50, "brace elision flat: pts[1].y == 50");
    check(pts[1].z == 60, "brace elision flat: pts[1].z == 60");
    check(pts[2].x == 0,  "brace elision flat: pts[2].x == 0 (zero-init)");
    check(pts[2].y == 0,  "brace elision flat: pts[2].y == 0 (zero-init)");
    check(pts[2].z == 0,  "brace elision flat: pts[2].z == 0 (zero-init)");

    /* ===== 5. Implicit zero-initialization of unspecified members ===== */
    check(q_global.x == 1, "zero-init: q.x == 1");
    check(q_global.y == 0, "zero-init: q.y == 0");
    check(q_global.z == 0, "zero-init: q.z == 0");

    /* Local zero-init */
    struct point q_local = { .y = 77 };
    check(q_local.x == 0,  "zero-init local: q.x == 0");
    check(q_local.y == 77, "zero-init local: q.y == 77");
    check(q_local.z == 0,  "zero-init local: q.z == 0");

    /* Array zero-init: only one element designated, rest must be 0 */
    int sparse[8] = { [4] = 44 };
    check(sparse[0] == 0,  "zero-init array: sparse[0] == 0");
    check(sparse[1] == 0,  "zero-init array: sparse[1] == 0");
    check(sparse[2] == 0,  "zero-init array: sparse[2] == 0");
    check(sparse[3] == 0,  "zero-init array: sparse[3] == 0");
    check(sparse[4] == 44, "zero-init array: sparse[4] == 44");
    check(sparse[5] == 0,  "zero-init array: sparse[5] == 0");
    check(sparse[6] == 0,  "zero-init array: sparse[6] == 0");
    check(sparse[7] == 0,  "zero-init array: sparse[7] == 0");

    /* ===== 6. GCC range designator [lo ... hi] ===== */
    check(range_global[0] == 0,  "range: range[0] == 0 (zero-init)");
    check(range_global[1] == 0,  "range: range[1] == 0 (zero-init)");
    check(range_global[2] == 42, "range: range[2] == 42");
    check(range_global[3] == 42, "range: range[3] == 42");
    check(range_global[4] == 42, "range: range[4] == 42");
    check(range_global[5] == 42, "range: range[5] == 42");
    check(range_global[6] == 0,  "range: range[6] == 0 (zero-init)");
    check(range_global[7] == 0,  "range: range[7] == 0 (zero-init)");
    check(range_global[8] == 0,  "range: range[8] == 0 (zero-init)");
    check(range_global[9] == 0,  "range: range[9] == 0 (zero-init)");

    /* Local range designator */
    int r_local[6] = { [0 ... 2] = 10, [4 ... 5] = 20 };
    check(r_local[0] == 10, "range local: r[0] == 10");
    check(r_local[1] == 10, "range local: r[1] == 10");
    check(r_local[2] == 10, "range local: r[2] == 10");
    check(r_local[3] == 0,  "range local: r[3] == 0 (zero-init)");
    check(r_local[4] == 20, "range local: r[4] == 20");
    check(r_local[5] == 20, "range local: r[5] == 20");

    /* ===== 7. Mixed-type struct designation with array sub-designators ===== */
    check(cfg_global.id    == 99,   "config: id == 99");
    check(cfg_global.value > 3.13 && cfg_global.value < 3.15,
          "config: value ~= 3.14");
    check(strcmp(cfg_global.name, "test") == 0, "config: name == \"test\"");
    check(cfg_global.flags[0] == 1, "config: flags[0] == 1");
    check(cfg_global.flags[1] == 0, "config: flags[1] == 0 (zero-init)");
    check(cfg_global.flags[2] == 0, "config: flags[2] == 0 (zero-init)");
    check(cfg_global.flags[3] == 8, "config: flags[3] == 8");

    /* ===== 8. Deep nesting (.inner.origin.x) ===== */
    check(deep_global.inner.origin.x == 100, "deep: inner.origin.x == 100");
    check(deep_global.inner.origin.y == 0,   "deep: inner.origin.y == 0 (zero-init)");
    check(deep_global.inner.origin.z == 300, "deep: inner.origin.z == 300");
    check(deep_global.inner.weight   == 50,  "deep: inner.weight == 50");
    check(deep_global.tag            == 7,   "deep: tag == 7");

    /* ===== 9. Designation overriding earlier designation ===== */
    struct point ovr = { .x = 10, .y = 20, .z = 30, .x = 99 };
    check(ovr.x == 99, "override: x overridden to 99");
    check(ovr.y == 20, "override: y remains 20");
    check(ovr.z == 30, "override: z remains 30");

    /* ===== 10. Array with mixed positional and designated init ===== */
    int mixed[5] = { 1, 2, [4] = 50 };
    check(mixed[0] == 1,  "mixed: mixed[0] == 1");
    check(mixed[1] == 2,  "mixed: mixed[1] == 2");
    check(mixed[2] == 0,  "mixed: mixed[2] == 0 (zero-init)");
    check(mixed[3] == 0,  "mixed: mixed[3] == 0 (zero-init)");
    check(mixed[4] == 50, "mixed: mixed[4] == 50");

    /* ===== Summary ===== */
    if (failures == 0) {
        printf("All designated initializer tests passed.\n");
    } else {
        printf("%d designated initializer test(s) FAILED.\n", failures);
    }

    return failures;
}
