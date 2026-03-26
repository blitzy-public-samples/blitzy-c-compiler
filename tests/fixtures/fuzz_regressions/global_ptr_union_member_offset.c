// Bug H regression test: Global pointer initializer through union->struct->field chain.
// The relocation addend must include the field offset within the inner struct.
// Previously BCC emitted addend=0 because infer_struct_type_from_expr() did not
// handle CType::Union when resolving member access types, so the field offset
// was lost (f3 at byte 12 was treated as byte 0).
#include <stdio.h>
#include <stdint.h>

struct S0 {
    const int32_t f0;
    int32_t f1;
    uint32_t f2;
    uint16_t f3;
};

union U3 {
    struct S0 f0;
    const uint8_t f1;
};

static union U3 g_38[1] = {{{1, 0xB9CEC02D, 4, 2}}};
static uint16_t *g_169 = &g_38[0].f0.f3;

// Also test nested struct -> union -> struct chain
struct Outer {
    int a;
    union {
        struct { int x; int y; int z; } inner;
        long long raw;
    } u;
};

static struct Outer g_outer = { 100, { .inner = { 10, 20, 30 } } };
static int *g_ptr_z = &g_outer.u.inner.z;

int main(void) {
    int errors = 0;

    // Test 1: write through global pointer to union->struct->field
    *g_169 = 0x1234;
    if (g_38[0].f0.f3 != 0x1234) {
        printf("FAIL: g_38[0].f0.f3 = %u, expected %u\n",
               (unsigned)g_38[0].f0.f3, 0x1234u);
        errors++;
    }

    // Test 2: read through global pointer to nested struct->union->struct->field
    if (*g_ptr_z != 30) {
        printf("FAIL: *g_ptr_z = %d, expected 30\n", *g_ptr_z);
        errors++;
    }

    // Test 3: write through the pointer and verify
    *g_ptr_z = 999;
    if (g_outer.u.inner.z != 999) {
        printf("FAIL: g_outer.u.inner.z = %d, expected 999\n", g_outer.u.inner.z);
        errors++;
    }

    if (errors == 0) {
        printf("OK: global pointer union member offset correct\n");
    }
    return errors;
}
