// Bug E regression test: struct_load_source Store handler clobbered R10
// when loading a 2-eightbyte struct from a global into a local.
// The store path used R10 as scratch but also loaded into R10, causing
// the second eightbyte to contain stale data.
#include <stdio.h>
#include <string.h>

struct TwoEight {
    long a;
    long b;
};

static struct TwoEight g_val = {0x1122334455667788LL, 0xAABBCCDDEEFF0011LL};

int main(void) {
    struct TwoEight local;
    local = g_val;
    if (local.a != 0x1122334455667788LL || local.b != (long)0xAABBCCDDEEFF0011LL) {
        printf("FAIL: a=0x%lx b=0x%lx\n", local.a, local.b);
        return 1;
    }
    // Also test function returning 2-eightbyte struct
    struct TwoEight copy = local;
    if (copy.a != local.a || copy.b != local.b) {
        printf("FAIL copy: a=0x%lx b=0x%lx\n", copy.a, copy.b);
        return 1;
    }
    printf("struct_store_r10_clobber OK\n");
    return 0;
}
