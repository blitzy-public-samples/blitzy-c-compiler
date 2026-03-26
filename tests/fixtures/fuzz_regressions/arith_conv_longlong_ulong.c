// Bug F regression test: usual arithmetic conversions must handle
// LongLong vs ULong correctly on LP64 (where both are 64-bit).
// Per C11 6.3.1.8: if unsigned long and signed long long have same
// size, both convert to unsigned long long.
#include <stdio.h>

int main(void) {
    unsigned long ul = 0xFFFFFFFFFFFFFFFFUL;  // max unsigned long
    long long sll = -1LL;
    
    // On LP64: both are 64-bit, so usual arithmetic conversion
    // should convert both to unsigned long long
    // ul == (unsigned long long)sll should be true
    int result = (ul == sll);  // Must be 1 (both become 0xFFFFFFFFFFFFFFFF as ULL)
    
    if (result != 1) {
        printf("FAIL: ul==sll gave %d, expected 1\n", result);
        return 1;
    }
    
    // Also test: unsigned long + long long should produce unsigned long long
    unsigned long a = 10;
    long long b = -3;
    // In unsigned: a + b = 10 + 0xFFFFFFFFFFFFFFFD = 7 (wrapping)
    unsigned long long sum = a + b;
    if (sum != 7) {
        printf("FAIL: sum=%llu, expected 7\n", sum);
        return 1;
    }
    
    printf("arith_conv_longlong_ulong OK\n");
    return 0;
}
