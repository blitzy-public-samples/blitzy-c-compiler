// Regression test: typeof on array subscript expression
// Bug: __typeof__(arr[i]) resolved to int instead of the actual element type,
// causing min/max macros to truncate 64-bit values when used with array elements.
#include <stdio.h>

#define min(a, b) ({              \
    __typeof__(a) _a = (a);       \
    __typeof__(b) _b = (b);       \
    _a < _b ? _a : _b;           \
})

#define max(a, b) ({              \
    __typeof__(a) _a = (a);       \
    __typeof__(b) _b = (b);       \
    _a > _b ? _a : _b;           \
})

int main(void) {
    int ok = 1;

    // Test 1: long long array subscript
    long long arr_ll[3] = { -8945932830116527018LL, 100LL, 200LL };
    int i = 0;
    long long m1 = min(arr_ll[i], arr_ll[i]);
    if (m1 != -8945932830116527018LL) ok = 0;

    // Test 2: unsigned long long array subscript
    unsigned long long arr_ull[2] = { 15354988195928104115ULL, 1ULL };
    unsigned long long m2 = min(arr_ull[0], arr_ull[1]);
    if (m2 != 1ULL) ok = 0;

    // Test 3: short array subscript (should promote correctly)
    short arr_s[3] = { -5, 10, 3 };
    short m3 = min(arr_s[0], arr_s[2]);
    if (m3 != -5) ok = 0;

    // Test 4: char array subscript
    char arr_c[3] = { 'a', 'z', 'm' };
    char m4 = min(arr_c[0], arr_c[2]);
    if (m4 != 'a') ok = 0;

    // Test 5: max with array subscript
    long long m5 = max(arr_ll[1], arr_ll[2]);
    if (m5 != 200LL) ok = 0;

    // Test 6: typeof on pointer subscript (pointer used as array)
    long long *ptr = arr_ll;
    long long m6 = min(ptr[0], ptr[1]);
    if (m6 != -8945932830116527018LL) ok = 0;

    if (ok) printf("typeof_array_subscript OK\n");
    else printf("FAIL\n");
    return !ok;
}
