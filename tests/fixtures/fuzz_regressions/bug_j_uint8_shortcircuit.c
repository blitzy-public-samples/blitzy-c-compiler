/* Bug J: Sub-register truncation for promoted I8/I16 arithmetic.
 * When I8/I16 arithmetic is promoted to 32-bit in x86-64 codegen, the
 * result must be truncated back to the original width. Without truncation,
 * uint8_t 255 + 1 = 256 (non-zero in 32-bit) instead of 0 (correct in
 * 8-bit), causing wrong short-circuit evaluation in && expressions. */
#include <stdio.h>
#include <stdint.h>

int main(void) {
    /* Test 1: uint8_t pre-increment overflow in && short-circuit */
    uint8_t v = 255;
    uint32_t b = 0x79DA46A1;
    int32_t c = 5;
    int r = (++v) && ((b ^= 1) | c);
    /* v wraps to 0, && should short-circuit, b NOT modified, r=0 */
    if (v != 0) return 1;
    if (b != 0x79DA46A1) return 2;
    if (r != 0) return 3;

    /* Test 2: uint8_t pre-increment overflow in if condition */
    uint8_t g = 255;
    int modified = 0;
    if (++g) { modified = 1; }
    if (g != 0) return 4;
    if (modified != 0) return 5;

    /* Test 3: uint16_t pre-increment overflow in && short-circuit */
    uint16_t u16 = 65535;
    int side = 99;
    int r2 = (++u16) && (side = 0);
    if (u16 != 0) return 6;
    if (side != 99) return 7;
    if (r2 != 0) return 8;

    /* Test 4: uint8_t multiplication overflow */
    uint8_t a = 200;
    uint8_t mul_b = 2;
    uint8_t mul_c = a * mul_b; /* 400 -> 144 (truncated to 8 bits) */
    if (mul_c != 144) return 9;

    printf("bug_j_uint8_shortcircuit OK\n");
    return 0;
}
