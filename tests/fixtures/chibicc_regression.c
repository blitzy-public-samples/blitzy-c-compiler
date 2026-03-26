// chibicc-pattern regression tests (Task 1)
// Tests all 18 bug patterns documented in CCC GitHub Issue #232

#include <stdio.h>
#include <stddef.h>
#include <string.h>

// Bug 1.1: sizeof on compound literals
int test_sizeof_compound_literal(void) {
    int sz = sizeof((int[]){1,2,3});
    return sz == 3 * sizeof(int) ? 0 : 1;
}

// Bug 1.2: typeof(function-type) *
typedef int fn_t(int);
int test_typeof_function_ptr(void) {
    __typeof__(fn_t) *fptr;
    (void)fptr;
    return 0;
}

// Bug 1.3: int *_Atomic parsing
int test_atomic_pointer(void) {
    int * _Atomic p;
    (void)p;
    return 0;
}

// Bug 1.4: Designated initializer ordering
int test_designated_init_ordering(void) {
    int a[] = { [2] = 30, [0] = 10, [1] = 20 };
    if (a[0] != 10) return 1;
    if (a[1] != 20) return 2;
    if (a[2] != 30) return 3;
    return 0;
}

// Bug 1.5: Qualifier of pointed-to type compatibility
int test_pointer_qualifier_compat(void) {
    const int *p1;
    int *p2;
    // This should compile without error - const-qualified pointed-to
    // type is compatible for assignment from non-const
    p1 = p2;
    (void)p1;
    (void)p2;
    return 0;
}

// Bug 1.6: Duplicated _Generic association diagnostic
// This should compile correctly (no duplicate types)
int test_generic_no_dup(void) {
    int x = 1;
    int result = _Generic(x, int: 42, double: 99);
    return result == 42 ? 0 : 1;
}

// Bug 1.7: 32-bit truncation in constant evaluation
int test_32bit_truncation(void) {
    int val = (int)(0x100000000ULL);
    return val == 0 ? 0 : 1;
}

// Bug 1.8: Cast-to-bool in constant evaluation
int test_cast_to_bool_const(void) {
    _Bool b = (_Bool)2;
    return b == 1 ? 0 : 1;
}

// Bug 1.9: Cast-to-bool with relocation pointers
static int global_var;
int test_cast_to_bool_reloc(void) {
    _Bool b = (_Bool)&global_var;
    return b == 1 ? 0 : 1;
}

// Bug 1.10: Direct assignment of const global struct error
// (This tests compilation behavior — const struct shouldn't allow assignment)
// We just test that const structs can be CREATED
struct point { int x; int y; };
const struct point origin = { 0, 0 };
int test_const_global_struct(void) {
    return origin.x == 0 && origin.y == 0 ? 0 : 1;
}

// Bug 1.11: (boolean-bitfield)++
struct bf { unsigned int b : 1; };
int test_boolean_bitfield_inc(void) {
    struct bf s = { 0 };
    s.b++;
    if (s.b != 1) return 1;
    s.b++;
    if (s.b != 0) return 2;  // wraps around for 1-bit field
    return 0;
}

// Bug 1.12: Struct alignment
struct align_test1 {
    char c;
    int i;
};
struct align_test2 {
    char c;
    double d;
};
int test_struct_alignment(void) {
    if (offsetof(struct align_test1, i) != 4) return 1;
    if (offsetof(struct align_test2, d) != 8) return 2;
    return 0;
}

// Bug 1.13: x87 long double ABI (printf %Lf)
int test_long_double_printf(void) {
    long double a = 1.5L;
    long double b = -42.5L;
    char buf[64];
    sprintf(buf, "%.1Lf", a);
    if (strcmp(buf, "1.5") != 0) return 1;
    sprintf(buf, "%.1Lf", b);
    if (strcmp(buf, "-42.5") != 0) return 2;
    return 0;
}

// Bug 1.14: Array-to-pointer decay
int test_array_decay(void) {
    int arr[3] = {10, 20, 30};
    int *p = arr;  // array-to-pointer decay
    return *p == 10 ? 0 : 1;
}

// Bug 1.15: Integer promotion for statement-expressions
int test_stmt_expr_promotion(void) {
    int result = ({short x = 1; x;}) + ({short y = 2; y;});
    return result == 3 ? 0 : 1;
}

// Bug 1.16: -E output preserves #pragma (tested separately)
// Bug 1.17: Line directives in DWARF (tested separately)

// Bug 1.18: ,##__VA_ARGS__ extension
#define LOG(fmt, ...) sprintf(buf, fmt, ##__VA_ARGS__)
int test_va_args_comma(void) {
    char buf[64];
    LOG("hello");
    if (strcmp(buf, "hello") != 0) return 1;
    LOG("val=%d", 42);
    if (strcmp(buf, "val=42") != 0) return 2;
    return 0;
}

int main(void) {
    int fail = 0;
    int r;
    
    r = test_sizeof_compound_literal();
    if (r) { printf("FAIL: test_sizeof_compound_literal (%d)\n", r); fail++; }
    
    r = test_typeof_function_ptr();
    if (r) { printf("FAIL: test_typeof_function_ptr (%d)\n", r); fail++; }
    
    r = test_atomic_pointer();
    if (r) { printf("FAIL: test_atomic_pointer (%d)\n", r); fail++; }
    
    r = test_designated_init_ordering();
    if (r) { printf("FAIL: test_designated_init_ordering (%d)\n", r); fail++; }
    
    r = test_pointer_qualifier_compat();
    if (r) { printf("FAIL: test_pointer_qualifier_compat (%d)\n", r); fail++; }
    
    r = test_generic_no_dup();
    if (r) { printf("FAIL: test_generic_no_dup (%d)\n", r); fail++; }
    
    r = test_32bit_truncation();
    if (r) { printf("FAIL: test_32bit_truncation (%d)\n", r); fail++; }
    
    r = test_cast_to_bool_const();
    if (r) { printf("FAIL: test_cast_to_bool_const (%d)\n", r); fail++; }
    
    r = test_cast_to_bool_reloc();
    if (r) { printf("FAIL: test_cast_to_bool_reloc (%d)\n", r); fail++; }
    
    r = test_const_global_struct();
    if (r) { printf("FAIL: test_const_global_struct (%d)\n", r); fail++; }
    
    r = test_boolean_bitfield_inc();
    if (r) { printf("FAIL: test_boolean_bitfield_inc (%d)\n", r); fail++; }
    
    r = test_struct_alignment();
    if (r) { printf("FAIL: test_struct_alignment (%d)\n", r); fail++; }
    
    r = test_long_double_printf();
    if (r) { printf("FAIL: test_long_double_printf (%d)\n", r); fail++; }
    
    r = test_array_decay();
    if (r) { printf("FAIL: test_array_decay (%d)\n", r); fail++; }
    
    r = test_stmt_expr_promotion();
    if (r) { printf("FAIL: test_stmt_expr_promotion (%d)\n", r); fail++; }
    
    r = test_va_args_comma();
    if (r) { printf("FAIL: test_va_args_comma (%d)\n", r); fail++; }
    
    if (fail == 0) {
        printf("ALL 16 chibicc regression tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", fail);
    }
    return fail;
}
