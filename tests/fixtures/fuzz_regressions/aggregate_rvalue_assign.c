// Bug G regression test: aggregate assignment/init rvalue pointer-vs-data
// confusion. Assignment expressions like (*p = x) return a pointer to the
// lhs, while function calls and va_arg return the struct data itself.
// The lowering must distinguish these two cases correctly.
#include <stdio.h>
#include <stdarg.h>

struct S { int x; int y; };

struct S make_s(int x, int y) {
    struct S s = {x, y};
    return s;
}

struct S g_s = {100, 200};

void check_va_arg(int dummy, ...) {
    __builtin_va_list ap;
    __builtin_va_start(ap, dummy);
    struct S val = __builtin_va_arg(ap, struct S);
    if (val.x != 42 || val.y != 99) {
        printf("FAIL va_arg: x=%d y=%d\n", val.x, val.y);
        __builtin_va_end(ap);
        return;
    }
    printf("va_arg OK: x=%d y=%d\n", val.x, val.y);
    __builtin_va_end(ap);
}

int main(void) {
    // Test 1: assignment returns pointer to lhs
    struct S a, b = {1, 2};
    struct S c = (a = b);
    if (c.x != 1 || c.y != 2) {
        printf("FAIL assign: x=%d y=%d\n", c.x, c.y);
        return 1;
    }
    
    // Test 2: function call returns data
    struct S d = make_s(10, 20);
    if (d.x != 10 || d.y != 20) {
        printf("FAIL funcall: x=%d y=%d\n", d.x, d.y);
        return 1;
    }
    
    // Test 3: va_arg returns data
    struct S va_s = {42, 99};
    check_va_arg(0, va_s);
    
    // Test 4: global struct init
    if (g_s.x != 100 || g_s.y != 200) {
        printf("FAIL global: x=%d y=%d\n", g_s.x, g_s.y);
        return 1;
    }
    
    printf("aggregate_rvalue_assign OK\n");
    return 0;
}
