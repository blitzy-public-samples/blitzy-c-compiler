// Regression test: Empty struct member (bare semicolons inside struct body)
// Bug: Bare ';' inside struct body was treated as a member, causing layout
// errors and miscompilation (e.g., struct { int x; ; int y; } had y at
// wrong offset).
#include <stdio.h>

struct S1 {
    int x;
    ;  // bare semicolon — must be ignored
    int y;
};

struct S2 {
    ;
    ;
    int val;
    ;
};

int main(void) {
    struct S1 s1 = { 10, 20 };
    struct S2 s2 = { 42 };

    int ok = 1;

    // Verify struct S1 layout
    if (s1.x != 10) ok = 0;
    if (s1.y != 20) ok = 0;

    // Verify struct S2 layout
    if (s2.val != 42) ok = 0;

    // Verify sizes are correct (no phantom members)
    if (sizeof(struct S1) != 2 * sizeof(int)) ok = 0;
    if (sizeof(struct S2) != sizeof(int)) ok = 0;

    if (ok) printf("empty_struct_member OK\n");
    else printf("FAIL\n");
    return !ok;
}
