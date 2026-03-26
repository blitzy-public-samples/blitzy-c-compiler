/* Bug I: Multi-dim array global initializer brace-elision symbol size.
 * When a multi-dimensional array is initialized with brace elision
 * (e.g., int g[10][4][6] = {1}), the serialized data size must match
 * the declared type size. */
#include <stdio.h>
int g[10][4][6] = {1};
int main(void) {
    printf("sizeof=%lu g[0][0][0]=%d g[1][0][0]=%d g[9][3][5]=%d\n",
           (unsigned long)sizeof(g), g[0][0][0], g[1][0][0], g[9][3][5]);
    /* Expected: sizeof=960 g[0][0][0]=1 g[1][0][0]=0 g[9][3][5]=0 */
    if (sizeof(g) != 960) return 1;
    if (g[0][0][0] != 1) return 2;
    if (g[1][0][0] != 0) return 3;
    if (g[9][3][5] != 0) return 4;
    printf("bug_i_array_brace_elision OK\n");
    return 0;
}
