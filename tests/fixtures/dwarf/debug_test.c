/*
 * tests/fixtures/dwarf/debug_test.c
 *
 * DWARF v4 debug information test fixture for BCC Checkpoint 4 validation.
 *
 * Purpose:
 *   This file exercises DWARF v4 debug information generation across multiple
 *   dimensions: subprogram entries, variable entries with diverse types, struct
 *   and member tags, array types, pointer types, and source-to-line mapping
 *   through varied control flow constructs.
 *
 * Validation contract:
 *   - Compiled with -g -O0: output MUST contain .debug_info, .debug_abbrev,
 *     .debug_line, and .debug_str sections with correct DW_TAG_compile_unit,
 *     DW_TAG_subprogram, DW_TAG_variable, DW_TAG_structure_type, and
 *     DW_TAG_member entries.
 *   - Compiled without -g: output MUST contain ZERO .debug_* sections
 *     (no debug leakage).
 *
 * Standard: C11 (ISO/IEC 9899:2011) — no GCC extensions used.
 */

/* --------------------------------------------------------------------
 * File-scope struct definition
 * Exercises: DW_TAG_structure_type, DW_TAG_member
 * -------------------------------------------------------------------- */
struct point {
    int x;
    int y;
};

/* --------------------------------------------------------------------
 * Global variable
 * Exercises: file-scope DW_TAG_variable
 * -------------------------------------------------------------------- */
static int global_counter = 42;

/* --------------------------------------------------------------------
 * Function: add
 * Exercises: DW_TAG_subprogram with parameters and return value
 *            DW_TAG_variable for local result
 *            Basic arithmetic on a distinct source line
 * -------------------------------------------------------------------- */
int add(int a, int b) {
    int result = a + b;
    return result;
}

/* --------------------------------------------------------------------
 * Function: factorial
 * Exercises: DW_TAG_subprogram for recursive function
 *            if/else control flow for .debug_line mapping
 *            DW_TAG_variable for parameter and locals
 * -------------------------------------------------------------------- */
int factorial(int n) {
    if (n <= 1) {
        return 1;
    } else {
        int prev = factorial(n - 1);
        return n * prev;
    }
}

/* --------------------------------------------------------------------
 * Function: process_array
 * Exercises: DW_TAG_subprogram with pointer parameter
 *            for-loop control flow for .debug_line mapping
 *            Pointer dereference and index arithmetic
 * -------------------------------------------------------------------- */
void process_array(int *arr, int len) {
    int sum = 0;
    int i;
    for (i = 0; i < len; i++) {
        sum += arr[i];
        arr[i] = sum;
    }
    global_counter += sum;
}

/* --------------------------------------------------------------------
 * Function: compute
 * Exercises: DW_TAG_subprogram with multiple local variables of
 *            diverse types — int, float, double, unsigned long, struct,
 *            pointer, array, char pointer
 *            while loop for .debug_line mapping
 * -------------------------------------------------------------------- */
int compute(int x) {
    /* Integer variable */
    int count = 0;

    /* Floating-point variables */
    float ratio = 3.14f;
    double accumulator = 0.0;

    /* Unsigned long variable */
    unsigned long big_value = 1000000UL;

    /* Struct variable */
    struct point origin;
    origin.x = 0;
    origin.y = 0;

    /* Array variable */
    int arr[10];
    int idx;
    for (idx = 0; idx < 10; idx++) {
        arr[idx] = idx * x;
    }

    /* Pointer variable */
    int *ptr = &arr[0];

    /* String pointer variable */
    char *message = "dwarf_debug_test";

    /* while loop exercising .debug_line across many source lines */
    while (count < x && count < 100) {
        accumulator += (double)arr[count % 10] * ratio;
        origin.x += count;
        origin.y += count * 2;
        big_value += (unsigned long)count;
        count++;
    }

    /* Use ptr and message to ensure they are not optimized away at -O0 */
    if (*ptr > 0) {
        accumulator += 1.0;
    }
    if (message[0] == 'd') {
        accumulator += 2.0;
    }

    /* Combine diverse results into a single return value */
    return (int)accumulator + origin.x + origin.y + (int)(big_value % 1000);
}

/* --------------------------------------------------------------------
 * Function: main
 * Exercises: DW_TAG_subprogram for entry point
 *            Function calls on distinct lines for .debug_line accuracy
 *            Local variables of various types
 *            if/else and for-loop control flow
 *            Returns 0 on success
 * -------------------------------------------------------------------- */
int main(void) {
    /* Local int variables */
    int a = 10;
    int b = 20;
    int sum;

    /* Local float variable */
    float f = 2.5f;

    /* Local double variable */
    double d = 1.0;

    /* Local struct variable */
    struct point p;
    p.x = 5;
    p.y = 10;

    /* Local array variable */
    int data[10];
    int i;

    /* Local pointer variable */
    int *data_ptr = &data[0];

    /* Local char pointer variable */
    char *label = "checkpoint4";

    /* Local unsigned long variable */
    unsigned long counter = 0UL;

    /* Call add — exercises cross-function .debug_line mapping */
    sum = add(a, b);

    /* Call factorial — exercises recursive subprogram entry */
    int fact_result = factorial(5);

    /* Initialize array via for-loop on distinct lines */
    for (i = 0; i < 10; i++) {
        data[i] = i + 1;
    }

    /* Call process_array — exercises pointer parameter subprogram */
    process_array(data, 10);

    /* Call compute — exercises multi-variable subprogram */
    int compute_result = compute(sum);

    /* Use all local variables to prevent dead-store elimination at -O0 */
    if (sum > 0) {
        d += (double)sum * (double)f;
    }

    counter = (unsigned long)fact_result + (unsigned long)compute_result;

    /* Use struct members */
    p.x += sum;
    p.y += fact_result;

    /* Use pointer */
    *data_ptr = (int)counter;

    /* Use label to keep it live */
    if (label[0] != 'c') {
        counter += 1;
    }

    /* Use global to verify file-scope variable is exercised */
    global_counter += p.x + p.y;

    /* Return 0 for success — final line for .debug_line termination */
    return 0;
}
