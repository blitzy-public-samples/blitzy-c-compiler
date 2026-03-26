/*
 * tests/fixtures/shared_lib/foo.c — Shared Library Exported Functions
 *
 * Checkpoint 4 validation source file.
 * Compile: ./bcc -fPIC -shared -o libfoo.so foo.c
 *
 * This file exercises BCC's shared library code generation:
 *   - GOT/PLT relocation emission
 *   - .dynamic/.dynsym/.rela.dyn/.rela.plt/.gnu.hash section generation
 *   - Symbol visibility control (default vs hidden)
 *   - PIC (Position-Independent Code) generation
 *   - ET_DYN ELF shared object production
 *   - Intra-library call resolution
 *   - Global data access through GOT
 *   - String literal access from .rodata in PIC mode
 *
 * Expected ELF structure of libfoo.so:
 *   - e_type == ET_DYN
 *   - .dynsym contains: foo_add, foo_multiply, foo_greeting,
 *                        foo_get_global, foo_global_value
 *   - .dynsym does NOT contain: foo_internal_helper (hidden visibility)
 *   - .dynamic section with DT_SYMTAB, DT_STRTAB, etc.
 *   - .rela.dyn and/or .rela.plt sections
 *   - .gnu.hash section
 *   - .got / .got.plt sections
 *   - .plt section
 *   - PT_DYNAMIC program header
 */

/* ------------------------------------------------------------------ */
/* Global variable — exported with default visibility.                */
/* Accessible by consumers of libfoo.so through the GOT.              */
/* ------------------------------------------------------------------ */
int foo_global_value = 42;

/* ------------------------------------------------------------------ */
/* Hidden helper — NOT exported in .dynsym.                           */
/* Uses __attribute__((visibility("hidden"))) to suppress export.     */
/* Called by exported functions to exercise intra-library relocation   */
/* resolution within the shared object.                               */
/* ------------------------------------------------------------------ */
__attribute__((visibility("hidden")))
int foo_internal_helper(int x) {
    return x * 2;
}

/* ------------------------------------------------------------------ */
/* Exported function: integer addition.                               */
/* Explicitly marked with __attribute__((visibility("default")))      */
/* to verify BCC's visibility attribute parsing and ELF emission.     */
/* Appears in .dynsym; callable via PLT by consumers.                 */
/* ------------------------------------------------------------------ */
__attribute__((visibility("default")))
int foo_add(int a, int b) {
    return a + b;
}

/* ------------------------------------------------------------------ */
/* Exported function: integer multiplication.                         */
/* Default visibility (implicit — no attribute needed, but still      */
/* exported). Calls foo_internal_helper() to test intra-library       */
/* call resolution within the .so (hidden symbol relocation).         */
/*                                                                    */
/* Arithmetic: foo_internal_helper(a) * b / 2                         */
/*           = (a * 2) * b / 2                                        */
/*           = a * b                                                  */
/* ------------------------------------------------------------------ */
int foo_multiply(int a, int b) {
    return foo_internal_helper(a) * b / 2;
}

/* ------------------------------------------------------------------ */
/* Exported function: returns string pointer from .rodata.            */
/* Tests GOT-based pointer return and .rodata data section            */
/* relocations in PIC mode.                                           */
/* ------------------------------------------------------------------ */
const char *foo_greeting(void) {
    return "Hello from shared library!";
}

/* ------------------------------------------------------------------ */
/* Exported function: returns global variable value through GOT.      */
/* Tests global data access via the Global Offset Table in PIC mode.  */
/* ------------------------------------------------------------------ */
int foo_get_global(void) {
    return foo_global_value;
}
