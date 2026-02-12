/*
 * foo.c — Shared library exported functions test fixture for BCC Checkpoint 4.
 *
 * Compilation:
 *   ./bcc -fPIC -shared -o libfoo.so foo.c
 *
 * This file is compiled with -fPIC -shared to produce a shared object
 * (libfoo.so). It exercises the following BCC shared library capabilities:
 *
 *   - PIC code generation with GOT-relative data access and PLT call stubs
 *   - .dynsym population with exported (default visibility) symbols
 *   - Symbol visibility filtering: hidden symbols excluded from .dynsym
 *   - .dynamic section generation (DT_SONAME, DT_SYMTAB, DT_STRTAB, etc.)
 *   - .rela.dyn / .rela.plt relocation sections for runtime fixups
 *   - .gnu.hash section for efficient dynamic symbol lookup
 *   - .got / .got.plt sections for position-independent addressing
 *   - .init_array / .fini_array sections via constructor/destructor attributes
 *   - ET_DYN ELF type in the output shared object
 *
 * Expected .dynsym exported symbols:
 *   add, multiply, get_library_name, get_shared_value, compute, shared_value
 *
 * Expected .dynsym absent symbols:
 *   internal_helper (hidden visibility)
 *
 * Used by: tests/checkpoint4_shared_lib.rs
 * Linked by: tests/fixtures/shared_lib/main.c
 */

/* ---------------------------------------------------------------------------
 * Global variable with default visibility.
 *
 * Must appear in .dynsym as a data symbol. When accessed from external code
 * linked against this shared library, the dynamic linker resolves this through
 * a GOT entry, producing a R_*_GLOB_DAT or R_*_COPY relocation in .rela.dyn.
 * ---------------------------------------------------------------------------
 */
int shared_value = 42;

/* ---------------------------------------------------------------------------
 * Hidden visibility function — internal to the shared library.
 *
 * The __attribute__((visibility("hidden"))) directive ensures this symbol is
 * NOT exported in .dynsym. It is only visible within the shared object itself.
 * readelf --dyn-syms libfoo.so must NOT list internal_helper.
 *
 * Under PIC, calls to hidden functions can use PC-relative addressing directly
 * (no PLT indirection needed), since the caller and callee are guaranteed to
 * reside within the same shared object.
 * ---------------------------------------------------------------------------
 */
__attribute__((visibility("hidden")))
int internal_helper(int x)
{
    return x * x;
}

/* ---------------------------------------------------------------------------
 * Exported functions with default visibility.
 *
 * All of the following functions must appear in .dynsym. When called from
 * external executables linked against libfoo.so, each call goes through a PLT
 * stub that performs lazy binding via the GOT.
 * ---------------------------------------------------------------------------
 */

/*
 * add — Basic integer addition.
 *
 * Tests: function call through PLT, basic arithmetic codegen under PIC.
 */
int add(int a, int b)
{
    return a + b;
}

/*
 * multiply — Integer multiplication.
 *
 * Tests: multi-symbol .dynsym population, second exported function validates
 * that the dynamic symbol table correctly handles more than one entry.
 */
int multiply(int a, int b)
{
    return a * b;
}

/*
 * get_library_name — Returns a pointer to a string literal.
 *
 * Tests: .rodata access via GOT in PIC mode. The string literal "libfoo"
 * resides in .rodata, and under -fPIC the address must be loaded through a
 * GOT-relative reference (e.g., RIP-relative via GOT on x86-64, ADRP+LDR
 * via GOT on AArch64, AUIPC+LD via GOT on RISC-V 64).
 */
const char *get_library_name(void)
{
    return "libfoo";
}

/*
 * get_shared_value — Returns the value of the exported global variable.
 *
 * Tests: global variable GOT entry. Accessing 'shared_value' from within the
 * same shared object under PIC still requires a GOT lookup when the symbol has
 * default visibility (it could be interposed at runtime).
 */
int get_shared_value(void)
{
    return shared_value;
}

/*
 * compute — Calls both hidden and visible functions.
 *
 * Tests intra-library call patterns under PIC:
 *   - internal_helper(x): direct PC-relative call (hidden, no PLT)
 *   - add(x, x): may go through PLT (default visibility, interposable)
 *
 * This validates that the code generator correctly differentiates between
 * hidden (direct) and default (PLT-indirect) call targets within the same
 * shared object.
 */
int compute(int x)
{
    return internal_helper(x) + add(x, x);
}

/* ---------------------------------------------------------------------------
 * Constructor and destructor — .init_array / .fini_array validation.
 *
 * These functions are registered in the .init_array and .fini_array ELF
 * sections respectively. The dynamic linker calls lib_init() when the shared
 * library is loaded (e.g., at dlopen or program startup) and lib_fini() when
 * it is unloaded. The function bodies are intentionally empty — the purpose
 * is to verify that BCC correctly emits the .init_array and .fini_array
 * sections with the appropriate function pointer entries.
 * ---------------------------------------------------------------------------
 */
__attribute__((constructor))
void lib_init(void)
{
    /* Intentionally empty — validates .init_array section emission. */
}

__attribute__((destructor))
void lib_fini(void)
{
    /* Intentionally empty — validates .fini_array section emission. */
}
