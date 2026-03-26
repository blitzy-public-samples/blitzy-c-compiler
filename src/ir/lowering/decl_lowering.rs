#![allow(
    clippy::needless_range_loop,
    clippy::only_used_in_recursion,
    clippy::doc_lazy_continuation,
    clippy::same_item_push
)]
//! # Declaration Lowering
//!
//! Handles the conversion of AST declarations into IR constructs:
//!
//! - **Global variables**: Converted to `GlobalVariable` entries in the `IrModule`
//!   with constant initializers (evaluated at compile time)
//! - **Function definitions**: Creates `IrFunction` with entry block, parameter
//!   allocas, local variable allocas, prologue/epilogue
//! - **Function declarations** (extern): Creates `FunctionDeclaration` entries
//! - **Static local variables**: Lowered as globals with internal linkage
//! - **Thread-local variables**: Lowered with `is_thread_local` flag
//!
//! ## Alloca-First Pattern
//!
//! The central architectural decision: ALL local variables start as `alloca`
//! instructions in the function's entry block, regardless of whether they
//! could be registers. This simplifies lowering enormously:
//!
//! 1. Scan function body for local variable declarations
//! 2. Emit alloca for each in the entry block
//! 3. Store initial values (if any) after the allocas
//! 4. The mem2reg pass (Phase 7) promotes eligible allocas to SSA registers
//!
//! ## Dependencies
//! - `crate::ir::*` — IR types, instructions, function, module, builder
//! - `crate::frontend::parser::ast` — AST declaration nodes
//! - `crate::common::*` — Types, diagnostics, target

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::{FxHashMap, FxHashSet};
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::{CType, StructField};
use crate::frontend::parser::ast;
use crate::ir::builder::IrBuilder;
use crate::ir::function::{
    CallingConvention, FunctionParam, IrFunction, Linkage as FnLinkage, Visibility as FnVisibility,
};
use crate::ir::instructions::{BlockId, Instruction, Value};
use crate::ir::module::{
    Constant, FunctionDeclaration, GlobalVariable, IrModule, Linkage, Visibility,
};
use crate::ir::types::IrType;

use super::expr_lowering;
use super::stmt_lowering;

// ---------------------------------------------------------------------------
// Symbol resolution helper
// ---------------------------------------------------------------------------

/// Resolve an AST `Symbol` to its string representation using the name table.
/// Returns an empty string if the symbol index is out of bounds.
#[inline]
pub(super) fn resolve_sym<'a>(
    name_table: &'a [String],
    sym: &crate::common::string_interner::Symbol,
) -> &'a str {
    let idx = sym.as_u32() as usize;
    if idx < name_table.len() {
        &name_table[idx]
    } else {
        ""
    }
}

// ---------------------------------------------------------------------------
// Local variable information struct
// ---------------------------------------------------------------------------

/// Metadata for a local variable discovered during function body scanning.
#[allow(dead_code)]
struct LocalVarInfo {
    name: String,
    c_type: CType,
    is_static: bool,
    has_initializer: bool,
    alignment: Option<usize>,
    span: Span,
    /// For static locals with initializers, store the AST initializer
    /// for compile-time constant evaluation.
    static_init: Option<ast::Initializer>,
    /// For VLAs, the size expression (e.g. `n` in `int v[n]`).
    /// When present, the local is allocated dynamically via StackAlloc
    /// at the point of declaration rather than as a fixed-size alloca
    /// in the entry block.
    vla_size_expr: Option<Box<ast::Expression>>,
}

// ===========================================================================
// Public API — lower_global_variable
// ===========================================================================

/// Lower a global variable declaration/definition to an IR GlobalVariable.
///
/// Extracts the variable name, type, storage class, and initializer from the
/// AST declaration, converts types, evaluates constant initializers, determines
/// linkage/visibility/section/alignment from attributes, and adds the resulting
/// `GlobalVariable` to the `IrModule`.
pub fn lower_global_variable(
    decl: &ast::Declaration,
    module: &mut IrModule,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) {
    let specifiers = &decl.specifiers;
    let storage_class = specifiers.storage_class;
    let attributes = &specifiers.attributes;

    let is_thread_local = matches!(storage_class, Some(ast::StorageClass::ThreadLocal));
    let spec_has_const = specifiers
        .type_qualifiers
        .iter()
        .any(|q| matches!(q, ast::TypeQualifier::Const));

    for init_decl in &decl.declarators {
        // GCC global register variable: `register T *name __asm__("reg");`
        // These bind a C identifier to a specific hardware register (e.g.
        // `tp` on RISC-V for `current`).  They must NOT produce any
        // storage in the ELF output — no .bss, .data, or .comm entry.
        // Instead, store the mapping in `register_globals` so expression
        // lowering can emit register reads/writes.
        if let Some(ref reg_name) = init_decl.asm_register {
            if let Some(var_name) = extract_declarator_name(&init_decl.declarator, name_table) {
                let c_type =
                    resolve_declaration_type(specifiers, &init_decl.declarator, target, name_table);
                module
                    .register_globals
                    .insert(var_name.to_string(), (reg_name.clone(), c_type));
            }
            continue;
        }

        let declarator = &init_decl.declarator;

        // Determine whether the *variable* itself is const.  When the
        // declarator contains a pointer derivation (e.g. `const char *p`),
        // the specifier-level `const` qualifies the pointee, not the
        // pointer variable — so the variable is mutable and must go in
        // `.data`, not `.rodata`.  Only when the declarator adds no pointer
        // (e.g. `const int x`) does the specifier `const` apply to the
        // variable itself.
        let is_const = if declarator.pointer.is_some() {
            false
        } else {
            spec_has_const
        };

        let var_name = match extract_declarator_name(declarator, name_table) {
            Some(name) => name,
            None => continue,
        };

        let c_type = resolve_declaration_type(specifiers, declarator, target, name_table);

        // Declarations whose resolved type is a function type (e.g. using
        // a function typedef: `fs_param_type fs_param_is_bool;`) are function
        // declarations, NOT variable tentative definitions.  They must never
        // produce BSS storage — they are implicitly extern.
        // Use is_function() which resolves through typedefs.
        let is_function_type = crate::common::types::is_function(&c_type);

        let (initializer, is_definition) = if let Some(ref init) = init_decl.initializer {
            let constant = evaluate_initializer_constant(
                init,
                &c_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            );
            (constant, true)
        } else if matches!(storage_class, Some(ast::StorageClass::Extern)) || is_function_type {
            (None, false)
        } else {
            (Some(Constant::ZeroInit), true)
        };

        // Infer array size from initializer when the declaration uses [].
        // For `char data[] = "hello"`, the C type is Array(Char, None) but
        // the actual size is determined by the string literal length + 1
        // (for the null terminator). Similarly for brace-init lists.
        let c_type = match (&c_type, &initializer) {
            (CType::Array(elem, None), Some(Constant::String(bytes))) => {
                // String initializer: bytes already includes null terminator.
                // For wide string literals (wchar_t/char16_t/char32_t), the
                // bytes are multi-byte encoded, so divide by element byte
                // size to get the element count.
                let elem_byte_size = wide_char_elem_size(elem);
                let elem_count = bytes.len() / elem_byte_size;
                CType::Array(elem.clone(), Some(elem_count))
            }
            (CType::Array(elem, None), Some(Constant::Array(elems))) => {
                // Brace-init list: size = number of elements
                CType::Array(elem.clone(), Some(elems.len()))
            }
            (CType::Array(elem, None), Some(Constant::Struct(elems))) => {
                // Struct-style list initializer
                CType::Array(elem.clone(), Some(elems.len()))
            }
            _ => c_type,
        };

        // Resolve forward-referenced struct/union types so the IR type
        // carries the full field list (needed for correct initializer byte
        // layout and struct size computation in the backend).
        let c_type = {
            let mut ct = c_type;
            super::SIZEOF_STRUCT_DEFS.with(|defs| {
                if let Some(ref sd) = *defs.borrow() {
                    resolve_struct_forward_ref(&mut ct, sd);
                }
            });
            ct
        };
        let ir_type = IrType::from_ctype(&c_type, target);

        let linkage = determine_linkage(storage_class, attributes, name_table);
        let _visibility = determine_visibility(attributes, name_table);
        let section = extract_section_attribute(attributes, name_table);
        let alignment = extract_alignment_attribute(attributes, name_table);

        // Post-process: convert any Constant::String values at
        // pointer-typed positions into Constant::GlobalRef pointing to
        // interned string literals in .rodata.  Without this, string
        // literal initializers for `const char *` struct fields would be
        // written inline (as raw bytes) rather than as a pointer +
        // relocation to a .rodata entry.
        let initializer =
            initializer.map(|init| fixup_string_ptrs_in_constant(init, &ir_type, module));

        let var_name_owned = var_name.clone();
        let mut global = GlobalVariable::new(var_name, ir_type, initializer);
        global.is_definition = is_definition;
        global.linkage = linkage;
        global.is_thread_local = is_thread_local;
        global.section = section;
        global.alignment = alignment;
        global.is_constant = is_const;

        // Store the original C type so that expression lowering can
        // recover full struct field information (including field names
        // and function pointer signatures) for member access operations.
        module
            .global_c_types
            .insert(var_name_owned.clone(), c_type.clone());

        // Immediately seed TYPEOF_CONTEXT so that subsequent global
        // declarations can resolve sizeof(prev_global) correctly during
        // Pass 1 (e.g. `int arr[3]; char e[sizeof(arr)];`).
        super::TYPEOF_CONTEXT.with(|ctx| {
            let mut map = ctx.borrow_mut();
            if let Some(ref mut m) = *map {
                m.entry(var_name_owned.clone())
                    .or_insert_with(|| c_type.clone());
            } else {
                let mut m = crate::common::fx_hash::FxHashMap::default();
                m.insert(var_name_owned.clone(), c_type.clone());
                *map = Some(m);
            }
        });

        // Check for __attribute__((alias("target"))) — this variable is
        // an alias for another symbol. Record the mapping so the ELF writer
        // can emit both symbols at the same address. The variable is still
        // added as a non-definition global so that code referencing it
        // can generate relocations; the ELF writer will emit a defined
        // symbol at the target's address instead of an undefined one.
        {
            let decl_attrs = &init_decl.declarator.attributes;
            if let Some(alias_target) = extract_alias_attribute(attributes, name_table)
                .or_else(|| extract_alias_attribute(decl_attrs, name_table))
            {
                module
                    .symbol_aliases
                    .push((var_name_owned.clone(), alias_target));
            }
        }

        module.add_global(global);
    }
}

// ===========================================================================
// String-in-pointer fixup for static initializers
// ===========================================================================

/// Recursively checks whether a `Constant` initializer, when paired with
/// its C-level type, contains data that would require linker relocation:
/// symbol addresses (`GlobalRef`/`GlobalRefOffset`) or string literals at
/// pointer-typed positions.  The check walks into struct fields and array
/// elements to detect nested pointers (e.g. `struct { const char *val; }`
/// inside a union member).
///
/// Used by the union initializer handler to decide whether the member
/// constant must be preserved in structured form (so that relocations
/// survive) rather than being serialized to flat bytes.
fn constant_contains_relocatable(c: &Constant, cty: &CType) -> bool {
    let resolved = crate::common::types::resolve_typedef(cty);
    match (c, resolved) {
        // GlobalRef / GlobalRefOffset always carry relocations.
        (Constant::GlobalRef(_) | Constant::GlobalRefOffset(_, _), _) => true,
        // String at a pointer-typed position = string literal as pointer.
        (Constant::String(_), CType::Pointer(..)) => true,
        // Recurse into struct fields, matching constant fields to C fields.
        (Constant::Struct(fields), CType::Struct { fields: defs, .. }) => fields
            .iter()
            .zip(defs.iter())
            .any(|(f, d)| constant_contains_relocatable(f, &d.ty)),
        // Recurse into array elements.
        (Constant::Array(elems), CType::Array(elem_ty, _)) => elems
            .iter()
            .any(|e| constant_contains_relocatable(e, elem_ty)),
        // For non-matching type combinations (e.g., Array constant at
        // Union type position, or Struct constant at a non-Struct type),
        // fall back to a type-independent deep scan for GlobalRef /
        // GlobalRefOffset entries.  This catches nested unions within
        // structs where the inner union handler already serialized the
        // member to Array form (e.g., JSCFunctionType inside a struct
        // within QuickJS's JSCFunctionListEntry union).
        _ => constant_tree_has_global_ref(c),
    }
}

/// Type-independent recursive check for `GlobalRef` / `GlobalRefOffset`
/// anywhere in a `Constant` tree.  This catches relocatable data that
/// appears in nested structures where the C type and Constant variant
/// don't align (e.g., a union's inner handler already serialized a
/// function pointer member to `Array([GlobalRef])` form).
fn constant_tree_has_global_ref(c: &Constant) -> bool {
    match c {
        Constant::GlobalRef(_) | Constant::GlobalRefOffset(_, _) => true,
        Constant::Struct(fields) => fields.iter().any(constant_tree_has_global_ref),
        Constant::Array(elems) => elems.iter().any(constant_tree_has_global_ref),
        _ => false,
    }
}

/// Walks a `Constant` initializer alongside its C-level type and collects
/// the byte offsets of all relocatable entries (GlobalRef, GlobalRefOffset,
/// String-at-Pointer).
///
/// This is used by the union initializer handler to properly distribute a
/// multi-field struct member across the union's `Array(I64, N)` IR
/// representation, placing relocatable entries at the correct element
/// indices while serializing non-relocatable data to byte-packed integers.
///
/// For example, `struct { uint8_t a; uint8_t b; func_ptr_union cfunc; }`
/// produces relocs = `[(8, GlobalRef("fn"))]` because the function pointer
/// lives at byte offset 8 within the struct (after 2 bytes + 6 padding).
fn collect_reloc_positions_in_constant(
    c: &Constant,
    cty: &CType,
    base_offset: usize,
    type_builder: &TypeBuilder,
    target: &Target,
    relocs: &mut Vec<(usize, Constant)>,
) {
    let resolved = crate::common::types::resolve_typedef(cty);
    match (c, resolved) {
        // Direct relocatable entries.
        (Constant::GlobalRef(_) | Constant::GlobalRefOffset(_, _), _) => {
            relocs.push((base_offset, c.clone()));
        }
        // String literal at a pointer-typed position.
        (Constant::String(_), CType::Pointer(..)) => {
            relocs.push((base_offset, c.clone()));
        }
        // Recurse into struct fields using the computed layout for correct
        // byte offsets (respecting alignment and padding).
        (
            Constant::Struct(fields),
            CType::Struct {
                fields: defs,
                packed,
                aligned,
                ..
            },
        ) => {
            let layout = type_builder.compute_struct_layout_with_fields(defs, *packed, *aligned);
            for (i, fc) in fields.iter().enumerate() {
                if let Some(fl) = layout.fields.get(i) {
                    let ft = if i < defs.len() {
                        &defs[i].ty
                    } else {
                        &CType::Int
                    };
                    collect_reloc_positions_in_constant(
                        fc,
                        ft,
                        base_offset + fl.offset,
                        type_builder,
                        target,
                        relocs,
                    );
                }
            }
        }
        // Recurse into C array elements.
        (Constant::Array(elems), CType::Array(elem_ty, _)) => {
            let elem_size = type_builder.sizeof_type(elem_ty);
            for (i, e) in elems.iter().enumerate() {
                collect_reloc_positions_in_constant(
                    e,
                    elem_ty,
                    base_offset + i * elem_size,
                    type_builder,
                    target,
                    relocs,
                );
            }
        }
        // Inner union already serialized to Array form by the recursive
        // union handler.  Each element is pointer-width.  Check elements
        // for relocatable entries.
        (Constant::Array(elems), CType::Union { .. }) => {
            let ptr_size: usize = if target.is_64bit() { 8 } else { 4 };
            for (i, e) in elems.iter().enumerate() {
                let elem_off = base_offset + i * ptr_size;
                match e {
                    Constant::GlobalRef(_)
                    | Constant::GlobalRefOffset(_, _)
                    | Constant::String(_) => {
                        relocs.push((elem_off, e.clone()));
                    }
                    // A Struct nested inside an already-serialized union
                    // element — recursively scan its fields.
                    Constant::Struct(inner_fields) => {
                        for f in inner_fields {
                            if matches!(f, Constant::GlobalRef(_) | Constant::GlobalRefOffset(_, _))
                            {
                                // This inner struct's relocatable field
                                // occupies the same element slot.
                                relocs.push((elem_off, f.clone()));
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

/// Recursively walks a `Constant` initializer alongside its IR type tree
/// and replaces `Constant::String` values that appear at `IrType::Ptr`
/// positions with `Constant::GlobalRef` entries.  The raw string bytes
/// are interned into the module's string pool (→ `.rodata`), and the
/// resulting label (e.g. `.L.str.N`) is used for the `GlobalRef`.
///
/// This is necessary because `evaluate_initializer_constant` always
/// produces `Constant::String(bytes)` for string literals — correct for
/// `char arr[] = "hello"` (array) but wrong for `const char *p = "hello"`
/// (pointer), where the struct field must hold an address, not inline data.
fn fixup_string_ptrs_in_constant(
    constant: Constant,
    ir_ty: &IrType,
    module: &mut IrModule,
) -> Constant {
    match (&constant, ir_ty) {
        // Core case: a String constant at a Ptr-typed position.
        // Intern the bytes and produce a GlobalRef.
        (Constant::String(bytes), IrType::Ptr) => {
            let id = module.intern_string(bytes.clone());
            let label = format!(".L.str.{}", id);
            Constant::GlobalRef(label)
        }
        // Union pointer member case: a String constant at an I64 or I32
        // position within a union-typed Array.  Union IR types are
        // `Array(I64, N)` (or I32 on 32-bit targets).  When a union
        // member is a pointer type initialized with a string literal,
        // the union handler preserves the String constant (instead of
        // serializing to bytes) so that we can intern it here and
        // produce a GlobalRef for the linker to relocate.
        (Constant::String(bytes), IrType::I64 | IrType::I32) => {
            let id = module.intern_string(bytes.clone());
            let label = format!(".L.str.{}", id);
            Constant::GlobalRef(label)
        }
        // Struct-within-union case: a Struct constant at an I64 or I32
        // position.  This occurs when a union member is a struct type
        // containing pointer fields (e.g., `struct { const char *val; }`).
        // The union handler preserved the Struct constant (instead of
        // serializing to bytes) when it detected nested relocatable data.
        // Recurse into the struct fields, treating each String field as
        // a pointer to be interned as a GlobalRef.
        (Constant::Struct(fields), IrType::I64 | IrType::I32) => {
            let new_fields: Vec<Constant> = fields
                .iter()
                .map(|f| {
                    // Assume every String inside a struct-within-union
                    // is a pointer.  Non-String constants pass through
                    // unchanged because they don't match any fixup case.
                    fixup_string_ptrs_in_constant(f.clone(), &IrType::Ptr, module)
                })
                .collect();
            Constant::Struct(new_fields)
        }
        // Recurse into struct fields.
        (Constant::Struct(fields), IrType::Struct(st)) => {
            let new_fields: Vec<Constant> = fields
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    let ft = st.fields.get(i).unwrap_or(ir_ty);
                    fixup_string_ptrs_in_constant(field.clone(), ft, module)
                })
                .collect();
            Constant::Struct(new_fields)
        }
        // Recurse into array elements.
        (Constant::Array(elements), IrType::Array(elem_ty, _)) => {
            let new_elems: Vec<Constant> = elements
                .iter()
                .map(|e| fixup_string_ptrs_in_constant(e.clone(), elem_ty, module))
                .collect();
            Constant::Array(new_elems)
        }
        // Everything else passes through unchanged.
        _ => constant,
    }
}

// ===========================================================================
// Public API — lower_function_definition
// ===========================================================================

/// Lower a complete function definition: create `IrFunction`, allocate
/// parameters, scan for local variables, create allocas, verify termination,
/// and add the function to the `IrModule`.
///
/// Implements the mandated alloca-first pattern.
pub fn lower_function_definition(
    func_def: &ast::FunctionDefinition,
    module: &mut IrModule,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
    struct_defs: &FxHashMap<String, CType>,
) {
    let specifiers = &func_def.specifiers;
    let declarator = &func_def.declarator;

    let func_name = match extract_declarator_name(declarator, name_table) {
        Some(name) => name,
        None => {
            diagnostics.emit_error(func_def.span, "function definition without a name");
            return;
        }
    };

    // Build the complete function type using the full declarator.
    // This correctly handles complex cases like function-returning-function-pointer:
    //   void (*memdbDlSym(sqlite3_vfs *pVfs, void *p, const char *zSym))(void)
    // where the return type is `void (*)(void)` and the simple approach of only
    // applying top-level pointer from the declarator would yield bare `void`.
    let base_spec_type = resolve_base_type_fast(specifiers, name_table);
    let full_type = apply_declarator_type(base_spec_type.clone(), declarator, name_table);
    let mut return_c_type = if let CType::Function {
        ref return_type, ..
    } = full_type
    {
        return_type.as_ref().clone()
    } else {
        // Fallback: simple pointer application (shouldn't happen for func defs).
        let mut rt = base_spec_type;
        if let Some(ref ptr) = declarator.pointer {
            rt = apply_pointer_layers(rt, ptr);
        }
        rt
    };
    // Resolve forward-referenced struct/union in the return type so that
    // ABI classification can see the actual struct fields (e.g., for
    // small-struct register return).
    resolve_struct_forward_ref(&mut return_c_type, struct_defs);
    let return_ir_type = IrType::from_ctype(&return_c_type, target);

    // Store the C-level return type so `lower_function_call` can recover
    // it even when the callee IR value is typed as opaque `Pointer(Void)`.
    module
        .func_c_return_types
        .insert(func_name.clone(), return_c_type.clone());

    let (param_declarations, is_variadic) = extract_function_params(declarator);

    // --- K&R (old-style) parameter type merging ---
    // For K&R function definitions like `void f(a, b) int a; double b; { ... }`,
    // the parameter list in the declarator only has identifier names (no types).
    // The actual parameter types come from `func_def.old_style_params`.
    // Build a name→CType map from the old-style declarations.
    let knr_param_types: FxHashMap<String, CType> = {
        let mut map = FxHashMap::default();
        for decl in &func_def.old_style_params {
            let base = resolve_base_type_fast(&decl.specifiers, name_table);
            for init_decl in &decl.declarators {
                if let Some(pname) = extract_declarator_name(&init_decl.declarator, name_table) {
                    let full_ty =
                        apply_declarator_type(base.clone(), &init_decl.declarator, name_table);
                    map.insert(pname, full_ty);
                }
            }
        }
        map
    };

    let mut builder = IrBuilder::new();

    // Build parameter list with SSA values, and collect C-level param types
    // for correct implicit conversions in `lower_function_call`.
    let mut ir_params = Vec::with_capacity(param_declarations.len());
    let mut c_param_types = Vec::with_capacity(param_declarations.len());
    for param_decl in &param_declarations {
        let param_name = extract_param_name(param_decl, name_table);
        // If K&R type info exists for this parameter, use it; otherwise fall back
        // to the (possibly empty/int) type from the declarator parameter list.
        let mut param_c_type = if let Some(knr_ty) = knr_param_types.get(&param_name) {
            knr_ty.clone()
        } else {
            resolve_param_type(param_decl, target, name_table)
        };
        // C11 §6.7.6.3p7: Parameter type adjustments — array→pointer, function→pointer.
        param_c_type = match param_c_type {
            CType::Array(elem, _) => {
                CType::Pointer(elem, crate::common::types::TypeQualifiers::default())
            }
            CType::Function { .. } => CType::Pointer(
                Box::new(param_c_type),
                crate::common::types::TypeQualifiers::default(),
            ),
            other => other,
        };
        // Resolve struct forward references so the IR type has the full
        // field layout — required for correct ABI classification (e.g.,
        // 16-byte structs passed in register pairs).
        resolve_struct_forward_ref(&mut param_c_type, struct_defs);
        c_param_types.push(param_c_type.clone());
        let param_ir_type = IrType::from_ctype(&param_c_type, target);
        let param_value = builder.fresh_value();
        ir_params.push(FunctionParam::new(param_name, param_ir_type, param_value));
    }

    // Store C-level parameter types so `lower_function_call` can insert
    // correct implicit conversions (e.g. sign-extend int → long long).
    module
        .func_c_param_types
        .insert(func_name.clone(), c_param_types);

    // Create IrFunction.
    let mut ir_function = IrFunction::new(func_name.clone(), ir_params, return_ir_type.clone());
    ir_function.calling_convention = CallingConvention::C;
    ir_function.is_variadic = is_variadic;
    ir_function.is_definition = true;

    // Collect all attributes from specifiers, function attrs, and declarator.
    let all_attributes = collect_all_attributes(specifiers, &func_def.attributes, declarator);

    // Set linkage.
    let mod_linkage = determine_linkage(specifiers.storage_class, &all_attributes, name_table);
    ir_function.linkage = convert_linkage_to_fn(mod_linkage);

    // Set visibility.
    let mod_vis = determine_visibility(&all_attributes, name_table);
    ir_function.visibility = convert_visibility_to_fn(mod_vis);

    // Set noreturn.
    ir_function.is_noreturn = specifiers
        .function_specifiers
        .iter()
        .any(|fs| matches!(fs, ast::FunctionSpecifier::Noreturn))
        || has_attribute(&all_attributes, "noreturn", name_table);

    // Set section and alignment.
    ir_function.section = extract_section_attribute(&all_attributes, name_table);
    ir_function.alignment = extract_alignment_attribute(&all_attributes, name_table);

    // Merge alignment from prior declarations.
    // Example: `void f(void) __attribute__((aligned(256)));` (declaration)
    //          `void f(void) { ... }` (definition, no attribute)
    // The definition should inherit alignment 256 from the declaration.
    if let Some(&decl_align) = module.func_alignments.get(&func_name) {
        ir_function.alignment = Some(ir_function.alignment.unwrap_or(0).max(decl_align));
    }
    // Store the final alignment in the module map so `__alignof__` can find it.
    if let Some(align) = ir_function.alignment {
        let entry = module.func_alignments.entry(func_name.clone()).or_insert(0);
        *entry = (*entry).max(align);
    }

    // --- Alloca-first: parameter allocation in entry block ---
    // Consume BlockId(0) from the builder so that subsequent create_block()
    // calls return BlockId(1), BlockId(2), etc.  The entry block already
    // exists (created by IrFunction::new) at blocks[0] = BlockId(0).
    let entry_block_id = builder.create_block(); // Returns BlockId(0), bumps next_block to 1
    debug_assert_eq!(entry_block_id.index(), 0);
    builder.set_insert_point(entry_block_id);

    let mut local_vars: FxHashMap<String, Value> = FxHashMap::default();

    allocate_parameters(
        &mut builder,
        &mut ir_function,
        &param_declarations,
        target,
        &mut local_vars,
        name_table,
        &knr_param_types,
    );

    // --- Seed the typeof resolution context with parameter types ---
    // This allows `typeof(param_name)` inside the function body to resolve
    // correctly during the local variable collection pass.
    // SAVE the current TYPEOF_CONTEXT (which has global variable types) so
    // we can restore it after this function's lowering completes.  This
    // prevents parameter names from shadowing global variable types in
    // TYPEOF_CONTEXT for subsequent functions (e.g. `void f(char *a)` must
    // not permanently overwrite global `const char a[]`).
    let saved_typeof_context = super::TYPEOF_CONTEXT.with(|ctx| ctx.borrow().clone());
    super::TYPEOF_CONTEXT.with(|ctx| {
        let mut map = ctx.borrow_mut().take().unwrap_or_default();
        for param_decl in &param_declarations {
            let pname = extract_param_name(param_decl, name_table);
            if !pname.is_empty() {
                // For K&R-style functions, use the K&R type map.
                let ptype = if let Some(knr_ty) = knr_param_types.get(&pname) {
                    knr_ty.clone()
                } else {
                    resolve_param_type(param_decl, target, name_table)
                };
                map.insert(pname, ptype);
            }
        }
        *ctx.borrow_mut() = Some(map);
    });

    // --- Alloca-first: scan for ALL local variable declarations ---
    let body_stmt = ast::Statement::Compound(func_def.body.clone());
    let mut locals = collect_local_variables(&body_stmt, name_table);

    // Resolve struct/union forward references: when a local variable is
    // declared as `struct S s;` and the struct definition came from a
    // top-level declaration, the collected CType has empty fields.
    // Replace with the full definition from struct_defs.
    for local_info in &mut locals {
        resolve_struct_forward_ref(&mut local_info.c_type, struct_defs);
    }

    // Handle static locals separately — they become globals.
    for local_info in &locals {
        if local_info.is_static {
            lower_static_local(
                &local_info.name,
                &func_name,
                &local_info.c_type,
                local_info.has_initializer,
                local_info.static_init.as_ref(),
                module,
                target,
                type_builder,
                diagnostics,
                enum_constants,
            );
        }
    }

    // --- Save parameter allocas before local variable allocation ---
    // C11 §6.2.1: Parameter names live in the outermost block scope of
    // the function body.  Inner-scope local variables with the same name
    // shadow them only within that inner scope.  The pre-scan
    // `allocate_local_variables` creates allocas for ALL locals (including
    // those in nested scopes) and inserts them into `local_vars`, which
    // would permanently overwrite parameter allocas for same-name locals.
    // We save the parameter mapping here and restore it after the pre-scan
    // so that the function body's top-level scope starts with parameter
    // names correctly mapped to parameter allocas.
    let param_alloca_map: FxHashMap<String, Value> = local_vars.clone();

    // Allocate stack-local variables in entry block.
    let stack_locals: Vec<&LocalVarInfo> = locals.iter().filter(|l| !l.is_static).collect();
    allocate_local_variables(
        &mut builder,
        &mut ir_function,
        &stack_locals,
        target,
        &mut local_vars,
    );

    // --- Restore parameter allocas that were overwritten by locals ---
    // After allocate_local_variables, some parameter names may have been
    // overwritten by same-name local variables from nested scopes.
    // Restore the parameter allocas so the function body's top-level scope
    // sees parameters, not inner-scope locals.  The scope save/restore in
    // lower_compound_statement will handle re-mapping to local allocas
    // when we enter the inner scope where the shadowing declaration lives.
    for (pname, palloca) in &param_alloca_map {
        local_vars.insert(pname.clone(), *palloca);
    }

    ir_function.local_count = local_vars.len() as u32;

    // --- Function prologue (IR-level, minimal) ---
    setup_function_prologue(&mut builder, &mut ir_function);

    // --- Build param_values from allocated parameter allocas ---
    // Parameters were stored in local_vars by allocate_parameters. Create
    // a parallel map so that expression lowering can find parameters.
    let mut param_values: FxHashMap<String, Value> = FxHashMap::default();
    for param_decl in &param_declarations {
        let pname = extract_param_name(param_decl, name_table);
        if std::env::var("BCC_DEBUG_KNR").is_ok() {
            eprintln!(
                "[KNR-PARAM-MAP] pname='{}' in_local_vars={}",
                pname,
                local_vars.contains_key(&pname)
            );
        }
        if !pname.is_empty() {
            if let Some(&alloca_val) = local_vars.get(&pname) {
                param_values.insert(pname, alloca_val);
            }
        }
    }

    // --- Build local_types from collected local variable info ---
    let mut local_types: FxHashMap<String, CType> = FxHashMap::default();
    for local_info in &locals {
        if local_info.vla_size_expr.is_some() {
            // VLA variables are stored as pointers to the element type.
            // This ensures array subscript access (`v[i]`) generates a
            // Load of the pointer followed by GEP, rather than treating
            // the alloca itself as the array base.
            if let CType::Array(ref elem, _) = local_info.c_type {
                local_types.insert(
                    local_info.name.clone(),
                    CType::Pointer(
                        elem.clone(),
                        crate::common::types::TypeQualifiers::default(),
                    ),
                );
            } else {
                local_types.insert(local_info.name.clone(), local_info.c_type.clone());
            }
        } else {
            local_types.insert(local_info.name.clone(), local_info.c_type.clone());
        }
    }
    // Add parameter types as well — resolve struct forward references so
    // that `struct Item *arr` parameter types carry the full struct layout,
    // enabling correct sizeof computation for pointer arithmetic.
    // C11 §6.7.6.3p7: parameter type adjustments (array→pointer, function→pointer).
    // For K&R-style functions, use the K&R type map which has the real types.
    for param_decl in &param_declarations {
        let pname = extract_param_name(param_decl, name_table);
        if !pname.is_empty() {
            let mut ptype = if let Some(knr_ty) = knr_param_types.get(&pname) {
                knr_ty.clone()
            } else {
                resolve_param_type(param_decl, target, name_table)
            };
            // Array-to-pointer / function-to-pointer decay for parameter types.
            ptype = match ptype {
                CType::Array(elem, _) => {
                    CType::Pointer(elem, crate::common::types::TypeQualifiers::default())
                }
                CType::Function { .. } => CType::Pointer(
                    Box::new(ptype),
                    crate::common::types::TypeQualifiers::default(),
                ),
                other => other,
            };
            resolve_struct_forward_ref(&mut ptype, struct_defs);
            local_types.insert(pname, ptype);
        }
    }

    // --- Build static_locals map from collected locals ---
    let mut static_locals: FxHashMap<String, String> = FxHashMap::default();
    for local_info in &locals {
        if local_info.is_static {
            let mangled = format!("{}.{}", func_name, local_info.name);
            static_locals.insert(local_info.name.clone(), mangled);
        }
    }

    // --- Pre-scan for labels to support forward references in asm goto ---
    let mut label_blocks: FxHashMap<String, BlockId> = FxHashMap::default();
    {
        let body_stmt = ast::Statement::Compound(func_def.body.clone());
        let mut label_names: Vec<String> = Vec::new();
        collect_label_names(&body_stmt, &mut label_names, name_table);
        for lname in label_names {
            let block_id = builder.create_block();
            ir_function.ensure_block(block_id);
            label_blocks.insert(lname, block_id);
        }
    }

    // --- Register a forward declaration for the current function ---
    // This allows self-referential patterns (recursion, taking own address,
    // passing self as function pointer) within the function body.  The
    // final `module.add_function()` will add the full definition, but
    // identifier lookup needs to find *something* during body lowering.
    {
        let fwd_param_types: Vec<IrType> =
            ir_function.params.iter().map(|p| p.ty.clone()).collect();
        let fwd_decl = crate::ir::module::FunctionDeclaration::new(
            func_name.clone(),
            return_ir_type.clone(),
            fwd_param_types,
        );
        module.add_declaration(fwd_decl);
    }

    // --- Evaluate VLA parameter dimension side-effects (C11 §6.7.6.2p5) ---
    // When a function parameter is declared as a VLA (e.g. `int arr[n++]`),
    // the dimension expression must be evaluated at function entry for its
    // side effects. The parameter itself decays to a pointer, but `n++` must
    // still execute. We evaluate these expressions here, before the body.
    evaluate_vla_param_side_effects(
        &param_declarations,
        &mut builder,
        &mut ir_function,
        module,
        target,
        type_builder,
        diagnostics,
        &local_vars,
        &param_values,
        name_table,
        &local_types,
        enum_constants,
        &mut static_locals,
        struct_defs,
        &func_name,
    );

    // --- Lower the function body statements ---
    {
        let body_stmt = ast::Statement::Compound(func_def.body.clone());
        let mut stmt_ctx = stmt_lowering::StmtLoweringContext {
            builder: &mut builder,
            function: &mut ir_function,
            module,
            target,
            diagnostics,
            local_vars: &mut local_vars,
            label_blocks: &mut label_blocks,
            loop_stack: Vec::new(),
            switch_ctx: None,
            recursion_depth: 0,
            type_builder,
            param_values: &param_values,
            name_table,
            local_types: &local_types,
            enum_constants,
            static_locals: &mut static_locals,
            struct_defs,
            current_function_name: Some(&func_name),
            scope_type_overrides: FxHashMap::default(),
            return_ctype: Some(return_c_type.clone()),
            layout_cache: FxHashMap::default(),
            vla_sizes: FxHashMap::default(),
            vla_stack_save: None,
            claimed_vars: FxHashSet::default(),
        };
        stmt_lowering::lower_statement(&mut stmt_ctx, &body_stmt);
    }

    // --- DEBUG: Print block info before verify ---
    if std::env::var("BCC_DEBUG_BLOCKS").is_ok() {
        let bc = ir_function.block_count();
        eprintln!("[DEBUG-BLOCKS] Function has {} blocks", bc);
        for idx in 0..bc {
            if let Some(block) = ir_function.get_block(idx) {
                let has_term = block.has_terminator();
                let inst_count = block.instructions.len();
                let last_inst = if inst_count > 0 {
                    format!("{}", block.instructions[inst_count - 1])
                } else {
                    "EMPTY".to_string()
                };
                eprintln!("[DEBUG-BLOCKS]   Block {} (label={:?}): {} insts, terminated={}, succs={:?}, preds={:?}, last_inst={}", 
                    idx, block.label, inst_count, has_term, block.successors(), block.predecessors(), last_inst);
            }
        }
    }
    // --- Verify function termination ---
    verify_function_termination(&mut ir_function, &return_ir_type, &mut builder, diagnostics);

    // Sync value_count from builder.
    ir_function.value_count = builder.fresh_value().0;

    // Restore TYPEOF_CONTEXT to its pre-function state so that parameter
    // types from this function don't leak to subsequent functions.
    super::TYPEOF_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = saved_typeof_context;
    });

    module.add_function(ir_function);
}

// ===========================================================================
// Public API — lower_function_declaration
// ===========================================================================

/// Lower an extern function declaration (no body) to a `FunctionDeclaration`.
pub fn lower_function_declaration(
    decl: &ast::Declaration,
    module: &mut IrModule,
    target: &Target,
    _diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    struct_defs: &FxHashMap<String, CType>,
) {
    let specifiers = &decl.specifiers;

    for init_decl in &decl.declarators {
        let declarator = &init_decl.declarator;

        let func_name = match extract_declarator_name(declarator, name_table) {
            Some(name) => name,
            None => continue,
        };

        let mut return_c_type = resolve_base_type_fast(specifiers, name_table);
        // Apply pointer indirection from the declarator (e.g., `void *foo()`)
        if let Some(ref ptr) = declarator.pointer {
            return_c_type = apply_pointer_layers(return_c_type, ptr);
        }
        // Resolve any forward-referenced struct/union tags in the return
        // type.  Without this, a declaration like `struct S0 func_4(...)`
        // processed before the struct definition would leave the return
        // type as `Struct { fields: [], name: "S0" }` (size 0), causing
        // the backend to omit the sret buffer for MEMORY-class returns.
        resolve_struct_forward_ref(&mut return_c_type, struct_defs);
        let return_ir_type = IrType::from_ctype(&return_c_type, target);

        // Store the C-level return type for `lower_function_call`.
        module
            .func_c_return_types
            .entry(func_name.clone())
            .or_insert_with(|| return_c_type.clone());

        let (param_decls, is_variadic) = extract_function_params(declarator);
        let mut c_param_types_decl = Vec::with_capacity(param_decls.len());
        let param_types: Vec<IrType> = param_decls
            .iter()
            .map(|pd| {
                let mut param_c_type = resolve_param_type(pd, target, name_table);
                // Resolve forward-referenced struct/union tags in parameter
                // types so ABI classification uses the correct struct size.
                resolve_struct_forward_ref(&mut param_c_type, struct_defs);
                c_param_types_decl.push(param_c_type.clone());
                IrType::from_ctype(&param_c_type, target)
            })
            .collect();

        // Store C-level parameter types so `lower_function_call` can insert
        // correct implicit conversions (e.g. sign-extend int → long long).
        module
            .func_c_param_types
            .entry(func_name.clone())
            .or_insert(c_param_types_decl);

        // Collect ALL attributes from specifiers + declarator + init_decl.
        let all_attrs_decl = {
            let mut a = Vec::new();
            a.extend_from_slice(&specifiers.attributes);
            a.extend_from_slice(&declarator.attributes);
            a.extend_from_slice(&init_decl.declarator.attributes);
            a
        };
        let linkage = determine_linkage(specifiers.storage_class, &all_attrs_decl, name_table);
        let visibility = determine_visibility(&all_attrs_decl, name_table);

        // Store function-level alignment override if present.
        // This is needed so that later `__alignof__(func_name)` can return
        // the correct value even when the function definition does not
        // repeat the `aligned` attribute (attribute inheritance from prior
        // declarations — standard C behaviour).
        if let Some(align_val) = extract_alignment_attribute(&all_attrs_decl, name_table) {
            // Store the maximum alignment seen across all declarations.
            let entry = module.func_alignments.entry(func_name.clone()).or_insert(0);
            *entry = (*entry).max(align_val);
        }

        let mut func_decl = FunctionDeclaration::new(func_name, return_ir_type, param_types);
        func_decl.is_variadic = is_variadic;
        func_decl.linkage = linkage;
        func_decl.visibility = visibility;

        module.add_declaration(func_decl);
    }
}

// ===========================================================================
// Public API - lower_local_initializer
// ===========================================================================

/// Lower a local variable initializer and emit store instructions.
///
/// Called after all allocas are created. For scalar types, lowers the
/// initializer expression and stores to the alloca. For aggregates,
/// handles initializer lists with per-element GEP + store.
pub fn lower_local_initializer(
    var_alloca: Value,
    initializer: &ast::Initializer,
    var_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
) {
    match initializer {
        ast::Initializer::Expression(expr) => {
            // ---------------------------------------------------------------
            // Special case: char/unsigned-char array initialized from a
            // string literal — `char a[4] = "abc"`.
            //
            // The expression evaluates to a *pointer* to the .rodata string
            // constant, but C semantics require *copying* the string bytes
            // into the local array (with zero-fill for any remaining
            // elements).  We detect this pattern here and emit byte-by-byte
            // stores (coalesced into word-size stores for efficiency).
            // ---------------------------------------------------------------
            let mut resolved_var_ty = var_type.clone();
            expr_lowering::resolve_forward_ref_type(&mut resolved_var_ty, ctx.struct_defs);

            if let Some((elem_ty, arr_size)) = is_char_array_type(&resolved_var_ty) {
                if let Some(str_bytes) = extract_string_literal_bytes(expr) {
                    // arr_size is in elements; convert to total byte count
                    // for the byte-level copy in lower_char_array_from_string.
                    let elem_bsz = wide_char_elem_size(&elem_ty);
                    let arr_byte_size = arr_size * elem_bsz;
                    lower_char_array_from_string(
                        var_alloca,
                        &str_bytes,
                        arr_byte_size,
                        &elem_ty,
                        ctx,
                    );
                    return;
                }
            }

            // For aggregate types (struct/union) larger than 8 bytes,
            // the backend cannot load/store the entire aggregate through
            // a single register.  Instead we get the source as an lvalue
            // (pointer) and emit chunk-by-chunk load/store pairs, exactly
            // as the assignment path does in expr_lowering::lower_assignment.
            let is_agg = crate::common::types::is_struct_or_union(&resolved_var_ty);
            let struct_size = if is_agg {
                crate::common::types::sizeof_ctype(&resolved_var_ty, ctx.target)
            } else {
                0
            };
            if is_agg && struct_size > 8 {
                // Determine whether the expression is a natural lvalue
                // (has a stable memory address) or an rvalue (e.g.,
                // a function call returning a struct by value).
                let is_natural_lvalue = is_natural_lvalue_expr(expr);

                if is_natural_lvalue {
                    // Get the RHS as an lvalue (pointer) — do NOT dereference.
                    let rhs_ptr = expr_lowering::lower_lvalue(ctx, expr);

                    // Emit individual 8-byte load/store pairs to copy the
                    // aggregate from source to destination.
                    let span = Span::dummy();
                    let mut offset: usize = 0;
                    while offset < struct_size {
                        let remaining = struct_size - offset;
                        let (chunk_sz, chunk_ir) = if remaining >= 8 {
                            (8, IrType::I64)
                        } else if remaining >= 4 {
                            (4, IrType::I32)
                        } else if remaining >= 2 {
                            (2, IrType::I16)
                        } else {
                            (1, IrType::I8)
                        };

                        // Compute source address: rhs_ptr + offset
                        let src_addr = if offset == 0 {
                            rhs_ptr
                        } else {
                            let off_val = make_index_value(ctx, offset as i64);
                            let (gep_val, gep_inst) = ctx.builder.build_gep(
                                rhs_ptr,
                                vec![off_val],
                                chunk_ir.clone(),
                                span,
                            );
                            emit_inst_to_ctx(ctx, gep_inst);
                            gep_val
                        };

                        // Load chunk from source.
                        let (loaded, li) = ctx.builder.build_load(src_addr, chunk_ir.clone(), span);
                        emit_inst_to_ctx(ctx, li);

                        // Compute destination address: var_alloca + offset
                        let dst_addr = if offset == 0 {
                            var_alloca
                        } else {
                            let off_val2 = make_index_value(ctx, offset as i64);
                            let (gep_val2, gep_inst2) = ctx.builder.build_gep(
                                var_alloca,
                                vec![off_val2],
                                chunk_ir.clone(),
                                span,
                            );
                            emit_inst_to_ctx(ctx, gep_inst2);
                            gep_val2
                        };

                        // Store chunk to destination.
                        let si = ctx.builder.build_store(loaded, dst_addr, span);
                        emit_inst_to_ctx(ctx, si);

                        offset += chunk_sz;
                    }
                } else {
                    // Rvalue expression returning a struct (function call,
                    // conditional, statement expression, va_arg, chained
                    // assignment, etc.).
                    let typed_val = expr_lowering::lower_expression_typed(ctx, expr);
                    let span_dummy = Span::dummy();

                    // Check if the expression returns a pointer to the
                    // aggregate data (e.g. chained assignment) vs actual
                    // struct data (function call, va_arg).
                    let is_assign_ptr = expr_returns_aggregate_pointer(expr);

                    let src_ptr = if is_assign_ptr {
                        // Value is already a pointer — use directly.
                        typed_val.value
                    } else {
                        // Value is struct data — spill to a temp alloca
                        // so we can GEP into the memory representation.
                        let struct_ir_ty = expr_lowering::ctype_to_ir(&resolved_var_ty, ctx.target);
                        let (tmp_alloca, tmp_inst) =
                            ctx.builder.build_alloca(struct_ir_ty.clone(), span_dummy);
                        emit_inst_to_ctx(ctx, tmp_inst);
                        let si = ctx
                            .builder
                            .build_store(typed_val.value, tmp_alloca, span_dummy);
                        emit_inst_to_ctx(ctx, si);
                        tmp_alloca
                    };

                    // Copy chunk-by-chunk from source pointer to var_alloca.
                    let mut offset: usize = 0;
                    while offset < struct_size {
                        let remaining = struct_size - offset;
                        let (chunk_sz, chunk_ir) = if remaining >= 8 {
                            (8, IrType::I64)
                        } else if remaining >= 4 {
                            (4, IrType::I32)
                        } else if remaining >= 2 {
                            (2, IrType::I16)
                        } else {
                            (1, IrType::I8)
                        };
                        let src = if offset == 0 {
                            src_ptr
                        } else {
                            let off_val = make_index_value(ctx, offset as i64);
                            let (gep, gi) = ctx.builder.build_gep(
                                src_ptr,
                                vec![off_val],
                                chunk_ir.clone(),
                                span_dummy,
                            );
                            emit_inst_to_ctx(ctx, gi);
                            gep
                        };
                        let (loaded, li) =
                            ctx.builder.build_load(src, chunk_ir.clone(), span_dummy);
                        emit_inst_to_ctx(ctx, li);
                        let dst = if offset == 0 {
                            var_alloca
                        } else {
                            let off_val2 = make_index_value(ctx, offset as i64);
                            let (gep2, gi2) = ctx.builder.build_gep(
                                var_alloca,
                                vec![off_val2],
                                chunk_ir.clone(),
                                span_dummy,
                            );
                            emit_inst_to_ctx(ctx, gi2);
                            gep2
                        };
                        let st = ctx.builder.build_store(loaded, dst, span_dummy);
                        emit_inst_to_ctx(ctx, st);
                        offset += chunk_sz;
                    }
                }
            } else {
                // Scalar or small-aggregate (≤8 bytes): lower the expression
                // and emit a single store.  This handles cases like
                // `char *p = 0;` where the literal 0 has type `int` (I32) but
                // the variable needs a pointer-width (I64) store.
                let typed_val = expr_lowering::lower_expression_typed(ctx, expr);
                let init_val = expr_lowering::insert_implicit_conversion(
                    ctx,
                    typed_val.value,
                    &typed_val.ty,
                    var_type,
                    Span::dummy(),
                );
                let store_inst = ctx.builder.build_store(init_val, var_alloca, Span::dummy());
                emit_inst_to_ctx(ctx, store_inst);
            }
        }
        ast::Initializer::List {
            designators_and_initializers,
            span,
            ..
        } => {
            lower_aggregate_local_init(
                var_alloca,
                designators_and_initializers,
                var_type,
                ctx,
                *span,
            );
        }
    }
}

// ===========================================================================
// Aggregate local initializer lowering
// ===========================================================================

/// Lower an aggregate initializer list (struct or array) into GEP + store
/// instructions for a local variable.
fn lower_aggregate_local_init(
    base_alloca: Value,
    init_list: &[ast::DesignatedInitializer],
    target_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) {
    let resolved = crate::common::types::resolve_typedef(target_type);

    match resolved {
        CType::Array(element_type, size_opt) => {
            let elem_size = ctx.type_builder.sizeof_type(element_type);

            // Detect brace elision: if element type is a struct and ALL
            // initializers are flat scalar expressions (no nested init-lists
            // or designators), we consume multiple scalars per element.
            //
            // CRITICAL: When the element type is a char array (e.g.
            // `char[3]`) and an initializer is a string literal, it
            // should NOT be treated as brace-elided scalars.  The string
            // literal initializes the entire char-array element directly
            // via `lower_single_init_element` → `lower_char_array_from_string`.
            let elem_resolved = crate::common::types::resolve_typedef(element_type);
            let scalars_per_elem = count_aggregate_scalar_fields(elem_resolved, ctx.type_builder);
            let has_any_designators = init_list.iter().any(|di| !di.designators.is_empty());
            let elem_is_char_array = is_char_array_type(elem_resolved).is_some();
            let has_string_literal_init = init_list.iter().any(|di| {
                if let ast::Initializer::Expression(ref expr) = di.initializer {
                    extract_string_literal_bytes(expr).is_some()
                } else {
                    false
                }
            });
            let all_flat_scalars = scalars_per_elem > 1
                && !has_any_designators
                && !(elem_is_char_array && has_string_literal_init)
                && init_list
                    .iter()
                    .all(|di| matches!(di.initializer, ast::Initializer::Expression(_)));

            let array_len = if all_flat_scalars {
                size_opt.unwrap_or((init_list.len() + scalars_per_elem - 1) / scalars_per_elem)
            } else {
                size_opt.unwrap_or(init_list.len())
            };

            // C99 §6.7.9/19: zero-initialize elements that are NOT
            // explicitly covered by the init list.  Skip when all
            // elements are supplied to avoid excessive IR generation.
            let effective_init_count = if all_flat_scalars {
                (init_list.len() + scalars_per_elem - 1) / scalars_per_elem
            } else {
                init_list.len()
            };
            let needs_array_zero = effective_init_count < array_len || has_any_designators;
            if needs_array_zero {
                for i in 0..array_len {
                    let byte_offset = (i * elem_size) as i64;
                    zero_init_field(base_alloca, byte_offset, element_type, ctx, span);
                }
            }

            if all_flat_scalars {
                // Brace elision: consume scalars_per_elem initializers per
                // array element, writing them into the struct fields.
                let mut init_cursor = 0usize;
                for arr_idx in 0..array_len {
                    if init_cursor >= init_list.len() {
                        break;
                    }
                    let byte_offset = (arr_idx * elem_size) as i64;
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (elem_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    // Fill aggregate fields from flat scalars
                    init_cursor = lower_brace_elision_into_aggregate(
                        elem_ptr,
                        init_list,
                        init_cursor,
                        elem_resolved,
                        ctx,
                        span,
                    );
                }
            } else {
                // Normal (non-elision) array init.
                let mut current_idx = 0usize;
                for desig_init in init_list.iter() {
                    // Handle GCC range designator: [low ... high] = value.
                    // A single DesignatedInitializer may cover many elements.
                    if let Some(ast::Designator::IndexRange(ref low_expr, ref high_expr, _)) =
                        desig_init.designators.first()
                    {
                        let low = evaluate_const_int_expr(low_expr).unwrap_or(current_idx);
                        let high = evaluate_const_int_expr(high_expr).unwrap_or(low);
                        for arr_idx in low..=high {
                            if arr_idx >= array_len {
                                break;
                            }
                            let byte_offset = (arr_idx * elem_size) as i64;
                            let idx_val = make_index_value(ctx, byte_offset);
                            let (elem_ptr, gep_inst) = ctx.builder.build_gep(
                                base_alloca,
                                vec![idx_val],
                                IrType::Ptr,
                                span,
                            );
                            emit_inst_to_ctx(ctx, gep_inst);
                            lower_single_init_element(
                                elem_ptr,
                                &desig_init.initializer,
                                element_type,
                                ctx,
                            );
                        }
                        current_idx = high + 1;
                        continue;
                    }

                    let actual_idx = resolve_designator_index(&desig_init.designators, current_idx);
                    if actual_idx >= array_len {
                        break;
                    }
                    let byte_offset = (actual_idx * elem_size) as i64;
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (elem_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    lower_single_init_element(elem_ptr, &desig_init.initializer, element_type, ctx);
                    current_idx = actual_idx + 1;
                }
            }
        }
        CType::Struct {
            ref fields,
            packed: is_packed,
            aligned: explicit_align,
            ..
        } => {
            // Compute full struct layout (handles bitfields properly).
            let layout = ctx.type_builder.compute_struct_layout_with_fields(
                fields,
                *is_packed,
                *explicit_align,
            );
            let field_offsets: Vec<usize> = layout.fields.iter().map(|fl| fl.offset).collect();

            // C99 §6.7.9/19: members not explicitly initialized shall
            // be zero-initialized.  Only emit zero-init stores for
            // fields that are NOT covered by the explicit init list.
            // This avoids generating excessive IR instructions for the
            // common case where all fields are supplied.
            let has_designators = init_list.iter().any(|di| !di.designators.is_empty());

            // Detect struct-level brace elision: when the init list is a
            // flat list of scalar expressions with no designators, and
            // the number of items exceeds the number of struct fields,
            // one or more fields must be aggregate sub-objects (arrays
            // or nested structs) that should consume multiple
            // initializers via brace elision per C99 §6.7.9/20.
            //
            // Also detect: struct has fewer fields than init items, AND
            // at least one field is an aggregate type.  This handles
            // `struct { int f[4]; } s = {1,2,3,4};`.
            let all_flat_exprs = !has_designators
                && init_list
                    .iter()
                    .all(|di| matches!(di.initializer, ast::Initializer::Expression(_)));
            let any_aggregate_field = fields.iter().any(|f| {
                let ft = crate::common::types::resolve_typedef(&f.ty);
                matches!(
                    ft,
                    CType::Struct { .. } | CType::Array(..) | CType::Union { .. }
                )
            });
            let use_brace_elision =
                all_flat_exprs && any_aggregate_field && init_list.len() > fields.len();

            if use_brace_elision {
                // Zero-initialize the entire struct first.
                for (fi, field) in fields.iter().enumerate() {
                    let byte_offset = field_offsets.get(fi).copied().unwrap_or(0) as i64;
                    zero_init_field(base_alloca, byte_offset, &field.ty, ctx, span);
                }
                // Use the recursive brace-elision lowering to consume the
                // correct number of flat initializers per aggregate field.
                lower_brace_elision_into_aggregate(base_alloca, init_list, 0, resolved, ctx, span);
            } else {
                let needs_zero_init = init_list.len() < fields.len() || has_designators;
                if needs_zero_init {
                    for (fi, field) in fields.iter().enumerate() {
                        let byte_offset = field_offsets.get(fi).copied().unwrap_or(0) as i64;
                        zero_init_field(base_alloca, byte_offset, &field.ty, ctx, span);
                    }
                }

                // Now store explicitly initialized fields, handling nested
                // designators (e.g., `.origin.x = 1`).
                // Build a mapping from "init index" to "field index" that
                // skips anonymous bitfields (C11 §6.7.9/9: unnamed members
                // are not initialized by the initializer list).
                let named_field_indices: Vec<usize> = fields
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| {
                        // Skip anonymous bitfields (name is None and it's a bitfield)
                        !(f.name.is_none() && f.bit_width.is_some())
                    })
                    .map(|(i, _)| i)
                    .collect();

                let mut field_cursor = 0usize; // index into named_field_indices
                for desig_init in init_list.iter() {
                    // When a designator is present, it names the target
                    // field explicitly; `default_idx` is irrelevant.
                    let default_field_idx = if desig_init.designators.is_empty() {
                        let fi = named_field_indices
                            .get(field_cursor)
                            .copied()
                            .unwrap_or(field_cursor);
                        field_cursor += 1;
                        fi
                    } else {
                        // Designator will resolve the field — advance
                        // field_cursor to match.
                        let resolved_fi = resolve_field_designator(
                            &desig_init.designators,
                            fields,
                            field_cursor,
                            ctx.name_table,
                        );
                        // Advance cursor past this field (and skip anonymous).
                        if let Some(pos) =
                            named_field_indices.iter().position(|&i| i == resolved_fi)
                        {
                            field_cursor = pos + 1;
                        }
                        resolved_fi
                    };
                    lower_designated_struct_init(
                        base_alloca,
                        &desig_init.designators,
                        &desig_init.initializer,
                        fields,
                        &field_offsets,
                        &layout.fields,
                        0, // base byte offset
                        default_field_idx,
                        ctx,
                        span,
                    );
                }
            }
        }
        CType::Union { ref fields, .. } => {
            // C11 §6.7.9/10: If the aggregate has union type, the first
            // named member is initialized (recursively).  All bytes of
            // the entire union object not explicitly initialized shall
            // be zero-initialized (C11 §6.7.9/21).  We zero-fill the
            // full union first, then overlay any explicit initializer.
            let union_size = ctx.type_builder.sizeof_type(resolved);
            // Zero-fill every byte of the union via I8 stores.
            for byte_idx in 0..union_size {
                let idx_val = make_index_value(ctx, byte_idx as i64);
                let (ptr, gep_inst) =
                    ctx.builder
                        .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                emit_inst_to_ctx(ctx, gep_inst);
                let zero_val = expr_lowering::emit_int_const_for_zero(ctx, IrType::I8);
                let store_inst = ctx.builder.build_store(zero_val, ptr, span);
                emit_inst_to_ctx(ctx, store_inst);
            }
            // Now overlay the explicit initializer if present.
            if let Some(first) = init_list.first() {
                let field_idx =
                    resolve_field_designator(&first.designators, fields, 0, ctx.name_table);
                if field_idx < fields.len() {
                    // When the selected union member is a bitfield, use
                    // bitfield store (read-modify-write with masking)
                    // instead of a plain store.  This ensures only the
                    // bitfield width bits are written, matching GCC/C11
                    // semantics — the remaining union bytes stay zero.
                    if let Some(bw) = fields[field_idx].bit_width {
                        if bw > 0 {
                            let tv = expr_lowering::lower_expression_typed(
                                ctx,
                                match &first.initializer {
                                    ast::Initializer::Expression(e) => e,
                                    _ => {
                                        // Nested list for bitfield — fall back
                                        lower_single_init_element(
                                            base_alloca,
                                            &first.initializer,
                                            &fields[field_idx].ty,
                                            ctx,
                                        );
                                        return;
                                    }
                                },
                            );
                            let val = tv.value;
                            // Use bitfield store: bit_offset=0 (union members
                            // always start at offset 0), bit_width=bw.
                            expr_lowering::lower_bitfield_store(
                                ctx,
                                base_alloca,
                                val,
                                &fields[field_idx].ty,
                                0,
                                bw as usize,
                                span,
                            );
                        }
                    } else if first.designators.len() > 1 {
                        // Chained designator such as `.f.b = 42` — the
                        // first designator (`.f`) selected the union member;
                        // forward remaining designators (`.b`) into the
                        // sub-aggregate initialization.
                        let resolved_member_ty =
                            crate::common::types::resolve_typedef(&fields[field_idx].ty);
                        if let CType::Struct {
                            fields: ref sub_fields,
                            packed: sub_packed,
                            aligned: sub_aligned,
                            ..
                        } = resolved_member_ty
                        {
                            let sub_layout = ctx.type_builder.compute_struct_layout_with_fields(
                                sub_fields,
                                *sub_packed,
                                *sub_aligned,
                            );
                            let inner_offsets: Vec<usize> =
                                sub_layout.fields.iter().map(|fl| fl.offset).collect();
                            lower_designated_struct_init(
                                base_alloca,
                                &first.designators[1..],
                                &first.initializer,
                                sub_fields,
                                &inner_offsets,
                                &sub_layout.fields,
                                0, // union members always start at byte 0
                                0,
                                ctx,
                                span,
                            );
                        } else if let CType::Union { .. } = resolved_member_ty {
                            // Nested union within union — recurse
                            let synthetic_item = ast::DesignatedInitializer {
                                designators: first.designators[1..].to_vec(),
                                initializer: first.initializer.clone(),
                                span: first.span,
                            };
                            let synthetic_init = ast::Initializer::List {
                                designators_and_initializers: vec![synthetic_item],
                                trailing_comma: false,
                                span: first.span,
                            };
                            lower_single_init_element(
                                base_alloca,
                                &synthetic_init,
                                &fields[field_idx].ty,
                                ctx,
                            );
                        } else {
                            // Non-aggregate sub-field — treat as plain store
                            lower_single_init_element(
                                base_alloca,
                                &first.initializer,
                                &fields[field_idx].ty,
                                ctx,
                            );
                        }
                    } else {
                        lower_single_init_element(
                            base_alloca,
                            &first.initializer,
                            &fields[field_idx].ty,
                            ctx,
                        );
                    }
                }
            }
        }
        _ => {
            // Scalar with brace-enclosed init: int x = {42};
            if let Some(first) = init_list.first() {
                lower_single_init_element(base_alloca, &first.initializer, target_type, ctx);
            }
        }
    }
}

/// Lower a single initializer element -- recursively handles nested lists.
fn lower_single_init_element(
    ptr: Value,
    init: &ast::Initializer,
    elem_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
) {
    match init {
        ast::Initializer::Expression(expr) => {
            // ---------------------------------------------------------------
            // Special case: char array member initialized from a string
            // literal — `struct S { char c[8]; }; struct S a = { "hello" };`
            //
            // The expression evaluates to a *pointer* to .rodata, but C
            // semantics require *copying* the string bytes into the member
            // array.  We detect this and emit byte-level stores.
            // ---------------------------------------------------------------
            let mut resolved_elem = elem_type.clone();
            expr_lowering::resolve_forward_ref_type(&mut resolved_elem, ctx.struct_defs);
            if let Some((char_elem_ty, arr_size)) = is_char_array_type(&resolved_elem) {
                if let Some(str_bytes) = extract_string_literal_bytes(expr) {
                    let elem_bsz = wide_char_elem_size(&char_elem_ty);
                    let arr_byte_size = arr_size * elem_bsz;
                    lower_char_array_from_string(
                        ptr,
                        &str_bytes,
                        arr_byte_size,
                        &char_elem_ty,
                        ctx,
                    );
                    return;
                }
            }

            let tv = expr_lowering::lower_expression_typed(ctx, expr);
            // Insert an implicit conversion when the expression value's type
            // differs from the target element type. This handles both
            // narrowing (e.g., `0x80` I32 → `char` I8) and widening
            // (e.g., `-1` I32 → `long` I64 with sign-extension).
            let val = expr_lowering::insert_implicit_conversion(
                ctx,
                tv.value,
                &tv.ty,
                elem_type,
                Span::dummy(),
            );
            let store_inst = ctx.builder.build_store(val, ptr, Span::dummy());
            emit_inst_to_ctx(ctx, store_inst);
        }
        ast::Initializer::List {
            designators_and_initializers,
            span,
            ..
        } => {
            lower_aggregate_local_init(ptr, designators_and_initializers, elem_type, ctx, *span);
        }
    }
}

// ===========================================================================
// Constant initializer evaluation (for globals)
// ===========================================================================

/// Evaluate an AST initializer as a compile-time constant for global variables.
pub(super) fn evaluate_initializer_constant(
    init: &ast::Initializer,
    expected_type: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) -> Option<Constant> {
    match init {
        ast::Initializer::Expression(expr) => evaluate_constant_expr(
            expr,
            expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
            enum_constants,
        ),
        ast::Initializer::List {
            designators_and_initializers,
            ..
        } => lower_designated_initializer(
            designators_and_initializers,
            expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
            enum_constants,
        ),
    }
}

/// Create an anonymous global variable for a compound literal used in a global
/// initializer context (e.g. `struct A *p = &(struct A){10, 20};`).
/// Returns the anonymous global name, or `None` if the compound literal cannot
/// be evaluated as a constant.
fn create_anonymous_compound_literal_global(
    type_name: &ast::TypeName,
    initializer: &ast::Initializer,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) -> Option<String> {
    // Resolve the C type from the type-name.
    let cty_raw = super::resolve_type_name_for_const(type_name, target, type_builder);
    // Resolve forward-referenced struct/union tags to get full field info.
    let cty = {
        let resolved_ref = crate::common::types::resolve_typedef(&cty_raw);
        let maybe_full: Option<CType> = match resolved_ref {
            CType::Struct {
                name: Some(ref tag),
                ref fields,
                ..
            } if fields.is_empty() => super::SIZEOF_STRUCT_DEFS.with(|defs| {
                defs.borrow()
                    .as_ref()
                    .and_then(|m| m.get(tag.as_str()).cloned())
            }),
            CType::Union {
                name: Some(ref tag),
                ref fields,
                ..
            } if fields.is_empty() => super::SIZEOF_STRUCT_DEFS.with(|defs| {
                defs.borrow()
                    .as_ref()
                    .and_then(|m| m.get(tag.as_str()).cloned())
            }),
            _ => None,
        };
        maybe_full.unwrap_or(cty_raw)
    };
    let ir_type = IrType::from_ctype(&cty, target);

    // Evaluate the initializer to a compile-time constant.
    let constant = match initializer {
        ast::Initializer::Expression(expr) => evaluate_constant_expr(
            expr,
            &cty,
            target,
            type_builder,
            diagnostics,
            name_table,
            enum_constants,
        )?,
        ast::Initializer::List {
            designators_and_initializers,
            ..
        } => lower_designated_initializer(
            designators_and_initializers,
            &cty,
            target,
            type_builder,
            diagnostics,
            name_table,
            enum_constants,
        )?,
    };

    // Generate a unique anonymous global name.
    let idx = super::COMPOUND_LITERAL_COUNTER.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    });
    let anon_name = format!("__compound_literal.{}", idx);

    // Push into the thread-local collection for draining after Pass 1.
    super::COMPOUND_LITERAL_GLOBALS.with(|cl| {
        cl.borrow_mut()
            .push((anon_name.clone(), ir_type, constant, cty));
    });

    Some(anon_name)
}

/// Evaluate a compile-time constant expression for a global variable initializer.
fn evaluate_constant_expr(
    expr: &ast::Expression,
    expected_type: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) -> Option<Constant> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(Constant::Integer(*value as i128)),
        ast::Expression::FloatLiteral { value, suffix, .. } => {
            // GCC imaginary suffix: `2.0i` → _Complex constant {0.0, 2.0}
            match suffix {
                ast::FloatSuffix::I | ast::FloatSuffix::FI => Some(Constant::Array(vec![
                    Constant::Float(0.0),
                    Constant::Float(*value),
                ])),
                ast::FloatSuffix::LI => {
                    // _Complex long double imaginary: convert both parts
                    // to long double byte representation.
                    let zero_ld = crate::common::long_double::LongDouble::from_f64(0.0);
                    let val_ld = crate::common::long_double::LongDouble::from_f64(*value);
                    Some(Constant::Array(vec![
                        Constant::LongDouble(zero_ld.to_bytes()),
                        Constant::LongDouble(val_ld.to_bytes()),
                    ]))
                }
                ast::FloatSuffix::L => {
                    // Long double literal (e.g. `2.0L`): convert the f64
                    // value to 80-bit extended precision bytes so the
                    // global initializer emits the correct representation.
                    let ld = crate::common::long_double::LongDouble::from_f64(*value);
                    Some(Constant::LongDouble(ld.to_bytes()))
                }
                _ => Some(Constant::Float(*value)),
            }
        }
        ast::Expression::StringLiteral {
            segments, prefix, ..
        } => {
            // Concatenate all segments for the string value.
            // C string literals include a null terminator — append it
            // so that the byte vector matches the declared array size.
            // Wide string literals (L, U, u) encode each character at the
            // target wchar_t/char16_t/char32_t width.  For wide strings,
            // raw bytes are UTF-8 decoded to code points first so that
            // multi-byte UTF-8 sequences produce a single wide character.
            let char_width: usize = match prefix {
                ast::StringPrefix::None | ast::StringPrefix::U8 => 1,
                ast::StringPrefix::L | ast::StringPrefix::U32 => 4,
                ast::StringPrefix::U16 => 2,
            };
            let mut raw_bytes = Vec::new();
            for seg in segments {
                raw_bytes.extend_from_slice(&seg.value);
            }
            let mut bytes = expand_string_bytes_for_width(&raw_bytes, char_width);
            // C11 §6.7.9p14: "including the terminating null character if
            // there is room or if the array is of unknown size".  When the
            // target is a fixed-size character array and the string
            // (including null terminator) exceeds the array, truncate to
            // exactly the array size so the null doesn't overflow into
            // adjacent storage.
            let resolved_et = crate::common::types::resolve_typedef(expected_type);
            if let CType::Array(ref _elem, Some(arr_len)) = resolved_et {
                let expected_bytes = arr_len * char_width;
                if bytes.len() > expected_bytes {
                    bytes.truncate(expected_bytes);
                }
            }
            Some(Constant::String(bytes))
        }
        ast::Expression::CharLiteral { value, .. } => Some(Constant::Integer(*value as i128)),
        ast::Expression::Identifier { name, .. } => {
            let name_str = resolve_sym(name_table, name).to_string();
            // Check if this identifier is an enum constant — resolve to
            // its integer value rather than emitting a symbol reference.
            if let Some(&val) = enum_constants.get(&name_str) {
                return Some(Constant::Integer(val));
            }
            Some(Constant::GlobalRef(name_str))
        }
        ast::Expression::UnaryOp { op, operand, .. } => match op {
            ast::UnaryOp::AddressOf => {
                // &identifier
                if let ast::Expression::Identifier { name, .. } = operand.as_ref() {
                    let name_str = resolve_sym(name_table, name).to_string();
                    return Some(Constant::GlobalRef(name_str));
                }
                // &array[index] — address of array element
                if let ast::Expression::ArraySubscript { base, index, .. } = operand.as_ref() {
                    if let Some((sym, byte_off)) =
                        evaluate_address_of_subscript(base, index, target, name_table)
                    {
                        if byte_off == 0 {
                            return Some(Constant::GlobalRef(sym));
                        }
                        return Some(Constant::GlobalRefOffset(sym, byte_off));
                    }
                }
                // &struct->member or &ptr[N] chains
                if let Some((sym, byte_off)) =
                    evaluate_address_of_member_chain(operand, target, name_table)
                {
                    if byte_off == 0 {
                        return Some(Constant::GlobalRef(sym));
                    }
                    return Some(Constant::GlobalRefOffset(sym, byte_off));
                }
                // &(CompoundLiteral) — create anonymous global for the compound literal
                // and return a GlobalRef to it.
                if let ast::Expression::CompoundLiteral {
                    type_name,
                    initializer,
                    ..
                } = operand.as_ref()
                {
                    if let Some(anon_name) = create_anonymous_compound_literal_global(
                        type_name,
                        initializer,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        enum_constants,
                    ) {
                        return Some(Constant::GlobalRef(anon_name));
                    }
                }
                Some(Constant::Undefined)
            }
            ast::UnaryOp::Negate => {
                let inner = evaluate_constant_expr(
                    operand,
                    expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                )?;
                match inner {
                    Constant::Integer(v) => Some(Constant::Integer(-v)),
                    Constant::Float(v) => Some(Constant::Float(-v)),
                    _ => Some(Constant::Undefined),
                }
            }
            ast::UnaryOp::BitwiseNot => {
                let inner = evaluate_constant_expr(
                    operand,
                    expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                )?;
                match inner {
                    Constant::Integer(v) => Some(Constant::Integer(!v)),
                    _ => Some(Constant::Undefined),
                }
            }
            ast::UnaryOp::LogicalNot => {
                let inner = evaluate_constant_expr(
                    operand,
                    expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                )?;
                match inner {
                    Constant::Integer(v) => Some(Constant::Integer(if v == 0 { 1 } else { 0 })),
                    _ => Some(Constant::Undefined),
                }
            }
            _ => Some(Constant::Undefined),
        },
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let lhs = evaluate_constant_expr(
                left,
                expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            )?;
            let rhs = evaluate_constant_expr(
                right,
                expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            )?;
            // Handle pointer-arithmetic on string literals and global refs:
            //   "foo" + 1  → GlobalRefOffset(anon_string, 1)
            //   &var + N   → GlobalRefOffset(var, N * elem_size)
            //   GlobalRef + N → GlobalRefOffset(name, N)
            //   GlobalRefOffset + N → GlobalRefOffset(name, off + N)
            match op {
                ast::BinaryOp::Add => {
                    // String + Integer → anonymous string global + offset
                    if let (Constant::String(ref bytes), Constant::Integer(off)) = (&lhs, &rhs) {
                        let anon = create_anonymous_string_global(bytes, target);
                        if *off == 0 {
                            return Some(Constant::GlobalRef(anon));
                        }
                        return Some(Constant::GlobalRefOffset(anon, *off as i64));
                    }
                    if let (Constant::Integer(off), Constant::String(ref bytes)) = (&lhs, &rhs) {
                        let anon = create_anonymous_string_global(bytes, target);
                        if *off == 0 {
                            return Some(Constant::GlobalRef(anon));
                        }
                        return Some(Constant::GlobalRefOffset(anon, *off as i64));
                    }
                    // GlobalRef + Integer → GlobalRefOffset
                    if let (Constant::GlobalRef(ref name), Constant::Integer(off)) = (&lhs, &rhs) {
                        if *off == 0 {
                            return Some(Constant::GlobalRef(name.clone()));
                        }
                        return Some(Constant::GlobalRefOffset(name.clone(), *off as i64));
                    }
                    if let (Constant::Integer(off), Constant::GlobalRef(ref name)) = (&lhs, &rhs) {
                        if *off == 0 {
                            return Some(Constant::GlobalRef(name.clone()));
                        }
                        return Some(Constant::GlobalRefOffset(name.clone(), *off as i64));
                    }
                    // GlobalRefOffset + Integer
                    if let (Constant::GlobalRefOffset(ref name, base_off), Constant::Integer(off)) =
                        (&lhs, &rhs)
                    {
                        let total = base_off + *off as i64;
                        if total == 0 {
                            return Some(Constant::GlobalRef(name.clone()));
                        }
                        return Some(Constant::GlobalRefOffset(name.clone(), total));
                    }
                    if let (Constant::Integer(off), Constant::GlobalRefOffset(ref name, base_off)) =
                        (&lhs, &rhs)
                    {
                        let total = base_off + *off as i64;
                        if total == 0 {
                            return Some(Constant::GlobalRef(name.clone()));
                        }
                        return Some(Constant::GlobalRefOffset(name.clone(), total));
                    }
                }
                ast::BinaryOp::Sub => {
                    // GlobalRef - Integer → GlobalRefOffset with negative offset
                    if let (Constant::GlobalRef(ref name), Constant::Integer(off)) = (&lhs, &rhs) {
                        let neg = -(*off as i64);
                        if neg == 0 {
                            return Some(Constant::GlobalRef(name.clone()));
                        }
                        return Some(Constant::GlobalRefOffset(name.clone(), neg));
                    }
                    if let (Constant::GlobalRefOffset(ref name, base_off), Constant::Integer(off)) =
                        (&lhs, &rhs)
                    {
                        let total = base_off - *off as i64;
                        if total == 0 {
                            return Some(Constant::GlobalRef(name.clone()));
                        }
                        return Some(Constant::GlobalRefOffset(name.clone(), total));
                    }
                    // String - Integer
                    if let (Constant::String(ref bytes), Constant::Integer(off)) = (&lhs, &rhs) {
                        let anon = create_anonymous_string_global(bytes, target);
                        let neg = -(*off as i64);
                        if neg == 0 {
                            return Some(Constant::GlobalRef(anon));
                        }
                        return Some(Constant::GlobalRefOffset(anon, neg));
                    }
                }
                _ => {}
            }
            evaluate_const_binop(op, &lhs, &rhs)
        }
        ast::Expression::Cast {
            operand, type_name, ..
        } => {
            let inner = evaluate_constant_expr(
                operand,
                expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            )?;
            // Apply cast truncation to match C semantics.
            let cast_ty = super::resolve_type_name_for_const(type_name, target, type_builder);
            Some(apply_const_cast(inner, &cast_ty, target))
        }
        ast::Expression::SizeofExpr { .. } | ast::Expression::SizeofType { .. } => Some(
            Constant::Integer(evaluate_sizeof_expr(expr, target, type_builder) as i128),
        ),
        ast::Expression::AlignofType { .. } => {
            Some(Constant::Integer(
                evaluate_alignof_expr(expr, target, type_builder) as i128,
            ))
        }
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let cond = evaluate_constant_expr(
                condition,
                expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            )?;
            match cond {
                Constant::Integer(v) if v != 0 => {
                    if let Some(ref te) = then_expr {
                        evaluate_constant_expr(
                            te,
                            expected_type,
                            target,
                            type_builder,
                            diagnostics,
                            name_table,
                            enum_constants,
                        )
                    } else {
                        // GCC extension: x ?: y — if condition is true, value is condition
                        Some(Constant::Integer(v))
                    }
                }
                Constant::Integer(_) => evaluate_constant_expr(
                    else_expr,
                    expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                ),
                _ => Some(Constant::Undefined),
            }
        }
        ast::Expression::Parenthesized { inner, .. } => evaluate_constant_expr(
            inner,
            expected_type,
            target,
            type_builder,
            diagnostics,
            name_table,
            enum_constants,
        ),
        // BuiltinCall in constant context (e.g. __builtin_constant_p,
        // __builtin_choose_expr) — evaluate known builtins.
        ast::Expression::BuiltinCall { builtin, args, .. } => match builtin {
            ast::BuiltinKind::ConstantP => Some(Constant::Integer(0)),
            ast::BuiltinKind::ChooseExpr if args.len() >= 3 => {
                let cond = evaluate_constant_expr(
                    &args[0],
                    expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                );
                match cond {
                    Some(Constant::Integer(v)) if v != 0 => evaluate_constant_expr(
                        &args[1],
                        expected_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        enum_constants,
                    ),
                    _ => evaluate_constant_expr(
                        &args[2],
                        expected_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        enum_constants,
                    ),
                }
            }
            ast::BuiltinKind::TypesCompatibleP => Some(Constant::Integer(0)),
            ast::BuiltinKind::Expect if !args.is_empty() => evaluate_constant_expr(
                &args[0],
                expected_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            ),
            _ => Some(Constant::Undefined),
        },
        // Comma operator — evaluate both, return last.
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                evaluate_constant_expr(
                    last,
                    expected_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                )
            } else {
                Some(Constant::Undefined)
            }
        }
        // Statement expression ({...}) — treat as non-constant but don't
        // error out if it produces a known value (e.g. kernel macros).
        ast::Expression::StatementExpression { .. } => {
            // Conservatively treat as undefined — not evaluable at
            // compile time in the general case.
            Some(Constant::Undefined)
        }
        // FunctionCall — not a compile-time constant, but don't error
        // (some kernel macros produce these in initializer context).
        ast::Expression::FunctionCall { .. } => Some(Constant::Undefined),
        // MemberAccess and PointerMemberAccess — might be a compile-time
        // address constant if the expression is an array-to-pointer decay
        // on a global struct member (e.g. `g.arr` where arr is an array
        // field inside global struct g).  Use evaluate_address_of_member_chain
        // to resolve the base symbol + byte offset.
        ast::Expression::MemberAccess { .. } | ast::Expression::PointerMemberAccess { .. } => {
            if let Some((sym, byte_off)) =
                evaluate_address_of_member_chain(expr, target, name_table)
            {
                if byte_off == 0 {
                    return Some(Constant::GlobalRef(sym));
                }
                return Some(Constant::GlobalRefOffset(sym, byte_off));
            }
            Some(Constant::Undefined)
        }
        // AddressOfLabel (GCC &&label) — NOT a link-time constant.
        // Label addresses are function-local code addresses that can only
        // be resolved at runtime (via LEA of the assembler-local label).
        // Return None to force runtime initialization for static locals
        // containing label address initializers.
        ast::Expression::AddressOfLabel { .. } => None,
        // ArraySubscript — not compile-time constant.
        ast::Expression::ArraySubscript { .. } => Some(Constant::Undefined),
        // Generic (_Generic) — not evaluated here.
        ast::Expression::Generic { .. } => Some(Constant::Undefined),
        // CompoundLiteral — create anonymous global and return its value.
        ast::Expression::CompoundLiteral {
            type_name,
            initializer,
            ..
        } => {
            if let Some(anon_name) = create_anonymous_compound_literal_global(
                type_name,
                initializer,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            ) {
                Some(Constant::GlobalRef(anon_name))
            } else {
                Some(Constant::Undefined)
            }
        }
        _ => {
            diagnostics.emit_error(
                Span::dummy(),
                "initializer element is not a compile-time constant",
            );
            Some(Constant::Undefined)
        }
    }
}

/// Apply a C-type cast to a constant value, truncating to the correct bit width.
///
/// In C, casts like `(int)(0x100000000ULL)` must truncate to 32 bits, yielding 0.
/// Similarly, `(long long)(val)` truncates to 64 bits.  The constant evaluator
/// works with `i128`, so we must manually apply the target type's width.
fn apply_const_cast(c: Constant, cast_ty: &CType, target: &Target) -> Constant {
    match c {
        Constant::Integer(val) => {
            let truncated = truncate_const_to_ctype(val, cast_ty, target);
            Constant::Integer(truncated)
        }
        // Float → integer or integer → float casts in global initializers:
        // pass through for now (the caller already evaluated the inner expr).
        other => other,
    }
}

/// Truncate an i128 constant to the bit-width of a C type, performing
/// sign-extension for signed types and zero-extension for unsigned types.
fn truncate_const_to_ctype(val: i128, cty: &CType, target: &Target) -> i128 {
    match cty {
        CType::Bool => {
            if val != 0 {
                1
            } else {
                0
            }
        }
        CType::Char | CType::SChar => (val as i8) as i128,
        CType::UChar => (val as u8) as i128,
        CType::Short => (val as i16) as i128,
        CType::UShort => (val as u16) as i128,
        CType::Int => (val as i32) as i128,
        CType::UInt => (val as u32) as i128,
        CType::Long | CType::LongLong => {
            if target.pointer_width() == 4 {
                // ILP32: long = 32 bits
                match cty {
                    CType::Long => (val as i32) as i128,
                    _ => (val as i64) as i128,
                }
            } else {
                (val as i64) as i128
            }
        }
        CType::ULong | CType::ULongLong => {
            if target.pointer_width() == 4 {
                match cty {
                    CType::ULong => (val as u32) as i128,
                    _ => (val as u64) as i128,
                }
            } else {
                (val as u64) as i128
            }
        }
        CType::Pointer(..) => {
            if target.pointer_width() == 4 {
                (val as u32) as i128
            } else {
                (val as u64) as i128
            }
        }
        CType::Enum { .. } => (val as i32) as i128,
        // For types we can't determine a width for, pass through unchanged.
        _ => val,
    }
}

/// Serialize a `Constant` into a little-endian byte buffer of the given
/// size.  This handles scalars (`Integer`, `Float`, `LongDouble`), raw
/// byte strings (`String`), aggregate types (`Array`, `Struct`), and
/// zero-initialization, producing the exact memory representation that
/// the ELF `.data` section should contain.
fn constant_to_le_bytes(
    c: &Constant,
    expected_size: usize,
    ctype: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
) -> Vec<u8> {
    // Resolve the ctype through typedefs to check the underlying kind.
    let resolved = crate::common::types::resolve_typedef(ctype);
    match c {
        Constant::ZeroInit => vec![0u8; expected_size],
        Constant::Integer(v) => {
            // When the target type is a floating-point type but the constant
            // was evaluated as an integer (e.g. `double C = 2;`), convert
            // the integer value to the appropriate float representation so
            // the correct IEEE 754 bytes are emitted.
            match resolved {
                CType::Float => {
                    let fv = *v as f32;
                    let bits = fv.to_bits();
                    let mut buf = vec![0u8; expected_size];
                    for b in 0..expected_size.min(4) {
                        buf[b] = ((bits >> (b * 8)) & 0xFF) as u8;
                    }
                    return buf;
                }
                CType::Double => {
                    let fv = *v as f64;
                    let bits = fv.to_bits();
                    let mut buf = vec![0u8; expected_size];
                    for b in 0..expected_size.min(8) {
                        buf[b] = ((bits >> (b * 8)) & 0xFF) as u8;
                    }
                    return buf;
                }
                CType::LongDouble => {
                    // Store integer-initialized long double as f64
                    // representation so that the codegen's `movsd` (SSE
                    // double load) reads the correct value.  The codegen
                    // treats F80 internally at f64 precision.
                    let fv = *v as f64;
                    let f64_bits = fv.to_bits();
                    let mut buf = vec![0u8; expected_size];
                    for b in 0..expected_size.min(8) {
                        buf[b] = ((f64_bits >> (b * 8)) & 0xFF) as u8;
                    }
                    return buf;
                }
                _ => {}
            }
            let vu = *v as u128;
            let mut buf = vec![0u8; expected_size];
            for b in 0..expected_size.min(16) {
                buf[b] = ((vu >> (b * 8)) & 0xFF) as u8;
            }
            buf
        }
        Constant::Float(v) => {
            // When the declared C type is long double but the constant
            // was stored as Constant::Float (e.g. `long double G = 2.0;`
            // without the `L` suffix), store the f64 representation in
            // the first 8 bytes and zero-pad to expected_size.  The
            // codegen loads F80 globals with `movsd` (SSE double load),
            // so the f64 bytes must be in the low 8 bytes.
            let bits = v.to_bits();
            let mut buf = vec![0u8; expected_size];
            for b in 0..expected_size.min(8) {
                buf[b] = ((bits >> (b * 8)) & 0xFF) as u8;
            }
            buf
        }
        Constant::LongDouble(raw) => {
            // 80-bit extended precision.  Convert to f64 representation
            // for storage so that the codegen's `movsd` (SSE double load)
            // reads the correct value.  All long-double arithmetic in BCC
            // is performed at f64 precision internally.
            let raw_arr: [u8; 10] = {
                let mut arr = [0u8; 10];
                for (i, &b) in raw.iter().enumerate().take(10) {
                    arr[i] = b;
                }
                arr
            };
            let ld = crate::common::long_double::LongDouble::from_bytes(&raw_arr);
            let f64_val = ld.to_f64();
            let f64_bits = f64_val.to_bits();
            let mut buf = vec![0u8; expected_size];
            for b in 0..expected_size.min(8) {
                buf[b] = ((f64_bits >> (b * 8)) & 0xFF) as u8;
            }
            buf
        }
        Constant::String(data) => {
            let mut buf = vec![0u8; expected_size];
            for (i, &byte) in data.iter().enumerate() {
                if i < buf.len() {
                    buf[i] = byte;
                }
            }
            buf
        }
        Constant::Array(elems) => {
            let elem_ty = match resolved {
                CType::Array(et, _) => et.as_ref(),
                _ => &CType::Char,
            };
            let elem_size = type_builder.sizeof_type(elem_ty);
            let mut buf = vec![0u8; expected_size];
            for (i, elem) in elems.iter().enumerate() {
                let elem_bytes =
                    constant_to_le_bytes(elem, elem_size, elem_ty, target, type_builder);
                let off = i * elem_size;
                for (j, &b) in elem_bytes.iter().enumerate() {
                    if off + j < buf.len() {
                        buf[off + j] = b;
                    }
                }
            }
            buf
        }
        Constant::Struct(field_constants) => {
            let mut buf = vec![0u8; expected_size];
            let fields = match resolved {
                CType::Struct { fields, .. } => fields,
                CType::Union { fields, .. } => fields,
                _ => return buf,
            };
            let (is_packed, explicit_align) = match resolved {
                CType::Struct {
                    packed, aligned, ..
                } => (*packed, *aligned),
                _ => (false, None),
            };
            let layout =
                type_builder.compute_struct_layout_with_fields(fields, is_packed, explicit_align);
            for (fi, fc) in field_constants.iter().enumerate() {
                if let Some(fl) = layout.fields.get(fi) {
                    let ft = if fi < fields.len() {
                        &fields[fi].ty
                    } else {
                        &CType::Int
                    };
                    let fb = constant_to_le_bytes(fc, fl.size, ft, target, type_builder);
                    for (j, &b) in fb.iter().enumerate() {
                        if fl.offset + j < buf.len() {
                            buf[fl.offset + j] = b;
                        }
                    }
                }
            }
            buf
        }
        _ => vec![0u8; expected_size],
    }
}

/// Create an anonymous read-only global for a string literal used in
/// pointer-arithmetic within a global initializer (e.g. `"foo" + 1`).
/// Returns the generated symbol name.
fn create_anonymous_string_global(bytes: &[u8], target: &Target) -> String {
    let idx = super::COMPOUND_LITERAL_COUNTER.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    });
    let anon_name = format!("__anon_str.{}", idx);
    let ir_type = IrType::Array(Box::new(IrType::I8), bytes.len());
    let constant = Constant::String(bytes.to_vec());
    let cty = CType::Array(Box::new(CType::Char), Some(bytes.len()));
    super::COMPOUND_LITERAL_GLOBALS.with(|cl| {
        cl.borrow_mut()
            .push((anon_name.clone(), ir_type, constant, cty));
    });
    // Mark read-only so it goes to .rodata.
    let _ = target;
    anon_name
}

/// Extract an f64 from a Constant (Integer or Float).
fn const_as_f64(c: &Constant) -> f64 {
    match c {
        Constant::Float(f) => *f,
        Constant::Integer(i) => *i as f64,
        _ => 0.0,
    }
}

fn evaluate_const_binop(op: &ast::BinaryOp, lhs: &Constant, rhs: &Constant) -> Option<Constant> {
    match (lhs, rhs) {
        (Constant::Integer(a), Constant::Integer(b)) => {
            // Determine if both operands fit in 64-bit range.
            // C's widest standard integer type is `long long` (64-bit).
            // When both values fit in i64, perform arithmetic in i64
            // to match C overflow/truncation semantics.
            let fits_i64 = *a == (*a as i64 as i128) && *b == (*b as i64 as i128);
            let result = match op {
                ast::BinaryOp::Add => {
                    if fits_i64 {
                        (*a as i64).wrapping_add(*b as i64) as i128
                    } else {
                        a.wrapping_add(*b)
                    }
                }
                ast::BinaryOp::Sub => {
                    if fits_i64 {
                        (*a as i64).wrapping_sub(*b as i64) as i128
                    } else {
                        a.wrapping_sub(*b)
                    }
                }
                ast::BinaryOp::Mul => {
                    if fits_i64 {
                        (*a as i64).wrapping_mul(*b as i64) as i128
                    } else {
                        a.wrapping_mul(*b)
                    }
                }
                ast::BinaryOp::Div if *b != 0 => {
                    if fits_i64 {
                        (*a as i64).wrapping_div(*b as i64) as i128
                    } else {
                        a.wrapping_div(*b)
                    }
                }
                ast::BinaryOp::Mod if *b != 0 => {
                    if fits_i64 {
                        (*a as i64).wrapping_rem(*b as i64) as i128
                    } else {
                        a.wrapping_rem(*b)
                    }
                }
                ast::BinaryOp::BitwiseAnd => *a & *b,
                ast::BinaryOp::BitwiseOr => *a | *b,
                ast::BinaryOp::BitwiseXor => *a ^ *b,
                ast::BinaryOp::ShiftLeft => {
                    if fits_i64 {
                        (*a as i64).wrapping_shl(*b as u32) as i128
                    } else {
                        a.wrapping_shl(*b as u32)
                    }
                }
                ast::BinaryOp::ShiftRight => {
                    if fits_i64 {
                        (*a as i64).wrapping_shr(*b as u32) as i128
                    } else {
                        a.wrapping_shr(*b as u32)
                    }
                }
                ast::BinaryOp::Equal => {
                    if a == b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::NotEqual => {
                    if a != b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::Less => {
                    if a < b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::LessEqual => {
                    if a <= b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::Greater => {
                    if a > b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::GreaterEqual => {
                    if a >= b {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::LogicalAnd => {
                    if *a != 0 && *b != 0 {
                        1
                    } else {
                        0
                    }
                }
                ast::BinaryOp::LogicalOr => {
                    if *a != 0 || *b != 0 {
                        1
                    } else {
                        0
                    }
                }
                _ => return Some(Constant::Undefined),
            };
            Some(Constant::Integer(result))
        }
        (Constant::Float(a), Constant::Float(b)) => {
            let result = match op {
                ast::BinaryOp::Add => a + b,
                ast::BinaryOp::Sub => a - b,
                ast::BinaryOp::Mul => a * b,
                ast::BinaryOp::Div if *b != 0.0 => a / b,
                _ => return Some(Constant::Undefined),
            };
            Some(Constant::Float(result))
        }
        // _Complex constant arithmetic: Array([real, imag]) ± Array/Float
        (Constant::Array(a_parts), Constant::Array(b_parts))
            if a_parts.len() == 2 && b_parts.len() == 2 =>
        {
            let ar = const_as_f64(&a_parts[0]);
            let ai = const_as_f64(&a_parts[1]);
            let br = const_as_f64(&b_parts[0]);
            let bi = const_as_f64(&b_parts[1]);
            match op {
                ast::BinaryOp::Add => Some(Constant::Array(vec![
                    Constant::Float(ar + br),
                    Constant::Float(ai + bi),
                ])),
                ast::BinaryOp::Sub => Some(Constant::Array(vec![
                    Constant::Float(ar - br),
                    Constant::Float(ai - bi),
                ])),
                ast::BinaryOp::Mul => Some(Constant::Array(vec![
                    Constant::Float(ar * br - ai * bi),
                    Constant::Float(ar * bi + ai * br),
                ])),
                _ => Some(Constant::Undefined),
            }
        }
        // Float + Complex → promote scalar to Complex{scalar, 0.0}
        (Constant::Float(a), Constant::Array(b_parts)) if b_parts.len() == 2 => {
            let br = const_as_f64(&b_parts[0]);
            let bi = const_as_f64(&b_parts[1]);
            match op {
                ast::BinaryOp::Add => Some(Constant::Array(vec![
                    Constant::Float(a + br),
                    Constant::Float(bi),
                ])),
                ast::BinaryOp::Sub => Some(Constant::Array(vec![
                    Constant::Float(a - br),
                    Constant::Float(-bi),
                ])),
                ast::BinaryOp::Mul => {
                    // (a+0i)(br+bi*i) = (a*br) + (a*bi)i
                    Some(Constant::Array(vec![
                        Constant::Float(a * br),
                        Constant::Float(a * bi),
                    ]))
                }
                ast::BinaryOp::Div if br != 0.0 || bi != 0.0 => {
                    // (a+0i)/(br+bi*i)
                    let denom = br * br + bi * bi;
                    Some(Constant::Array(vec![
                        Constant::Float((a * br) / denom),
                        Constant::Float((-a * bi) / denom),
                    ]))
                }
                _ => Some(Constant::Undefined),
            }
        }
        // Complex + Float
        (Constant::Array(a_parts), Constant::Float(b)) if a_parts.len() == 2 => {
            let ar = const_as_f64(&a_parts[0]);
            let ai = const_as_f64(&a_parts[1]);
            match op {
                ast::BinaryOp::Add => Some(Constant::Array(vec![
                    Constant::Float(ar + b),
                    Constant::Float(ai),
                ])),
                ast::BinaryOp::Sub => Some(Constant::Array(vec![
                    Constant::Float(ar - b),
                    Constant::Float(ai),
                ])),
                ast::BinaryOp::Mul => {
                    // (ar+ai*i)(b+0i) = (ar*b) + (ai*b)i
                    Some(Constant::Array(vec![
                        Constant::Float(ar * b),
                        Constant::Float(ai * b),
                    ]))
                }
                ast::BinaryOp::Div if *b != 0.0 => Some(Constant::Array(vec![
                    Constant::Float(ar / b),
                    Constant::Float(ai / b),
                ])),
                _ => Some(Constant::Undefined),
            }
        }
        // Integer + Complex → promote integer to float, then complex add
        (Constant::Integer(a), Constant::Array(b_parts)) if b_parts.len() == 2 => {
            let af = *a as f64;
            let br = const_as_f64(&b_parts[0]);
            let bi = const_as_f64(&b_parts[1]);
            match op {
                ast::BinaryOp::Add => Some(Constant::Array(vec![
                    Constant::Float(af + br),
                    Constant::Float(bi),
                ])),
                ast::BinaryOp::Sub => Some(Constant::Array(vec![
                    Constant::Float(af - br),
                    Constant::Float(-bi),
                ])),
                ast::BinaryOp::Mul => {
                    // (af+0i)(br+bi*i) = (af*br) + (af*bi)i
                    Some(Constant::Array(vec![
                        Constant::Float(af * br),
                        Constant::Float(af * bi),
                    ]))
                }
                ast::BinaryOp::Div if br != 0.0 || bi != 0.0 => {
                    let denom = br * br + bi * bi;
                    Some(Constant::Array(vec![
                        Constant::Float((af * br) / denom),
                        Constant::Float((-af * bi) / denom),
                    ]))
                }
                _ => Some(Constant::Undefined),
            }
        }
        // Complex + Integer → promote integer to float, then complex add
        (Constant::Array(a_parts), Constant::Integer(b)) if a_parts.len() == 2 => {
            let ar = const_as_f64(&a_parts[0]);
            let ai = const_as_f64(&a_parts[1]);
            let bf = *b as f64;
            match op {
                ast::BinaryOp::Add => Some(Constant::Array(vec![
                    Constant::Float(ar + bf),
                    Constant::Float(ai),
                ])),
                ast::BinaryOp::Sub => Some(Constant::Array(vec![
                    Constant::Float(ar - bf),
                    Constant::Float(ai),
                ])),
                ast::BinaryOp::Mul => {
                    // (ar+ai*i)(bf+0i) = (ar*bf) + (ai*bf)i
                    Some(Constant::Array(vec![
                        Constant::Float(ar * bf),
                        Constant::Float(ai * bf),
                    ]))
                }
                ast::BinaryOp::Div if *b != 0 => Some(Constant::Array(vec![
                    Constant::Float(ar / bf),
                    Constant::Float(ai / bf),
                ])),
                _ => Some(Constant::Undefined),
            }
        }
        // Integer + Float → promote integer, compute float result
        (Constant::Integer(a), Constant::Float(b)) => {
            let af = *a as f64;
            let result = match op {
                ast::BinaryOp::Add => af + b,
                ast::BinaryOp::Sub => af - b,
                ast::BinaryOp::Mul => af * b,
                ast::BinaryOp::Div if *b != 0.0 => af / b,
                _ => return Some(Constant::Undefined),
            };
            Some(Constant::Float(result))
        }
        // Float + Integer → promote integer, compute float result
        (Constant::Float(a), Constant::Integer(b)) => {
            let bf = *b as f64;
            let result = match op {
                ast::BinaryOp::Add => a + bf,
                ast::BinaryOp::Sub => a - bf,
                ast::BinaryOp::Mul => a * bf,
                ast::BinaryOp::Div if *b != 0 => a / bf,
                _ => return Some(Constant::Undefined),
            };
            Some(Constant::Float(result))
        }
        _ => Some(Constant::Undefined),
    }
}

// ===========================================================================
// Designated initializer lowering (constant, for globals)
// ===========================================================================

/// Lower a designated initializer list for a struct or array global.
fn lower_designated_initializer(
    init_list: &[ast::DesignatedInitializer],
    target_type: &CType,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    name_table: &[String],
    enum_constants: &FxHashMap<String, i128>,
) -> Option<Constant> {
    let resolved_ref = crate::common::types::resolve_typedef(target_type);

    // Resolve forward-referenced (empty-field) named structs/unions using the
    // thread-local struct definitions registry populated during Pass 0.
    // When a variable is declared as `struct Foo x = {...};`, the AST stores
    // the struct type with the tag name but without field information (the
    // definition body is in a separate AST node).  We look up the full
    // definition so the initializer can be correctly lowered.
    let resolved_owned: Option<CType> = match resolved_ref {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => super::SIZEOF_STRUCT_DEFS.with(|defs| {
            let borrow = defs.borrow();
            borrow.as_ref().and_then(|m| m.get(tag.as_str()).cloned())
        }),
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => super::SIZEOF_STRUCT_DEFS.with(|defs| {
            let borrow = defs.borrow();
            borrow.as_ref().and_then(|m| m.get(tag.as_str()).cloned())
        }),
        _ => None,
    };
    let resolved = resolved_owned.as_ref().unwrap_or(resolved_ref);

    match resolved {
        CType::Array(element_type, size_opt) => {
            let array_len = size_opt.unwrap_or(init_list.len());
            let mut elements = vec![Constant::ZeroInit; array_len];
            let mut current_idx = 0usize;
            for desig_init in init_list {
                // Handle GCC range designator: [low ... high] = value.
                if let Some(ast::Designator::IndexRange(ref low_expr, ref high_expr, _)) =
                    desig_init.designators.first()
                {
                    let low = evaluate_const_int_expr(low_expr).unwrap_or(current_idx);
                    let high = evaluate_const_int_expr(high_expr).unwrap_or(low);
                    if let Some(c) = evaluate_initializer_constant(
                        &desig_init.initializer,
                        element_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        enum_constants,
                    ) {
                        for arr_idx in low..=high {
                            if arr_idx < array_len {
                                elements[arr_idx] = c.clone();
                            }
                        }
                    }
                    current_idx = high + 1;
                    continue;
                }
                current_idx = resolve_designator_index(&desig_init.designators, current_idx);
                if current_idx < array_len {
                    if let Some(c) = evaluate_initializer_constant(
                        &desig_init.initializer,
                        element_type,
                        target,
                        type_builder,
                        diagnostics,
                        name_table,
                        enum_constants,
                    ) {
                        elements[current_idx] = c;
                    }
                }
                current_idx += 1;
            }
            Some(Constant::Array(elements))
        }
        CType::Struct {
            ref fields,
            packed: is_packed,
            aligned: explicit_align,
            ..
        } => {
            // Check if the struct has any bitfield members.
            let has_bitfields = fields.iter().any(|f| f.bit_width.is_some());

            if has_bitfields {
                // For bitfield structs, build a packed byte buffer directly
                // because Constant::Struct(per-field) doesn't handle the
                // bit-level packing required by bitfield storage units.
                let layout = type_builder.compute_struct_layout_with_fields(
                    fields,
                    *is_packed,
                    *explicit_align,
                );
                let struct_size = type_builder.sizeof_type(resolved);
                let mut bytes = vec![0u8; struct_size];

                // Evaluate each field value.
                let field_count = fields.len();
                let mut field_constants: Vec<Option<Constant>> = vec![None; field_count];
                let mut current_idx = 0usize;
                for desig_init in init_list {
                    current_idx = resolve_field_designator(
                        &desig_init.designators,
                        fields,
                        current_idx,
                        name_table,
                    );
                    // Skip anonymous bitfields (unnamed padding fields).
                    while current_idx < field_count && fields[current_idx].name.is_none() {
                        current_idx += 1;
                    }
                    if current_idx < field_count {
                        let field_type = &fields[current_idx].ty;
                        if let Some(c) = evaluate_initializer_constant(
                            &desig_init.initializer,
                            field_type,
                            target,
                            type_builder,
                            diagnostics,
                            name_table,
                            enum_constants,
                        ) {
                            field_constants[current_idx] = Some(c);
                        }
                    }
                    current_idx += 1;
                }

                // Pack each field into the byte buffer.
                for (fi, _field) in fields.iter().enumerate() {
                    let c = match &field_constants[fi] {
                        Some(c) => c.clone(),
                        None => Constant::ZeroInit,
                    };
                    if let Some(fl) = layout.fields.get(fi) {
                        if let Some((bit_offset_in_unit, bit_width)) = fl.bitfield_info {
                            if bit_width == 0 {
                                continue; // zero-width bitfield
                            }
                            let val: i128 = match &c {
                                Constant::Integer(v) => *v,
                                Constant::Float(v) => v.to_bits() as i128,
                                Constant::ZeroInit => 0,
                                _ => 0,
                            };
                            // Pack into the storage unit at fl.offset
                            let mask = if bit_width >= 128 {
                                !0u128
                            } else {
                                (1u128 << bit_width) - 1
                            };
                            let masked_val = (val as u128) & mask;
                            let shifted = masked_val << bit_offset_in_unit;

                            // OR into the byte buffer at the storage unit's
                            // byte offset (little-endian).
                            let unit_bytes = fl.size.max(1);
                            for b in 0..unit_bytes {
                                let byte_idx = fl.offset + b;
                                if byte_idx < bytes.len() {
                                    bytes[byte_idx] |= ((shifted >> (b * 8)) & 0xFF) as u8;
                                }
                            }
                        } else {
                            // Regular (non-bitfield) field — write at byte
                            // offset.  Serialize the constant to bytes
                            // and copy into the buffer.
                            let field_bytes = constant_to_le_bytes(
                                &c,
                                fl.size,
                                &fields[fi].ty,
                                target,
                                type_builder,
                            );
                            for (b, &byte) in field_bytes.iter().enumerate() {
                                let byte_idx = fl.offset + b;
                                if byte_idx < bytes.len() {
                                    bytes[byte_idx] = byte;
                                }
                            }
                        }
                    }
                }

                // -----------------------------------------------------------
                // Post-processing: when any non-bitfield field holds a
                // GlobalRef / GlobalRefOffset (function-pointer initializer),
                // the flat byte buffer loses the relocation info (because
                // constant_to_le_bytes writes zeros for GlobalRef).
                //
                // Detect this and produce a Constant::Struct matching the
                // IR type's allocation-unit layout so that
                // collect_constant_relocs in generation.rs can find the
                // references and emit proper R_*_64 / R_*_32 relocations.
                // -----------------------------------------------------------
                let has_relocatable_fields = field_constants.iter().enumerate().any(|(fi, fc)| {
                    if let Some(ref c) = fc {
                        // Only care about non-bitfield fields
                        if let Some(fl) = layout.fields.get(fi) {
                            if fl.bitfield_info.is_none() {
                                return matches!(
                                    c,
                                    Constant::GlobalRef(_) | Constant::GlobalRefOffset(_, _)
                                ) || constant_tree_has_global_ref(c);
                            }
                        }
                    }
                    false
                });

                if has_relocatable_fields {
                    // Build a Constant::Struct whose elements correspond to
                    // the IR struct's allocation units (I64, I32, etc.).
                    // The IR type for a bitfield struct is
                    //   Struct([unit_ty; unit_count])
                    // where unit_ty/unit_size are chosen to match the struct's
                    // alignment.
                    let struct_align = type_builder.alignof_type(resolved);
                    let (unit_size, _unit_ty_unused) = if struct_align >= 8 && struct_size % 8 == 0
                    {
                        (8usize, ())
                    } else if struct_align >= 4 && struct_size % 4 == 0 {
                        (4usize, ())
                    } else if struct_align >= 2 && struct_size % 2 == 0 {
                        (2usize, ())
                    } else {
                        (1usize, ())
                    };
                    let unit_count = if unit_size > 0 && struct_size > 0 {
                        struct_size / unit_size
                    } else {
                        0
                    };

                    // Map: byte_offset → (GlobalRef constant) for non-bitfield
                    // relocatable fields.
                    let mut reloc_map: Vec<(usize, Constant)> = Vec::new();
                    for (fi, fc) in field_constants.iter().enumerate() {
                        if let Some(ref c) = fc {
                            if let Some(fl) = layout.fields.get(fi) {
                                if fl.bitfield_info.is_none()
                                    && (matches!(
                                        c,
                                        Constant::GlobalRef(_) | Constant::GlobalRefOffset(_, _)
                                    ) || constant_tree_has_global_ref(c))
                                {
                                    reloc_map.push((fl.offset, c.clone()));
                                }
                            }
                        }
                    }

                    let mut struct_elems: Vec<Constant> = Vec::with_capacity(unit_count);
                    for ui in 0..unit_count {
                        let unit_start = ui * unit_size;
                        // Check if a relocatable field starts at this unit.
                        if let Some((_off, ref reloc_const)) =
                            reloc_map.iter().find(|(off, _)| *off == unit_start)
                        {
                            struct_elems.push(reloc_const.clone());
                        } else {
                            // Pack the bytes into an integer constant.
                            let mut val: u64 = 0;
                            for b in 0..unit_size {
                                let byte_idx = unit_start + b;
                                if byte_idx < bytes.len() {
                                    val |= (bytes[byte_idx] as u64) << (b * 8);
                                }
                            }
                            struct_elems.push(Constant::Integer(val as i128));
                        }
                    }
                    Some(Constant::Struct(struct_elems))
                } else {
                    Some(Constant::String(bytes))
                }
            } else {
                // Non-bitfield struct — use per-field constants, with brace
                // elision support for aggregate (array/struct) sub-fields.
                let field_count = fields.len();
                let mut field_values = vec![Constant::ZeroInit; field_count];
                let mut current_idx = 0usize;
                let mut init_iter = init_list.iter().peekable();
                while let Some(desig_init) = init_iter.next() {
                    current_idx = resolve_field_designator(
                        &desig_init.designators,
                        fields,
                        current_idx,
                        name_table,
                    );
                    // Skip anonymous fields (unnamed bitfields, anonymous
                    // struct/union padding fields).
                    while current_idx < field_count && fields[current_idx].name.is_none() {
                        current_idx += 1;
                    }
                    if current_idx < field_count {
                        let field_type = &fields[current_idx].ty;
                        let resolved_ft = crate::common::types::resolve_typedef(field_type);

                        // Detect brace elision: if the current initializer is
                        // a scalar Expression destined for an aggregate
                        // (array/struct) field and there are enough remaining
                        // scalar inits, collect them into a sub-list and
                        // recurse.
                        //
                        // Exception: a string literal directly initializing a
                        // char array field (e.g. `char a[32] = "abc"`) is NOT
                        // brace elision — the string literal is a complete
                        // initializer for the array.
                        let is_string_for_char_array = matches!(
                            &desig_init.initializer,
                            ast::Initializer::Expression(expr) if matches!(
                                expr.as_ref(),
                                ast::Expression::StringLiteral { .. }
                            )
                        ) && matches!(
                            resolved_ft,
                            CType::Array(ref elem, _)
                            if matches!(
                                elem.as_ref(),
                                CType::Char | CType::SChar | CType::UChar
                                    | CType::Short | CType::UShort
                                    | CType::Int | CType::UInt
                            )
                        );
                        let needs_brace_elision = desig_init.designators.is_empty()
                            && matches!(resolved_ft, CType::Array(..) | CType::Struct { .. })
                            && matches!(&desig_init.initializer, ast::Initializer::Expression(_))
                            && !is_string_for_char_array;

                        if needs_brace_elision {
                            // Count how many scalar elements the aggregate
                            // needs in a flat initializer.
                            let sub_count = flat_scalar_count(resolved_ft, type_builder);
                            let mut sub_inits: Vec<ast::DesignatedInitializer> = Vec::new();
                            // First item is the current desig_init
                            sub_inits.push(ast::DesignatedInitializer {
                                designators: Vec::new(),
                                initializer: desig_init.initializer.clone(),
                                span: desig_init.span,
                            });
                            // Consume additional items from the iterator
                            for _ in 1..sub_count {
                                if let Some(next) = init_iter.next() {
                                    sub_inits.push(ast::DesignatedInitializer {
                                        designators: Vec::new(),
                                        initializer: next.initializer.clone(),
                                        span: next.span,
                                    });
                                } else {
                                    break;
                                }
                            }
                            if let Some(c) = lower_designated_initializer(
                                &sub_inits,
                                resolved_ft,
                                target,
                                type_builder,
                                diagnostics,
                                name_table,
                                enum_constants,
                            ) {
                                field_values[current_idx] = c;
                            }
                        } else if desig_init.designators.len() > 1 {
                            // Chained designator such as `.data.string = { ... }`.
                            // `resolve_field_designator` only consumed the first
                            // designator (`.data`).  The remaining designators
                            // (`.string`) must be forwarded to the recursive call
                            // so the union/struct member selection works correctly.
                            let remaining_desigs = desig_init.designators[1..].to_vec();
                            let synthetic_item = ast::DesignatedInitializer {
                                designators: remaining_desigs,
                                initializer: desig_init.initializer.clone(),
                                span: desig_init.span,
                            };
                            let synthetic_init = ast::Initializer::List {
                                designators_and_initializers: vec![synthetic_item],
                                trailing_comma: false,
                                span: desig_init.span,
                            };
                            if let Some(c) = evaluate_initializer_constant(
                                &synthetic_init,
                                field_type,
                                target,
                                type_builder,
                                diagnostics,
                                name_table,
                                enum_constants,
                            ) {
                                field_values[current_idx] = c;
                            }
                        } else if let Some(c) = evaluate_initializer_constant(
                            &desig_init.initializer,
                            field_type,
                            target,
                            type_builder,
                            diagnostics,
                            name_table,
                            enum_constants,
                        ) {
                            field_values[current_idx] = c;
                        }
                    }
                    current_idx += 1;
                }
                Some(Constant::Struct(field_values))
            }
        }
        CType::Union { ref fields, .. } => {
            if let Some(first) = init_list.first() {
                let field_idx = resolve_field_designator(&first.designators, fields, 0, name_table);
                if field_idx < fields.len() {
                    let member_ty = &fields[field_idx].ty;

                    // Handle nested designator chains like `.f.d = value`.
                    // resolve_field_designator only consumes the first
                    // designator.  If there are remaining designators,
                    // wrap them into a synthetic InitList so the recursive
                    // call processes them against the member type.
                    let member_const = if first.designators.len() > 1 {
                        let remaining_desig = first.designators[1..].to_vec();
                        let synthetic_item = ast::DesignatedInitializer {
                            designators: remaining_desig,
                            initializer: first.initializer.clone(),
                            span: first.span,
                        };
                        let synthetic_init = ast::Initializer::List {
                            designators_and_initializers: vec![synthetic_item],
                            trailing_comma: false,
                            span: first.span,
                        };
                        evaluate_initializer_constant(
                            &synthetic_init,
                            member_ty,
                            target,
                            type_builder,
                            diagnostics,
                            name_table,
                            enum_constants,
                        )
                    } else {
                        evaluate_initializer_constant(
                            &first.initializer,
                            member_ty,
                            target,
                            type_builder,
                            diagnostics,
                            name_table,
                            enum_constants,
                        )
                    };

                    // A union's IR type is Array(I32/I64, N), NOT Struct.
                    // If the member constant is a Struct/Array, we must
                    // serialize it to a byte buffer so that
                    // constant_to_bytes_typed can produce the correct
                    // output against the union's Array IR type.
                    if let Some(mc) = member_const {
                        // When the union member is a bitfield, mask the
                        // constant value to the bitfield width and produce
                        // a byte buffer.  Without this, the full integer
                        // value leaks into the union storage — violating
                        // C11 §6.7.2.1p10 (bitfield width limits).
                        let bit_width_opt = fields[field_idx].bit_width;
                        if let Some(bw) = bit_width_opt {
                            let union_size =
                                crate::common::types::sizeof_ctype(target_type, target);
                            let mut bytes = vec![0u8; union_size];
                            let val: u128 = match &mc {
                                Constant::Integer(v) => *v as u128,
                                Constant::Float(v) => v.to_bits() as u128,
                                Constant::ZeroInit => 0,
                                _ => 0,
                            };
                            let mask = if bw >= 128 { !0u128 } else { (1u128 << bw) - 1 };
                            let masked = val & mask;
                            // Store at bit offset 0 within the first
                            // storage unit (union always starts at 0).
                            let unit_bytes = ((bw as usize) + 7) / 8;
                            let unit_bytes = unit_bytes.max(1).min(union_size);
                            for b in 0..unit_bytes {
                                if b < bytes.len() {
                                    bytes[b] = ((masked >> (b * 8)) & 0xFF) as u8;
                                }
                            }
                            return Some(Constant::String(bytes));
                        }

                        // Check if the member constant carries relocation
                        // information (address of a global, string literal
                        // used as a pointer, etc.).  Serializing such
                        // constants to raw bytes with `constant_to_le_bytes`
                        // loses the relocation, causing the linker to emit
                        // zero bytes instead of patching the address.
                        //
                        // The check is recursive: it detects not only bare
                        // pointers (`const char *`) but also struct members
                        // containing pointers (e.g., `struct { uint8_t a;
                        // uint8_t b; func_ptr_union cfunc; }`) and nested
                        // unions (e.g., QuickJS `JSCFunctionType`).
                        //
                        // For relocation-bearing constants, we produce an
                        // `Array(I64, N)` constant that distributes the
                        // member data across elements based on byte offset:
                        //  - Relocatable entries (GlobalRef, String-at-ptr)
                        //    are placed at the element index corresponding
                        //    to their byte offset / element_size.
                        //  - Non-relocatable data is serialized to bytes
                        //    and packed into Integer constants.
                        //
                        // This correctly handles multi-field struct members
                        // where the pointer is NOT at byte offset 0 (e.g.,
                        // `struct { uint8_t, uint8_t, func_ptr }` has the
                        // pointer at offset 8, landing in element 1).
                        let needs_reloc = constant_contains_relocatable(&mc, member_ty);
                        if needs_reloc {
                            let union_ir_ty = IrType::from_ctype(target_type, target);
                            if let IrType::Array(ref elem_ir_ty, count) = union_ir_ty {
                                let count = count.max(1);
                                let elem_size = match elem_ir_ty.as_ref() {
                                    IrType::I64 => 8usize,
                                    IrType::I32 => 4usize,
                                    IrType::I16 => 2usize,
                                    _ => 1usize,
                                };
                                let union_size = count * elem_size;

                                // 1) Serialize the member to a byte
                                //    buffer.  Relocatable entries produce
                                //    zeros in the buffer (placeholder).
                                let member_size =
                                    crate::common::types::sizeof_ctype(member_ty, target);
                                let mut bytes = constant_to_le_bytes(
                                    &mc,
                                    member_size,
                                    member_ty,
                                    target,
                                    type_builder,
                                );
                                bytes.resize(union_size, 0);

                                // 2) Collect byte offsets of all
                                //    relocatable entries.
                                let mut relocs = Vec::new();
                                collect_reloc_positions_in_constant(
                                    &mc,
                                    member_ty,
                                    0,
                                    type_builder,
                                    target,
                                    &mut relocs,
                                );

                                // 3) Build Array elements: relocatable
                                //    entries at their correct positions,
                                //    byte-packed Integers elsewhere.
                                let mut elems = Vec::with_capacity(count);
                                for i in 0..count {
                                    let offset = i * elem_size;
                                    if let Some((_, ref rc)) =
                                        relocs.iter().find(|(o, _)| *o == offset)
                                    {
                                        elems.push(rc.clone());
                                    } else {
                                        // Pack serialized bytes into an
                                        // Integer constant.
                                        let end = (offset + elem_size).min(bytes.len());
                                        let start = offset.min(bytes.len());
                                        let avail = end - start;
                                        if elem_size <= 4 {
                                            let mut buf = [0u8; 4];
                                            buf[..avail].copy_from_slice(&bytes[start..end]);
                                            elems.push(Constant::Integer(i128::from(
                                                u32::from_le_bytes(buf),
                                            )));
                                        } else {
                                            let mut buf = [0u8; 8];
                                            buf[..avail].copy_from_slice(&bytes[start..end]);
                                            elems.push(Constant::Integer(i128::from(
                                                u64::from_le_bytes(buf),
                                            )));
                                        }
                                    }
                                }
                                return Some(Constant::Array(elems));
                            }
                        }

                        // For non-relocation constants, serialize the member
                        // value to bytes.  This ensures that:
                        //  1. The value is truncated to the member's byte
                        //     width (e.g. -1L stored into int32_t becomes
                        //     4 bytes, not 8).
                        //  2. The remaining union bytes are zero-padded.
                        // Without this, a Constant::Integer(-1) (8 bytes)
                        // stored into a union { int32_t; int64_t; } would
                        // write 8 bytes of 0xFF instead of 4 + 4 zeros.
                        let member_size = crate::common::types::sizeof_ctype(member_ty, target);
                        let union_size = crate::common::types::sizeof_ctype(target_type, target);
                        let mut bytes =
                            constant_to_le_bytes(&mc, member_size, member_ty, target, type_builder);
                        bytes.resize(union_size, 0);
                        return Some(Constant::String(bytes));
                    }
                }
            }
            Some(Constant::ZeroInit)
        }
        _ => {
            if let Some(first) = init_list.first() {
                evaluate_initializer_constant(
                    &first.initializer,
                    target_type,
                    target,
                    type_builder,
                    diagnostics,
                    name_table,
                    enum_constants,
                )
            } else {
                Some(Constant::ZeroInit)
            }
        }
    }
}

// ===========================================================================
// Parameter allocation
// ===========================================================================

/// Create allocas for all function parameters in the entry block.
/// Evaluate VLA parameter dimension expressions for their side effects.
///
/// C11 §6.7.6.2p5: If a function parameter is declared as a VLA
/// (e.g., `int arr[n++]`), the dimension expression is evaluated at
/// function entry even though the parameter type decays to a pointer.
/// The result of the evaluation is discarded — only the side effects
/// (like `n++`) matter.
#[allow(clippy::too_many_arguments)]
fn evaluate_vla_param_side_effects(
    params: &[ast::ParameterDeclaration],
    builder: &mut IrBuilder,
    function: &mut IrFunction,
    module: &mut IrModule,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    local_vars: &FxHashMap<String, Value>,
    param_values: &FxHashMap<String, Value>,
    name_table: &[String],
    local_types: &FxHashMap<String, CType>,
    enum_constants: &FxHashMap<String, i128>,
    static_locals: &mut FxHashMap<String, String>,
    struct_defs: &FxHashMap<String, CType>,
    func_name: &str,
) {
    for param_decl in params {
        // Extract VLA size expressions from the parameter's declarator.
        // For `int arr[i++]`, this returns the expression `i++`.
        let vla_expr = param_decl
            .declarator
            .as_ref()
            .and_then(extract_vla_size_expr);
        if let Some(vla_expr) = vla_expr {
            // Evaluate the expression (for side effects only).
            let empty_labels = FxHashMap::default();
            let empty_vla = FxHashMap::default();
            let empty_scope_overrides = FxHashMap::default();
            let mut layout_cache = FxHashMap::default();
            let mut expr_ctx = expr_lowering::ExprLoweringContext {
                builder,
                function,
                module,
                target,
                type_builder,
                diagnostics,
                local_vars,
                param_values,
                name_table,
                local_types,
                enum_constants,
                static_locals,
                struct_defs,
                label_blocks: &empty_labels,
                current_function_name: Some(func_name),
                enclosing_loop_stack: Vec::new(),
                scope_type_overrides: &empty_scope_overrides,
                last_bitfield_info: None,
                layout_cache: &mut layout_cache,
                vla_sizes: &empty_vla,
            };
            // Lower the expression — the result value is discarded,
            // but any side effects (increment, decrement, function
            // calls in the dimension expression) will be emitted.
            let _ = expr_lowering::lower_expression(&mut expr_ctx, &vla_expr);
        }
    }
}

fn allocate_parameters(
    builder: &mut IrBuilder,
    function: &mut IrFunction,
    params: &[ast::ParameterDeclaration],
    target: &Target,
    local_vars: &mut FxHashMap<String, Value>,
    name_table: &[String],
    knr_param_types: &FxHashMap<String, CType>,
) {
    for (idx, param_decl) in params.iter().enumerate() {
        let param_name = extract_param_name(param_decl, name_table);
        if param_name.is_empty() {
            continue;
        }

        if std::env::var("BCC_DEBUG_KNR").is_ok() {
            eprintln!(
                "[KNR-DEBUG] allocate_parameters: idx={} name='{}' knr_types={:?}",
                idx,
                param_name,
                knr_param_types.keys().collect::<Vec<_>>()
            );
        }

        // For K&R-style functions, use the K&R type map which has the real
        // parameter types.  Without this, `void f(p) Point p;` where
        // `Point = {long, long}` would use resolve_param_type which returns
        // Int (4 bytes) instead of the full struct (16 bytes), causing
        // stack corruption from overlapping allocas.
        let param_c_type = if let Some(knr_ty) = knr_param_types.get(&param_name) {
            knr_ty.clone()
        } else {
            resolve_param_type(param_decl, target, name_table)
        };
        // C11 §6.7.6.3p7: Function parameter type adjustments —
        // "array of T" → "pointer to T", "function returning T" → "pointer to function returning T".
        // Without this, `char bar[]` would produce a 0/1-byte alloca instead
        // of an 8-byte pointer alloca, causing stack corruption.
        let param_c_type = match param_c_type {
            CType::Array(elem, _) => {
                CType::Pointer(elem, crate::common::types::TypeQualifiers::default())
            }
            CType::Function { .. } => CType::Pointer(
                Box::new(param_c_type),
                crate::common::types::TypeQualifiers::default(),
            ),
            other => other,
        };
        // Resolve forward-referenced struct/union tag references to their
        // full definitions so that the alloca gets the correct size.
        // Without this, `union YYMINORTYPE minor` (tag reference without
        // body) would produce a 0-byte alloca instead of the full union size.
        let param_c_type = resolve_sizeof_struct_ref(param_c_type, target);
        let param_ir_type = IrType::from_ctype(&param_c_type, target);

        if std::env::var("BCC_DEBUG_KNR").is_ok() {
            eprintln!(
                "[KNR-DEBUG]   param '{}': c_type={:?} ir_type={:?} size={}",
                param_name,
                param_c_type,
                param_ir_type,
                param_ir_type.size_bytes(target)
            );
        }

        // Round up struct alloca sizes to the next power-of-2 when the
        // natural struct size is not a power-of-2 (e.g. 3, 5, 6, 7 bytes).
        // The ABI passes small structs in registers sized to the next
        // power-of-2 (3 bytes → I32 register), so a 4-byte register store
        // into a 3-byte alloca overflows by 1 byte, corrupting the saved
        // frame pointer.  Padding the alloca to the register width prevents
        // this while keeping member-access offsets unchanged (GEP uses byte
        // offsets, not the alloca's IR type).
        let alloca_ir_type = match &param_ir_type {
            IrType::Struct(_) | IrType::Array(_, _) => {
                let sz = param_ir_type.size_bytes(target);
                if sz > 0 && sz <= 8 && !sz.is_power_of_two() {
                    let rounded = sz.next_power_of_two();
                    IrType::Array(Box::new(IrType::I8), rounded)
                } else {
                    param_ir_type.clone()
                }
            }
            _ => param_ir_type.clone(),
        };

        let (alloca_val, alloca_inst) = builder.build_alloca(alloca_ir_type, Span::dummy());
        push_inst_to_entry(function, alloca_inst);

        let param_value = if idx < function.params.len() {
            function.params[idx].value
        } else {
            Value::UNDEF
        };

        let store_inst = builder.build_store(param_value, alloca_val, Span::dummy());
        push_inst_to_entry(function, store_inst);

        local_vars.insert(param_name, alloca_val);
    }
}

// ===========================================================================
// Local variable scanning
// ===========================================================================

/// Recursively scan the function body for ALL local variable declarations.
fn collect_local_variables(body: &ast::Statement, name_table: &[String]) -> Vec<LocalVarInfo> {
    let mut locals = Vec::new();
    collect_locals_recursive(body, &mut locals, name_table);
    locals
}

/// Recursively collect all C label names from a statement tree.
///
/// Pre-scanning for labels enables forward references in `asm goto`
/// statements — the label block is created before the function body
/// is lowered, so `wire_asm_goto_targets` can always find the block.
/// Checks whether an initializer (recursively) contains any `AddressOfLabel`
/// expressions (GCC `&&label`). Used to downgrade static local variables to
/// stack-allocated when they contain label address initializers, since label
/// addresses are function-local code pointers that cannot be resolved as
/// link-time constants.
/// Public accessor for `initializer_contains_label_address`.
pub fn initializer_contains_label_address_pub(init: &ast::Initializer) -> bool {
    initializer_contains_label_address(init)
}

fn initializer_contains_label_address(init: &ast::Initializer) -> bool {
    match init {
        ast::Initializer::Expression(expr) => expr_contains_label_address(expr),
        ast::Initializer::List {
            designators_and_initializers,
            ..
        } => designators_and_initializers
            .iter()
            .any(|item| initializer_contains_label_address(&item.initializer)),
    }
}

/// Recursively checks if an expression contains `AddressOfLabel`.
fn expr_contains_label_address(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::AddressOfLabel { .. } => true,
        ast::Expression::Parenthesized { inner, .. } => expr_contains_label_address(inner),
        ast::Expression::Cast { operand, .. } => expr_contains_label_address(operand),
        ast::Expression::UnaryOp { operand, .. } => expr_contains_label_address(operand),
        ast::Expression::Binary { left, right, .. } => {
            expr_contains_label_address(left) || expr_contains_label_address(right)
        }
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            expr_contains_label_address(condition)
                || then_expr
                    .as_ref()
                    .map_or(false, |e| expr_contains_label_address(e))
                || expr_contains_label_address(else_expr)
        }
        _ => false,
    }
}

fn collect_label_names(stmt: &ast::Statement, labels: &mut Vec<String>, name_table: &[String]) {
    match stmt {
        ast::Statement::Labeled {
            label, statement, ..
        } => {
            // Use the same naming convention as lower_label in stmt_lowering.rs:
            // format!("label_{}", label.as_u32())
            labels.push(format!("label_{}", label.as_u32()));
            let _ = name_table; // suppress unused
            collect_label_names(statement, labels, name_table);
        }
        ast::Statement::Compound(compound) => {
            for item in &compound.items {
                match item {
                    ast::BlockItem::Statement(s) => collect_label_names(s, labels, name_table),
                    ast::BlockItem::Declaration(_) => {}
                }
            }
        }
        ast::Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_label_names(then_branch, labels, name_table);
            if let Some(ref e) = else_branch {
                collect_label_names(e, labels, name_table);
            }
        }
        ast::Statement::While { body, .. }
        | ast::Statement::DoWhile { body, .. }
        | ast::Statement::Switch { body, .. } => {
            collect_label_names(body, labels, name_table);
        }
        ast::Statement::For { body, .. } => {
            collect_label_names(body, labels, name_table);
        }
        ast::Statement::Case { statement, .. }
        | ast::Statement::Default { statement, .. }
        | ast::Statement::CaseRange { statement, .. } => {
            collect_label_names(statement, labels, name_table);
        }
        _ => {}
    }
}

fn collect_locals_recursive(
    stmt: &ast::Statement,
    locals: &mut Vec<LocalVarInfo>,
    name_table: &[String],
) {
    match stmt {
        ast::Statement::Compound(compound) => {
            for item in &compound.items {
                match item {
                    ast::BlockItem::Declaration(decl) => {
                        collect_locals_from_declaration(decl, locals, name_table);
                    }
                    ast::BlockItem::Statement(inner_stmt) => {
                        collect_locals_recursive(inner_stmt, locals, name_table);
                    }
                }
            }
        }
        ast::Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_locals_recursive(then_branch, locals, name_table);
            if let Some(ref else_stmt) = else_branch {
                collect_locals_recursive(else_stmt, locals, name_table);
            }
        }
        ast::Statement::While { body, .. } | ast::Statement::DoWhile { body, .. } => {
            collect_locals_recursive(body, locals, name_table);
        }
        ast::Statement::For { init, body, .. } => {
            if let Some(ref for_init) = init {
                match for_init {
                    ast::ForInit::Declaration(decl) => {
                        collect_locals_from_declaration(decl, locals, name_table);
                    }
                    ast::ForInit::Expression(_) => {}
                }
            }
            collect_locals_recursive(body, locals, name_table);
        }
        ast::Statement::Switch { body, .. } => {
            collect_locals_recursive(body, locals, name_table);
        }
        ast::Statement::Labeled { statement, .. }
        | ast::Statement::Case { statement, .. }
        | ast::Statement::Default { statement, .. } => {
            collect_locals_recursive(statement, locals, name_table);
        }
        ast::Statement::CaseRange { statement, .. } => {
            collect_locals_recursive(statement, locals, name_table);
        }
        _ => {}
    }
}

/// Infer array size from an initializer for incomplete array declarations (`T arr[] = { ... }`).
///
/// Returns the number of elements in the initializer list, or the string
/// length (including null terminator) for string initializers.
fn infer_array_size_from_initializer(init: &ast::Initializer, elem_type: &CType) -> usize {
    match init {
        ast::Initializer::List {
            designators_and_initializers,
            ..
        } => {
            // Count the number of top-level initializer elements.
            // For designated initializers, we use the max index + 1.
            let mut max_index: usize = 0;
            for (i, di) in designators_and_initializers.iter().enumerate() {
                let effective_idx = if di.designators.is_empty() {
                    i
                } else {
                    // Check for array index designator [N].
                    let mut idx = i;
                    for d in &di.designators {
                        if let ast::Designator::Index(expr, _span) = d {
                            if let ast::Expression::IntegerLiteral { value, .. } = expr.as_ref() {
                                idx = *value as usize;
                            }
                        }
                    }
                    idx
                };
                if effective_idx >= max_index {
                    max_index = effective_idx + 1;
                }
            }
            max_index
        }
        ast::Initializer::Expression(expr_box) => {
            if let ast::Expression::StringLiteral {
                segments, prefix, ..
            } = expr_box.as_ref()
            {
                // Count the number of CHARACTERS (not bytes) for array sizing.
                // For narrow strings, each raw byte is one character.
                // For wide strings, raw bytes are UTF-8 sequences that must
                // be decoded to count the actual number of Unicode code points.
                let char_width: usize = match prefix {
                    ast::StringPrefix::None | ast::StringPrefix::U8 => 1,
                    ast::StringPrefix::L | ast::StringPrefix::U32 => 4,
                    ast::StringPrefix::U16 => 2,
                };
                let _ = elem_type;
                if char_width == 1 {
                    let total_bytes: usize = segments.iter().map(|s| s.value.len()).sum();
                    total_bytes + 1 // null terminator
                } else {
                    // For wide strings, decode UTF-8 to count code points.
                    let mut raw_bytes = Vec::new();
                    for seg in segments {
                        raw_bytes.extend_from_slice(&seg.value);
                    }
                    let codepoints = decode_bytes_to_codepoints(&raw_bytes);
                    codepoints.len() + 1 // +1 for null terminator
                }
            } else {
                0
            }
        }
    }
}

fn collect_locals_from_declaration(
    decl: &ast::Declaration,
    locals: &mut Vec<LocalVarInfo>,
    name_table: &[String],
) {
    let specifiers = &decl.specifiers;
    let storage_class = specifiers.storage_class;

    if matches!(storage_class, Some(ast::StorageClass::Extern)) {
        return;
    }

    // C11 §6.7.1p7 / §6.2.2p5: A function declaration at block scope has
    // external linkage. Skip collecting it as a local variable — the actual
    // function definition exists at file scope. This prevents `float fx();`
    // inside a function body from getting a stack alloca.
    if !matches!(storage_class, Some(ast::StorageClass::Typedef)) {
        // Check if ALL declarators in this declaration resolve to function types.
        // A mixed declaration like `float fx(), a;` should skip fx but keep a.
        // We'll handle this per-declarator below instead.
    }

    let mut is_static = matches!(storage_class, Some(ast::StorageClass::Static))
        || matches!(storage_class, Some(ast::StorageClass::ThreadLocal));

    // If the declaration is static but the initializer contains label addresses
    // (&&label — GCC computed goto), downgrade to stack-allocated. Label addresses
    // are function-local code pointers that cannot be resolved at link time as
    // global initializer constants; they must be materialized at runtime via LEA.
    if is_static {
        for init_decl in &decl.declarators {
            if let Some(ref init) = init_decl.initializer {
                if initializer_contains_label_address(init) {
                    is_static = false;
                    break;
                }
            }
        }
    }

    for init_decl in &decl.declarators {
        let declarator = &init_decl.declarator;
        let var_name = match extract_declarator_name(declarator, name_table) {
            Some(name) => name,
            None => continue,
        };

        let mut c_type =
            resolve_declaration_type(specifiers, declarator, &Target::X86_64, name_table);

        // C11 §6.7.1p7: block-scope function declarations have external
        // linkage and should not receive stack allocas.  Skip them so that
        // `float fx(), a;` collects `a` but not `fx`.
        if matches!(c_type, CType::Function { .. }) {
            continue;
        }

        let alignment = extract_alignment_attribute(&specifiers.attributes, name_table);

        // Infer array size from initializer for incomplete array types (`T arr[] = { ... }`).
        if let CType::Array(ref elem, None) = c_type {
            if let Some(ref init) = init_decl.initializer {
                let inferred_size = infer_array_size_from_initializer(init, elem);
                if inferred_size > 0 {
                    c_type = CType::Array(elem.clone(), Some(inferred_size));
                }
            }
        }

        let static_init = if is_static {
            init_decl.initializer.clone()
        } else {
            None
        };

        // Register the variable's resolved CType in the typeof resolution
        // context so that subsequent `typeof(var_name)` can find it.
        super::TYPEOF_CONTEXT.with(|ctx| {
            let mut borrow = ctx.borrow_mut();
            if let Some(ref mut map) = *borrow {
                map.insert(var_name.clone(), c_type.clone());
            }
        });

        // Detect VLA: if the type is Array(_, None) and the declarator
        // has a non-constant size expression, extract it for dynamic allocation.
        let vla_size_expr = if matches!(c_type, CType::Array(_, None)) && !is_static {
            extract_vla_size_expr(declarator)
        } else {
            None
        };

        locals.push(LocalVarInfo {
            name: var_name,
            c_type,
            is_static,
            has_initializer: init_decl.initializer.is_some(),
            alignment,
            span: decl.span,
            static_init,
            vla_size_expr,
        });
    }
}

// ===========================================================================
// Local variable allocation
// ===========================================================================

/// Extract the VLA size expression from a declarator.
/// For `int v[n]`, returns `Some(Box(n))`.  For constant-sized arrays, returns `None`.
fn extract_vla_size_expr(declarator: &ast::Declarator) -> Option<Box<ast::Expression>> {
    extract_vla_from_direct_decl(&declarator.direct)
}

/// Public variant for use from stmt_lowering.
pub fn extract_vla_size_expr_pub(declarator: &ast::Declarator) -> Option<Box<ast::Expression>> {
    extract_vla_size_expr(declarator)
}

fn extract_vla_from_direct_decl(dd: &ast::DirectDeclarator) -> Option<Box<ast::Expression>> {
    match dd {
        ast::DirectDeclarator::Array {
            size: Some(expr), ..
        } => {
            // Check if the expression is a compile-time constant.
            // If it IS constant, this is not a VLA.
            if evaluate_const_int_expr(expr).is_some() {
                None
            } else {
                Some(expr.clone())
            }
        }
        ast::DirectDeclarator::Parenthesized(inner) => extract_vla_size_expr(inner),
        _ => None,
    }
}

/// Extract ALL VLA dimension expressions from a (possibly multi-dimensional)
/// array declarator.  For `int a[n][m]`, returns `vec![n, m]` (outermost to
/// innermost).  Constant dimensions contribute their numeric value as an
/// integer literal expression.  Returns an empty Vec if no VLA dims found.
pub fn extract_all_vla_dims(declarator: &ast::Declarator) -> Vec<ast::Expression> {
    let mut dims = Vec::new();
    collect_array_dims(&declarator.direct, &mut dims);
    dims
}

fn collect_array_dims(dd: &ast::DirectDeclarator, dims: &mut Vec<ast::Expression>) {
    match dd {
        ast::DirectDeclarator::Array { base, size, .. } => {
            // Recurse into the base first (inner dimensions come first in
            // the AST: `a[outer][inner]` is parsed as Array(Array(Ident, inner), outer)).
            collect_array_dims(base, dims);
            // Then add this dimension.
            if let Some(expr) = size {
                dims.push(*expr.clone());
            }
        }
        ast::DirectDeclarator::Parenthesized(inner) => {
            collect_array_dims(&inner.direct, dims);
        }
        _ => {}
    }
}

fn allocate_local_variables(
    builder: &mut IrBuilder,
    function: &mut IrFunction,
    locals: &[&LocalVarInfo],
    target: &Target,
    local_vars: &mut FxHashMap<String, Value>,
) {
    for (idx, local) in locals.iter().enumerate() {
        // Skip VLAs — they are allocated dynamically via StackAlloc
        // at the point of declaration, not as a fixed-size alloca in the
        // entry block.  A placeholder Ptr alloca is created so the
        // variable has an entry in `local_vars`; the actual StackAlloc
        // pointer will be stored into this alloca at lowering time.
        if local.vla_size_expr.is_some() {
            // Create a pointer-sized alloca to hold the VLA pointer.
            let ptr_ty = IrType::Ptr;
            let (alloca_val, alloca_inst) = builder.build_alloca(ptr_ty.clone(), local.span);
            push_inst_to_entry(function, alloca_inst);
            local_vars.insert(local.name.clone(), alloca_val);

            let decl_line = if local.span.start > 0 {
                local.span.start
            } else {
                1
            };
            function
                .local_var_debug_info
                .push(crate::ir::function::LocalVarDebugInfo {
                    name: local.name.clone(),
                    ir_type: ptr_ty,
                    alloca_index: idx as u32,
                    decl_line,
                });
            continue;
        }
        // Resolve forward-referenced struct/union tag references so the
        // alloca size matches the full struct/union definition.
        let resolved_type = resolve_sizeof_struct_ref(local.c_type.clone(), target);
        let ir_type = IrType::from_ctype(&resolved_type, target);
        // Round up non-power-of-2 struct allocas so whole-struct register
        // stores (e.g. `struct s = foo();` returning in I32 for 3-byte
        // struct) do not overflow the alloca.
        let alloca_ir_type = {
            let sz = ir_type.size_bytes(target);
            if sz > 0 && sz <= 8 && !sz.is_power_of_two() {
                IrType::Array(Box::new(IrType::I8), sz.next_power_of_two())
            } else {
                ir_type.clone()
            }
        };
        let (alloca_val, alloca_inst) = builder.build_alloca(alloca_ir_type, local.span);
        push_inst_to_entry(function, alloca_inst);
        local_vars.insert(local.name.clone(), alloca_val);

        // Record debug metadata so the DWARF emitter can produce
        // DW_TAG_variable entries with proper names, types, and locations.
        let decl_line = if local.span.start > 0 {
            local.span.start
        } else {
            1
        };
        function
            .local_var_debug_info
            .push(crate::ir::function::LocalVarDebugInfo {
                name: local.name.clone(),
                ir_type,
                alloca_index: idx as u32,
                decl_line,
            });
    }
}

// ===========================================================================
// Static local and thread-local variable lowering
// ===========================================================================

fn lower_static_local(
    name: &str,
    func_name: &str,
    c_type: &CType,
    has_initializer: bool,
    init_expr: Option<&ast::Initializer>,
    module: &mut IrModule,
    target: &Target,
    type_builder: &TypeBuilder,
    diagnostics: &mut DiagnosticEngine,
    enum_constants: &FxHashMap<String, i128>,
) {
    let mangled_name = format!("{}.{}", func_name, name);

    // Attempt to extract a compile-time constant from the initializer.
    // Delegate to evaluate_initializer_constant so that function pointers,
    // address-of expressions, designated initializers, and other non-trivial
    // constant expressions are handled correctly — matching how global
    // variable initializers are processed.
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    let constant = if has_initializer {
        if let Some(init) = init_expr {
            evaluate_initializer_constant(
                init,
                c_type,
                target,
                type_builder,
                diagnostics,
                name_table,
                enum_constants,
            )
        } else {
            Some(Constant::ZeroInit)
        }
    } else {
        Some(Constant::ZeroInit)
    };

    // Infer array size from initializer when declaration uses [].
    let c_type = match (c_type, &constant) {
        (CType::Array(elem, None), Some(Constant::String(bytes))) => {
            let elem_byte_size = wide_char_elem_size(elem);
            CType::Array(elem.clone(), Some(bytes.len() / elem_byte_size))
        }
        (CType::Array(elem, None), Some(Constant::Array(elems))) => {
            CType::Array(elem.clone(), Some(elems.len()))
        }
        _ => c_type.clone(),
    };
    let ir_type = IrType::from_ctype(&c_type, target);

    // Post-process: convert Constant::String at Ptr-typed positions into
    // Constant::GlobalRef (same fixup applied to global variables).
    let constant = constant.map(|c| fixup_string_ptrs_in_constant(c, &ir_type, module));

    // Store the original C type so that lookup_var_type can retrieve the
    // accurate struct/union/typedef information instead of using the lossy
    // ir_type_to_approx_ctype reverse mapping (which turns IrType::Struct
    // into CType::Int).
    module
        .global_c_types
        .insert(mangled_name.clone(), c_type.clone());

    let mut global = GlobalVariable::new(mangled_name, ir_type, constant);
    global.linkage = Linkage::Internal;
    global.is_definition = true;
    module.add_global(global);
}

// ===========================================================================
// Function prologue/epilogue
// ===========================================================================

fn setup_function_prologue(_builder: &mut IrBuilder, _function: &mut IrFunction) {
    // Prologue is implicitly defined by alloca instructions in the entry block.
}

fn verify_function_termination(
    function: &mut IrFunction,
    return_type: &IrType,
    builder: &mut IrBuilder,
    diagnostics: &mut DiagnosticEngine,
) {
    let block_count = function.block_count();
    for idx in 0..block_count {
        if let Some(block) = function.get_block_mut(idx) {
            if !block.has_terminator() {
                if return_type.is_void() {
                    let ret_inst = builder.build_return(None, Span::dummy());
                    block.push_instruction(ret_inst);
                } else {
                    diagnostics
                        .emit_warning(Span::dummy(), "control reaches end of non-void function");
                    let ret_inst = builder.build_return(Some(Value::UNDEF), Span::dummy());
                    block.push_instruction(ret_inst);
                }
            }
        }
    }
}

// ===========================================================================
// Linkage and visibility helpers
// ===========================================================================

fn determine_linkage(
    storage_class: Option<ast::StorageClass>,
    attributes: &[ast::Attribute],
    name_table: &[String],
) -> Linkage {
    if has_attribute(attributes, "weak", name_table) {
        return Linkage::Weak;
    }
    match storage_class {
        Some(ast::StorageClass::Static) => Linkage::Internal,
        Some(ast::StorageClass::Extern) => Linkage::External,
        _ => Linkage::External,
    }
}

fn determine_visibility(attributes: &[ast::Attribute], name_table: &[String]) -> Visibility {
    for attr in attributes {
        if resolve_sym(name_table, &attr.name) == "visibility" {
            if let Some(first_arg) = attr.args.first() {
                match first_arg {
                    ast::AttributeArg::String(bytes, _) => {
                        let val = String::from_utf8_lossy(bytes);
                        match val.as_ref() {
                            "hidden" => return Visibility::Hidden,
                            "protected" => return Visibility::Protected,
                            "default" => return Visibility::Default,
                            _ => {}
                        }
                    }
                    ast::AttributeArg::Identifier(sym, _) => match resolve_sym(name_table, sym) {
                        "hidden" => return Visibility::Hidden,
                        "protected" => return Visibility::Protected,
                        "default" => return Visibility::Default,
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }
    Visibility::Default
}

/// Convert module::Linkage to function::Linkage.
fn convert_linkage_to_fn(linkage: Linkage) -> FnLinkage {
    match linkage {
        Linkage::External => FnLinkage::External,
        Linkage::Internal => FnLinkage::Internal,
        Linkage::Weak => FnLinkage::Weak,
        Linkage::Common => FnLinkage::External,
    }
}

/// Convert module::Visibility to function::Visibility.
fn convert_visibility_to_fn(vis: Visibility) -> FnVisibility {
    match vis {
        Visibility::Default => FnVisibility::Default,
        Visibility::Hidden => FnVisibility::Hidden,
        Visibility::Protected => FnVisibility::Protected,
    }
}

// ===========================================================================
// Attribute extraction helpers
// ===========================================================================

fn has_attribute(attributes: &[ast::Attribute], name: &str, name_table: &[String]) -> bool {
    attributes
        .iter()
        .any(|a| resolve_sym(name_table, &a.name) == name)
}

fn extract_section_attribute(
    attributes: &[ast::Attribute],
    name_table: &[String],
) -> Option<String> {
    for attr in attributes {
        if resolve_sym(name_table, &attr.name) == "section" {
            if let Some(first_arg) = attr.args.first() {
                match first_arg {
                    ast::AttributeArg::String(bytes, _) => {
                        return Some(String::from_utf8_lossy(bytes).into_owned());
                    }
                    ast::AttributeArg::Identifier(sym, _) => {
                        return Some(resolve_sym(name_table, sym).to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

/// Extract the target name from `__attribute__((alias("target")))`.
///
/// Returns `Some(target_name)` if the alias attribute is present, `None`
/// otherwise.
fn extract_alias_attribute(attributes: &[ast::Attribute], name_table: &[String]) -> Option<String> {
    for attr in attributes {
        if resolve_sym(name_table, &attr.name) == "alias" {
            if let Some(first_arg) = attr.args.first() {
                match first_arg {
                    ast::AttributeArg::String(bytes, _) => {
                        return Some(String::from_utf8_lossy(bytes).into_owned());
                    }
                    ast::AttributeArg::Identifier(sym, _) => {
                        return Some(resolve_sym(name_table, sym).to_string());
                    }
                    _ => {}
                }
            }
        }
    }
    None
}

/// Extract struct/union fields from the AST `StructOrUnionSpecifier` into
/// `StructField` entries that can be used in `CType::Struct` / `CType::Union`.
pub fn extract_struct_union_fields(spec: &ast::StructOrUnionSpecifier) -> Vec<StructField> {
    // Clone the thread-local name table once for this call (fallback path).
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    extract_struct_union_fields_fast(spec, name_table)
}

/// Performance-optimized struct/union field extraction.
/// Uses the provided `name_table` reference instead of cloning the interner snapshot.
pub fn extract_struct_union_fields_fast(
    spec: &ast::StructOrUnionSpecifier,
    name_table: &[String],
) -> Vec<StructField> {
    let mut fields = Vec::new();
    if let Some(ref members) = spec.members {
        for member in members {
            // Resolve base type from the member's specifier-qualifier list
            let member_base = resolve_base_type_from_sqlist_fast(&member.specifiers, name_table);
            if member.declarators.is_empty() {
                // Anonymous struct/union member (no declarators)
                fields.push(StructField {
                    name: None,
                    ty: member_base.clone(),
                    bit_width: None,
                });
            } else {
                for sd in &member.declarators {
                    let bit_width = sd
                        .bit_width
                        .as_ref()
                        .and_then(|e| evaluate_const_int_expr(e).map(|v| v as u32));
                    if let Some(ref declarator) = sd.declarator {
                        let name = extract_declarator_name(declarator, name_table);
                        // Apply pointer/array modifiers from declarator
                        let member_type =
                            apply_declarator_type(member_base.clone(), declarator, name_table);
                        fields.push(StructField {
                            name,
                            ty: member_type,
                            bit_width,
                        });
                    } else {
                        // Anonymous bitfield: `int : 3;`
                        fields.push(StructField {
                            name: None,
                            ty: member_base.clone(),
                            bit_width,
                        });
                    }
                }
            }
        }
    }
    fields
}

/// Resolve forward-referenced struct/union types: if a CType::Struct (or Union)
/// has a tag name but empty fields, replace it with the full definition from the
/// struct definitions registry.
fn resolve_struct_forward_ref(ctype: &mut CType, struct_defs: &FxHashMap<String, CType>) {
    // Resolve the outermost lightweight tag reference (empty fields +
    // named) by looking up the full definition in struct_defs.  We limit
    // recursion to avoid the quadratic/exponential expansion that occurs
    // when every nested struct field is eagerly expanded.  Deeper fields
    // are resolved on-demand via sizeof_resolved and field-offset lookups.
    resolve_struct_forward_ref_inner(ctype, struct_defs, 0);
}

/// Shallow forward-reference resolution: resolve only the outermost
/// lightweight tag reference (empty fields, named struct/union) by
/// replacing it with the full definition from struct_defs.  Does NOT
/// recurse into the resolved definition's fields — nested forward
/// references are resolved lazily during sizeof/field-offset lookups
/// via `sizeof_ctype_resolved` and `lookup_member_offset`.
fn resolve_struct_forward_ref_inner(
    ctype: &mut CType,
    struct_defs: &FxHashMap<String, CType>,
    _depth: usize,
) {
    match ctype {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                let has_fields = match full_def {
                    CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => {
                        !f.is_empty()
                    }
                    _ => true,
                };
                if has_fields {
                    *ctype = full_def.clone();
                    // Do NOT recurse into the resolved struct's fields.
                    // Nested lightweight references will be resolved lazily
                    // when field sizes or offsets are needed.
                }
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            if let Some(full_def) = struct_defs.get(tag.as_str()) {
                let has_fields = match full_def {
                    CType::Struct { fields: f, .. } | CType::Union { fields: f, .. } => {
                        !f.is_empty()
                    }
                    _ => true,
                };
                if has_fields {
                    *ctype = full_def.clone();
                }
            }
        }
        // For non-empty structs/unions, don't recurse into fields — the
        // definition already carries its field types and nested references
        // will be resolved lazily.
        CType::Struct { .. } | CType::Union { .. } => {}
        CType::Pointer(inner, _) => {
            // Recurse into the pointee type to resolve forward-referenced
            // struct/union tags.  This is essential for K&R-style function
            // definitions where `struct S *p` carries a lightweight tag
            // reference (Struct { name: "S", fields: [] }) as the pointee.
            resolve_struct_forward_ref_inner(inner, struct_defs, 0);
        }
        // Unwrap containers that might hold a forward-referenced tag.
        CType::Array(inner, _) => {
            resolve_struct_forward_ref_inner(inner, struct_defs, 0);
        }
        CType::Typedef {
            ref mut underlying, ..
        } => {
            resolve_struct_forward_ref_inner(underlying, struct_defs, 0);
        }
        CType::Atomic(inner) | CType::Complex(inner) => {
            resolve_struct_forward_ref_inner(inner, struct_defs, 0);
        }
        _ => {}
    }
}

/// Resolve a base C type from a specifier-qualifier list (used for struct members).
pub fn resolve_base_type_from_sqlist(sqlist: &ast::SpecifierQualifierList) -> CType {
    // Fallback for callers without a name_table — clones the interner snapshot.
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    resolve_base_type_from_sqlist_fast(sqlist, name_table)
}

/// Performance-optimized specifier-qualifier list resolution.
/// Uses the provided `name_table` reference to avoid O(n) cloning.
fn resolve_base_type_from_sqlist_fast(
    sqlist: &ast::SpecifierQualifierList,
    name_table: &[String],
) -> CType {
    let type_specs = &sqlist.type_specifiers;
    if type_specs.is_empty() {
        return CType::Int;
    }
    if type_specs.len() == 1 {
        return map_single_type_specifier_fast(&type_specs[0], name_table);
    }
    resolve_multi_word_type_fast(type_specs, name_table)
}

pub(super) fn extract_alignment_attribute(
    attributes: &[ast::Attribute],
    name_table: &[String],
) -> Option<usize> {
    for attr in attributes {
        let n = resolve_sym(name_table, &attr.name);
        if n == "aligned" || n == "__aligned__" {
            if let Some(ast::AttributeArg::Expression(expr)) = attr.args.first() {
                if let ast::Expression::IntegerLiteral { value, .. } = expr.as_ref() {
                    return Some(*value as usize);
                }
            }
            // aligned without argument defaults to max alignment (usually 16).
            return Some(16);
        }
    }
    None
}

/// Extract the maximum field-level `__attribute__((aligned(N)))` from all
/// members of a struct/union.  Per C11 + GCC extension, the alignment of a
/// struct is at least as strict as the alignment of any of its members.
/// When a member declaration carries `aligned(N)`, the struct's overall
/// alignment must be raised to at least N.
pub(super) fn extract_max_member_alignment(
    members: &[ast::StructMember],
    name_table: &[String],
) -> Option<usize> {
    let mut max_align: Option<usize> = None;
    for member in members {
        // Scan specifier-qualifier attributes (e.g. `int __attribute__((aligned(8))) a;`)
        if let Some(a) = extract_alignment_attribute(&member.specifiers.attributes, name_table) {
            max_align = Some(max_align.map_or(a, |cur: usize| cur.max(a)));
        }
        // Scan member-level attributes (after declarator list)
        if let Some(a) = extract_alignment_attribute(&member.attributes, name_table) {
            max_align = Some(max_align.map_or(a, |cur: usize| cur.max(a)));
        }
        // Scan declarator-level attributes
        for decl in &member.declarators {
            if let Some(ref d) = decl.declarator {
                if let Some(a) = extract_alignment_attribute(&d.attributes, name_table) {
                    max_align = Some(max_align.map_or(a, |cur: usize| cur.max(a)));
                }
            }
        }
    }
    max_align
}

fn collect_all_attributes(
    specifiers: &ast::DeclarationSpecifiers,
    func_attrs: &[ast::Attribute],
    declarator: &ast::Declarator,
) -> Vec<ast::Attribute> {
    let mut all = Vec::new();
    all.extend_from_slice(&specifiers.attributes);
    all.extend_from_slice(func_attrs);
    all.extend_from_slice(&declarator.attributes);
    all
}

// ===========================================================================
// Type resolution helpers
// ===========================================================================

/// Fallback for callers without a name_table — clones the interner snapshot.
#[allow(dead_code)]
fn resolve_base_type(specifiers: &ast::DeclarationSpecifiers, _target: &Target) -> CType {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    resolve_base_type_fast(specifiers, name_table)
}

/// Performance-optimized base type resolution.
/// Uses the provided `name_table` reference to avoid O(n) cloning per call.
#[inline]
pub fn resolve_base_type_fast(
    specifiers: &ast::DeclarationSpecifiers,
    name_table: &[String],
) -> CType {
    let type_specs = &specifiers.type_specifiers;
    let mut base = if type_specs.is_empty() {
        CType::Int
    } else if type_specs.len() == 1 {
        map_single_type_specifier_fast(&type_specs[0], name_table)
    } else {
        resolve_multi_word_type_fast(type_specs, name_table)
    };
    // Apply __attribute__((mode(QI/HI/SI/DI/TI/SF/DF/XF))) if present.
    base = apply_mode_attribute(base, &specifiers.attributes, name_table);
    base
}

/// Apply GCC `__attribute__((mode(...)))` to change the integer/float
/// size of a type.  QI=1, HI=2, SI=4, DI=8, TI=16, SF=float, DF=double, XF=long double.
fn apply_mode_attribute(
    base: CType,
    attributes: &[ast::Attribute],
    name_table: &[String],
) -> CType {
    for attr in attributes {
        let n = resolve_sym(name_table, &attr.name);
        if n == "mode" || n == "__mode__" {
            if let Some(arg) = attr.args.first() {
                let mode_name = match arg {
                    ast::AttributeArg::Identifier(sym, _) => resolve_sym(name_table, sym),
                    _ => continue,
                };
                let is_unsigned = matches!(
                    base,
                    CType::UChar
                        | CType::UShort
                        | CType::UInt
                        | CType::ULong
                        | CType::ULongLong
                        | CType::UInt128
                        | CType::Bool
                );
                let is_float = matches!(base, CType::Float | CType::Double | CType::LongDouble);
                return match mode_name {
                    "QI" | "__QI__" | "byte" | "__byte__" => {
                        if is_unsigned {
                            CType::UChar
                        } else {
                            CType::SChar
                        }
                    }
                    "HI" | "__HI__" => {
                        if is_unsigned {
                            CType::UShort
                        } else {
                            CType::Short
                        }
                    }
                    "SI" | "__SI__" | "word" | "__word__" => {
                        if is_float {
                            CType::Float
                        } else if is_unsigned {
                            CType::UInt
                        } else {
                            CType::Int
                        }
                    }
                    "DI" | "__DI__" => {
                        if is_float {
                            CType::Double
                        } else if is_unsigned {
                            CType::ULongLong
                        } else {
                            CType::LongLong
                        }
                    }
                    "TI" | "__TI__" => {
                        if is_unsigned {
                            CType::UInt128
                        } else {
                            CType::Int128
                        }
                    }
                    "SF" | "__SF__" => CType::Float,
                    "DF" | "__DF__" => CType::Double,
                    "XF" | "__XF__" => CType::LongDouble,
                    "pointer" | "__pointer__" => {
                        // mode(pointer) = pointer-width integer (8 bytes on 64-bit)
                        if is_unsigned {
                            CType::ULong
                        } else {
                            CType::Long
                        }
                    }
                    _ => base,
                };
            }
        }
    }
    base
}

/// Resolve a Symbol to its actual string name using the name_table.
/// Returns the symbol string if the index is in range, otherwise the
/// debug representation.
fn sym_to_string(sym: &crate::common::string_interner::Symbol, name_table: &[String]) -> String {
    let idx = sym.as_u32() as usize;
    if idx < name_table.len() {
        name_table[idx].clone()
    } else {
        sym.to_string()
    }
}

/// Delegate to the name_table-aware version using the thread-local snapshot.
/// NOTE: This clones the snapshot — callers in hot paths should prefer
/// `map_single_type_specifier_with_names` with an already-borrowed table.
#[allow(dead_code)]
fn map_single_type_specifier(spec: &ast::TypeSpecifier) -> CType {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    map_single_type_specifier_with_names(spec, name_table)
}

/// Performance-optimized base type resolution for a single type specifier.
/// Uses the provided `name_table` reference to avoid cloning the interner snapshot.
#[inline]
fn map_single_type_specifier_fast(spec: &ast::TypeSpecifier, name_table: &[String]) -> CType {
    map_single_type_specifier_with_names(spec, name_table)
}

// ---------------------------------------------------------------------------
// typeof resolution helpers
// ---------------------------------------------------------------------------

/// Resolve a `typeof(...)` argument to a concrete CType.
fn resolve_typeof_arg(arg: &ast::TypeofArg, name_table: &[String]) -> CType {
    match arg {
        ast::TypeofArg::TypeName(tn) => {
            let base = resolve_base_type_from_sqlist_fast(&tn.specifier_qualifiers, name_table);
            if let Some(ref abs) = tn.abstract_declarator {
                apply_abstract_declarator_to_type(base, abs)
            } else {
                base
            }
        }
        ast::TypeofArg::Expression(expr) => infer_typeof_expr_ctype(expr, name_table),
    }
}

/// Infer the C type of an expression for `typeof(expr)`.
///
/// This mirrors `SemanticAnalyzer::infer_typeof_expr_type` but operates in
/// the IR lowering context where the only type information available is the
/// thread-local `TYPEOF_CONTEXT` (variable name → CType map).
/// Resolve a struct/union member type for typeof inference.
/// Given a struct/union type and a member symbol, look through the member
/// definitions to find the matching member's type.
fn infer_typeof_resolve_member(
    struct_ty: &CType,
    member: crate::common::string_interner::Symbol,
    name_table: &[String],
) -> CType {
    let member_name = resolve_sym(name_table, &member);
    let resolved = crate::common::types::resolve_typedef(struct_ty);
    match resolved {
        CType::Struct { ref fields, .. } | CType::Union { ref fields, .. } => {
            for f in fields {
                if let Some(ref fname) = f.name {
                    if fname.as_str() == member_name {
                        return f.ty.clone();
                    }
                }
                // Check anonymous struct/union members recursively.
                if f.name.is_none() {
                    let inner = infer_typeof_resolve_member(&f.ty, member, name_table);
                    if !matches!(inner, CType::Int) {
                        return inner;
                    }
                }
            }
            CType::Int
        }
        _ => CType::Int,
    }
}

fn infer_typeof_expr_ctype(expr: &ast::Expression, name_table: &[String]) -> CType {
    match expr {
        // Identifier — look up in the typeof resolution context.
        ast::Expression::Identifier { name, .. } => {
            let var_name = resolve_sym(name_table, name).to_string();
            super::TYPEOF_CONTEXT.with(|ctx| {
                let borrow = ctx.borrow();
                borrow
                    .as_ref()
                    .and_then(|map| map.get(&var_name).cloned())
                    .unwrap_or(CType::Int)
            })
        }

        // Integer literals — type depends on suffix.
        ast::Expression::IntegerLiteral { suffix, .. } => match suffix {
            ast::IntegerSuffix::None => CType::Int,
            ast::IntegerSuffix::U => CType::UInt,
            ast::IntegerSuffix::L => CType::Long,
            ast::IntegerSuffix::UL => CType::ULong,
            ast::IntegerSuffix::LL => CType::LongLong,
            ast::IntegerSuffix::ULL => CType::ULongLong,
        },

        // Float literals.
        ast::Expression::FloatLiteral { .. } => CType::Double,

        // Char literals.
        ast::Expression::CharLiteral { .. } => CType::Int,

        // String literals → pointer to char.
        ast::Expression::StringLiteral { .. } => CType::Pointer(
            Box::new(CType::Char),
            crate::common::types::TypeQualifiers::default(),
        ),

        // Dereference: *ptr → pointee type.
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand,
            ..
        } => {
            let inner = infer_typeof_expr_ctype(operand, name_table);
            match inner {
                CType::Pointer(pointee, _) => *pointee,
                CType::Array(elem, _) => *elem,
                _ => CType::Int,
            }
        }

        // Address-of: &x → pointer to x's type.
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::AddressOf,
            operand,
            ..
        } => {
            let inner = infer_typeof_expr_ctype(operand, name_table);
            CType::Pointer(
                Box::new(inner),
                crate::common::types::TypeQualifiers::default(),
            )
        }

        // Cast expression: (type)expr → the cast target type.
        ast::Expression::Cast { type_name, .. } => {
            let base =
                resolve_base_type_from_sqlist_fast(&type_name.specifier_qualifiers, name_table);
            if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(base, abs)
            } else {
                base
            }
        }

        // sizeof always yields size_t (unsigned long on 64-bit).
        ast::Expression::SizeofExpr { .. } | ast::Expression::SizeofType { .. } => CType::ULong,

        // Logical not always yields int.
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::LogicalNot,
            ..
        } => CType::Int,

        // Other unary ops: preserve operand type.
        ast::Expression::UnaryOp { operand, .. } => infer_typeof_expr_ctype(operand, name_table),

        // Binary ops — simplistic: use left operand type (covers common
        // arithmetic cases; full usual-arithmetic-conversion is in sema).
        ast::Expression::Binary { left, op, .. } => {
            if matches!(
                op,
                ast::BinaryOp::Equal
                    | ast::BinaryOp::NotEqual
                    | ast::BinaryOp::Less
                    | ast::BinaryOp::LessEqual
                    | ast::BinaryOp::Greater
                    | ast::BinaryOp::GreaterEqual
                    | ast::BinaryOp::LogicalAnd
                    | ast::BinaryOp::LogicalOr
            ) {
                CType::Int
            } else {
                infer_typeof_expr_ctype(left, name_table)
            }
        }

        // Ternary: use the "then" branch type.
        ast::Expression::Conditional { then_expr, .. } => {
            if let Some(then_e) = then_expr {
                infer_typeof_expr_ctype(then_e, name_table)
            } else {
                CType::Int
            }
        }

        // Array subscript (a[i]): get element type from base.
        ast::Expression::ArraySubscript { base, .. } => {
            let arr_ty = infer_typeof_expr_ctype(base, name_table);
            match &arr_ty {
                CType::Array(elem, _) => (**elem).clone(),
                CType::Pointer(pointee, _) => (**pointee).clone(),
                _ => arr_ty,
            }
        }

        // Member access (s.member): resolve struct type and find member.
        ast::Expression::MemberAccess { object, member, .. } => {
            let obj_ty = infer_typeof_expr_ctype(object, name_table);
            infer_typeof_resolve_member(&obj_ty, *member, name_table)
        }

        // Pointer member access (p->member): dereference then find member.
        ast::Expression::PointerMemberAccess { object, member, .. } => {
            let ptr_ty = infer_typeof_expr_ctype(object, name_table);
            let obj_ty = match &ptr_ty {
                CType::Pointer(inner, _) => (**inner).clone(),
                CType::Array(inner, _) => (**inner).clone(),
                other => other.clone(),
            };
            infer_typeof_resolve_member(&obj_ty, *member, name_table)
        }

        // Post/pre increment/decrement: type of the operand.
        ast::Expression::PostIncrement { operand, .. }
        | ast::Expression::PostDecrement { operand, .. }
        | ast::Expression::PreIncrement { operand, .. }
        | ast::Expression::PreDecrement { operand, .. } => {
            infer_typeof_expr_ctype(operand, name_table)
        }

        // Compound literal: resolve the type name.
        ast::Expression::CompoundLiteral { type_name, .. } => {
            let base =
                resolve_base_type_from_sqlist_fast(&type_name.specifier_qualifiers, name_table);
            if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(base, abs)
            } else {
                base
            }
        }

        // Function call: for now return Int (full resolution would require
        // looking up the function return type).
        ast::Expression::FunctionCall { .. } => CType::Int,

        // Comma: type of the right operand.
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                infer_typeof_expr_ctype(last, name_table)
            } else {
                CType::Int
            }
        }

        // Parenthesized: transparent.
        ast::Expression::Parenthesized { inner, .. } => infer_typeof_expr_ctype(inner, name_table),

        // Statement expression — the type is determined by the last expression
        // in the compound block.  We must resolve identifiers declared inside
        // the statement expression from the compound's local declarations.
        ast::Expression::StatementExpression { compound, .. } => {
            if let Some(ast::BlockItem::Statement(ast::Statement::Expression(Some(
                ref inner_expr,
            )))) = compound.items.last()
            {
                // Build a local type map from compound declarations.
                let mut local_types = crate::common::fx_hash::FxHashMap::default();
                for item in &compound.items {
                    if let ast::BlockItem::Declaration(decl) = item {
                        let base = resolve_base_type_fast(&decl.specifiers, name_table);
                        for id in &decl.declarators {
                            if let Some(sym_name) =
                                extract_declarator_sym_name(&id.declarator, name_table)
                            {
                                let full_ty =
                                    apply_declarator_type(base.clone(), &id.declarator, name_table);
                                local_types.insert(sym_name, full_ty);
                            }
                        }
                    }
                }
                infer_typeof_with_stmt_locals(inner_expr, name_table, &local_types)
            } else {
                CType::Int
            }
        }

        // Default fallback.
        _ => CType::Int,
    }
}

/// Extract the declared name as a String from a declarator.
fn extract_declarator_sym_name(
    declarator: &ast::Declarator,
    name_table: &[String],
) -> Option<String> {
    extract_dd_sym_name(&declarator.direct, name_table)
}

fn extract_dd_sym_name(dd: &ast::DirectDeclarator, name_table: &[String]) -> Option<String> {
    match dd {
        ast::DirectDeclarator::Identifier(sym, _) => Some(resolve_sym(name_table, sym).to_string()),
        ast::DirectDeclarator::Parenthesized(inner) => {
            extract_declarator_sym_name(inner, name_table)
        }
        ast::DirectDeclarator::Array { base, .. } => extract_dd_sym_name(base, name_table),
        ast::DirectDeclarator::Function { base, .. } => extract_dd_sym_name(base, name_table),
    }
}

/// Infer typeof expression type with additional local declarations from a
/// statement expression compound block.  Identifiers are resolved first
/// from `locals`, then from TYPEOF_CONTEXT, matching the scoping semantics
/// of GCC's statement expressions used in min/max macros.
fn infer_typeof_with_stmt_locals(
    expr: &ast::Expression,
    name_table: &[String],
    locals: &crate::common::fx_hash::FxHashMap<String, CType>,
) -> CType {
    match expr {
        ast::Expression::Identifier { name, .. } => {
            let var_name = resolve_sym(name_table, name).to_string();
            // Check locals first
            if let Some(ty) = locals.get(&var_name) {
                return ty.clone();
            }
            // Fall back to TYPEOF_CONTEXT
            super::TYPEOF_CONTEXT.with(|ctx| {
                let borrow = ctx.borrow();
                borrow
                    .as_ref()
                    .and_then(|map| map.get(&var_name).cloned())
                    .unwrap_or(CType::Int)
            })
        }
        ast::Expression::Conditional {
            then_expr,
            else_expr,
            ..
        } => {
            if let Some(then_e) = then_expr {
                infer_typeof_with_stmt_locals(then_e, name_table, locals)
            } else {
                infer_typeof_with_stmt_locals(else_expr, name_table, locals)
            }
        }
        ast::Expression::Parenthesized { inner, .. } => {
            infer_typeof_with_stmt_locals(inner, name_table, locals)
        }
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                infer_typeof_with_stmt_locals(last, name_table, locals)
            } else {
                CType::Int
            }
        }
        ast::Expression::Binary { left, op, .. } => {
            if matches!(
                op,
                ast::BinaryOp::Equal
                    | ast::BinaryOp::NotEqual
                    | ast::BinaryOp::Less
                    | ast::BinaryOp::LessEqual
                    | ast::BinaryOp::Greater
                    | ast::BinaryOp::GreaterEqual
                    | ast::BinaryOp::LogicalAnd
                    | ast::BinaryOp::LogicalOr
            ) {
                CType::Int
            } else {
                infer_typeof_with_stmt_locals(left, name_table, locals)
            }
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand,
            ..
        } => {
            let inner = infer_typeof_with_stmt_locals(operand, name_table, locals);
            match inner {
                CType::Pointer(pointee, _) | CType::Array(pointee, _) => *pointee,
                _ => CType::Int,
            }
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::AddressOf,
            operand,
            ..
        } => {
            let inner = infer_typeof_with_stmt_locals(operand, name_table, locals);
            CType::Pointer(
                Box::new(inner),
                crate::common::types::TypeQualifiers::default(),
            )
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::LogicalNot,
            ..
        } => CType::Int,
        ast::Expression::UnaryOp { operand, .. } => {
            infer_typeof_with_stmt_locals(operand, name_table, locals)
        }
        ast::Expression::Cast { type_name, .. } => {
            let base =
                resolve_base_type_from_sqlist_fast(&type_name.specifier_qualifiers, name_table);
            if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(base, abs)
            } else {
                base
            }
        }
        ast::Expression::PostIncrement { operand, .. }
        | ast::Expression::PostDecrement { operand, .. }
        | ast::Expression::PreIncrement { operand, .. }
        | ast::Expression::PreDecrement { operand, .. } => {
            infer_typeof_with_stmt_locals(operand, name_table, locals)
        }
        // Array subscript (a[i]): get element type from base.
        ast::Expression::ArraySubscript { base, .. } => {
            let arr_ty = infer_typeof_with_stmt_locals(base, name_table, locals);
            match &arr_ty {
                CType::Array(elem, _) => (**elem).clone(),
                CType::Pointer(pointee, _) => (**pointee).clone(),
                _ => arr_ty,
            }
        }
        // Member access (s.member): resolve struct type and find member.
        ast::Expression::MemberAccess { object, member, .. } => {
            let obj_ty = infer_typeof_with_stmt_locals(object, name_table, locals);
            infer_typeof_resolve_member(&obj_ty, *member, name_table)
        }
        // Pointer member access (p->member): dereference then find member.
        ast::Expression::PointerMemberAccess { object, member, .. } => {
            let ptr_ty = infer_typeof_with_stmt_locals(object, name_table, locals);
            let obj_ty = match &ptr_ty {
                CType::Pointer(inner, _) => (**inner).clone(),
                CType::Array(inner, _) => (**inner).clone(),
                other => other.clone(),
            };
            infer_typeof_resolve_member(&obj_ty, *member, name_table)
        }
        // Compound literal: resolve the type name.
        ast::Expression::CompoundLiteral { type_name, .. } => {
            let base =
                resolve_base_type_from_sqlist_fast(&type_name.specifier_qualifiers, name_table);
            if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(base, abs)
            } else {
                base
            }
        }
        // sizeof always yields size_t (unsigned long on 64-bit).
        ast::Expression::SizeofExpr { .. } | ast::Expression::SizeofType { .. } => CType::ULong,
        // Nested statement expression — collect inner locals and recurse.
        ast::Expression::StatementExpression { compound, .. } => {
            if let Some(ast::BlockItem::Statement(ast::Statement::Expression(Some(
                ref inner_expr,
            )))) = compound.items.last()
            {
                let mut inner_locals = locals.clone();
                for item in &compound.items {
                    if let ast::BlockItem::Declaration(decl) = item {
                        let base = resolve_base_type_fast(&decl.specifiers, name_table);
                        for id in &decl.declarators {
                            if let Some(sym_name) =
                                extract_declarator_sym_name(&id.declarator, name_table)
                            {
                                let full_ty =
                                    apply_declarator_type(base.clone(), &id.declarator, name_table);
                                inner_locals.insert(sym_name, full_ty);
                            }
                        }
                    }
                }
                infer_typeof_with_stmt_locals(inner_expr, name_table, &inner_locals)
            } else {
                CType::Int
            }
        }
        // For anything else, delegate to the standard typeof inference.
        _ => infer_typeof_expr_ctype(expr, name_table),
    }
}

fn map_single_type_specifier_with_names(spec: &ast::TypeSpecifier, name_table: &[String]) -> CType {
    match spec {
        ast::TypeSpecifier::Void => CType::Void,
        ast::TypeSpecifier::Char => CType::Char,
        ast::TypeSpecifier::Short => CType::Short,
        ast::TypeSpecifier::Int => CType::Int,
        ast::TypeSpecifier::Long => CType::Long,
        ast::TypeSpecifier::Float => CType::Float,
        ast::TypeSpecifier::Double => CType::Double,
        ast::TypeSpecifier::Bool => CType::Bool,
        ast::TypeSpecifier::Signed => CType::Int,
        ast::TypeSpecifier::Unsigned => CType::UInt,
        ast::TypeSpecifier::Int128 => CType::Int128,
        ast::TypeSpecifier::Float128 => CType::LongDouble,
        ast::TypeSpecifier::Float64 => CType::Double,
        ast::TypeSpecifier::Float32 => CType::Float,
        ast::TypeSpecifier::Float16 => CType::Float,
        ast::TypeSpecifier::Struct(s) => {
            let fields = extract_struct_union_fields_fast(s, name_table);
            let packed = s.attributes.iter().any(|a| {
                let n = resolve_sym(name_table, &a.name);
                n == "packed" || n == "__packed__"
            });
            let mut aligned = extract_alignment_attribute(&s.attributes, name_table);
            // Also propagate field-level __attribute__((aligned(N))) to the
            // struct's overall alignment (C11 + GCC extension).
            if let Some(ref members) = s.members {
                if let Some(field_align) = extract_max_member_alignment(members, name_table) {
                    aligned = Some(aligned.map_or(field_align, |a| a.max(field_align)));
                }
            }
            CType::Struct {
                name: s.tag.as_ref().map(|t| sym_to_string(t, name_table)),
                fields,
                packed,
                aligned,
            }
        }
        ast::TypeSpecifier::Union(u) => {
            let fields = extract_struct_union_fields_fast(u, name_table);
            let packed = u.attributes.iter().any(|a| {
                let n = resolve_sym(name_table, &a.name);
                n == "packed" || n == "__packed__"
            });
            let mut aligned = extract_alignment_attribute(&u.attributes, name_table);
            // Also propagate field-level aligned to union alignment.
            if let Some(ref members) = u.members {
                if let Some(field_align) = extract_max_member_alignment(members, name_table) {
                    aligned = Some(aligned.map_or(field_align, |a| a.max(field_align)));
                }
            }
            CType::Union {
                name: u.tag.as_ref().map(|t| sym_to_string(t, name_table)),
                fields,
                packed,
                aligned,
            }
        }
        ast::TypeSpecifier::Enum(e) => {
            let tag_name = e.tag.as_ref().map(|t| sym_to_string(t, name_table));
            // Look up the correct underlying type from the
            // ENUM_UNDERLYING_TYPES registry (populated in pass 0.5).
            // GCC treats enum bitfields as unsigned when all values ≥ 0.
            let underlying = if let Some(ref tn) = tag_name {
                super::ENUM_UNDERLYING_TYPES.with(|m| {
                    let borrow = m.borrow();
                    borrow
                        .as_ref()
                        .and_then(|map| map.get(tn).cloned())
                        .unwrap_or(CType::Int)
                })
            } else if let Some(ref enumerators) = e.enumerators {
                // Anonymous enum — compute underlying from values inline.
                let mut all_non_negative = true;
                let mut next_val: i128 = 0;
                for en in enumerators {
                    if let Some(ref val_expr) = en.value {
                        if let Some(v) = evaluate_const_int_expr(val_expr) {
                            next_val = v as i128;
                        }
                    }
                    if next_val < 0 {
                        all_non_negative = false;
                    }
                    next_val += 1;
                }
                if all_non_negative {
                    CType::UInt
                } else {
                    CType::Int
                }
            } else {
                CType::Int
            };
            CType::Enum {
                name: tag_name,
                underlying_type: Box::new(underlying),
            }
        }
        ast::TypeSpecifier::TypedefName(name) => {
            let td_name = sym_to_string(name, name_table);
            // Look up the resolved underlying type from the typedef map
            // populated during Pass 0.5 of lower_translation_unit.
            let underlying = super::TYPEDEF_MAP.with(|map| {
                let borrow = map.borrow();
                borrow
                    .as_ref()
                    .and_then(|m| m.get(&td_name).cloned())
                    .unwrap_or(CType::Int)
            });
            CType::Typedef {
                name: td_name,
                underlying: Box::new(underlying),
            }
        }
        ast::TypeSpecifier::Typeof(arg) => resolve_typeof_arg(arg, name_table),
        ast::TypeSpecifier::Atomic(_) => CType::Atomic(Box::new(CType::Int)),
        ast::TypeSpecifier::Complex => CType::Complex(Box::new(CType::Double)),
        ast::TypeSpecifier::AutoType => {
            // __auto_type — type will be inferred from initializer.
            // Return Int as a placeholder; the actual type gets resolved
            // during variable initialization lowering.
            CType::Int
        }
    }
}

#[allow(dead_code)]
fn resolve_multi_word_type(specs: &[ast::TypeSpecifier]) -> CType {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    resolve_multi_word_type_fast(specs, name_table)
}

/// Performance-optimized multi-word type resolution.
/// Avoids cloning the interner snapshot per call by using a borrowed name_table.
fn resolve_multi_word_type_fast(specs: &[ast::TypeSpecifier], name_table: &[String]) -> CType {
    let mut has_unsigned = false;
    let mut has_signed = false;
    let mut long_count = 0u32;
    let mut has_short = false;
    let mut has_char = false;
    let mut has_double = false;
    let mut has_float = false;
    let mut has_complex = false;

    for spec in specs {
        match spec {
            ast::TypeSpecifier::Unsigned => has_unsigned = true,
            ast::TypeSpecifier::Signed => has_signed = true,
            ast::TypeSpecifier::Long => long_count += 1,
            ast::TypeSpecifier::Short => has_short = true,
            ast::TypeSpecifier::Char => has_char = true,
            ast::TypeSpecifier::Int => {}
            ast::TypeSpecifier::Double => has_double = true,
            ast::TypeSpecifier::Float => has_float = true,
            ast::TypeSpecifier::Complex => has_complex = true,
            other => return map_single_type_specifier_fast(other, name_table),
        }
    }

    if has_complex {
        let base = if has_float {
            CType::Float
        } else if has_double && long_count > 0 {
            CType::LongDouble
        } else if has_double {
            CType::Double
        } else if has_char {
            if has_unsigned {
                CType::UChar
            } else if has_signed {
                CType::SChar
            } else {
                CType::Char
            }
        } else if has_short {
            if has_unsigned {
                CType::UShort
            } else {
                CType::Short
            }
        } else if long_count >= 2 {
            if has_unsigned {
                CType::ULongLong
            } else {
                CType::LongLong
            }
        } else if long_count == 1 {
            if has_unsigned {
                CType::ULong
            } else {
                CType::Long
            }
        } else if has_unsigned {
            CType::UInt
        } else {
            // Default: _Complex with `int`, `signed`, or bare _Complex → double
            // For `_Complex int` or `_Complex signed int`: use Int
            // For bare `_Complex`: use Double (C11 default)
            // We check has_signed because `_Complex signed` → _Complex int
            if has_signed || specs.iter().any(|s| matches!(s, ast::TypeSpecifier::Int)) {
                CType::Int
            } else {
                CType::Double
            }
        };
        return CType::Complex(Box::new(base));
    }
    if has_char {
        return if has_unsigned {
            CType::UChar
        } else if has_signed {
            CType::SChar
        } else {
            CType::Char
        };
    }
    if has_short {
        return if has_unsigned {
            CType::UShort
        } else {
            CType::Short
        };
    }
    if long_count >= 2 {
        return if has_unsigned {
            CType::ULongLong
        } else {
            CType::LongLong
        };
    }
    if long_count == 1 {
        if has_double {
            return CType::LongDouble;
        }
        return if has_unsigned {
            CType::ULong
        } else {
            CType::Long
        };
    }
    if has_double {
        return CType::Double;
    }
    if has_float {
        return CType::Float;
    }
    if has_unsigned {
        CType::UInt
    } else {
        CType::Int
    }
}

pub fn resolve_declaration_type(
    specifiers: &ast::DeclarationSpecifiers,
    declarator: &ast::Declarator,
    _target: &Target,
    name_table: &[String],
) -> CType {
    let base = resolve_base_type_fast(specifiers, name_table);
    apply_declarator_type(base, declarator, name_table)
}

fn apply_declarator_type(
    base: CType,
    declarator: &ast::Declarator,
    name_table: &[String],
) -> CType {
    // C declarator semantics: pointer modifiers apply to the base type FIRST,
    // producing the "pointed base".  Direct-declarator modifiers (array, function)
    // then wrap the pointed base.
    //
    // `void *x[3]`  →  pointer(void) → array(pointer(void), 3)
    // `void  x[3]`  →  void          → array(void, 3)
    // `int  *f(int)` →  pointer(int)  → function(pointer(int), [int])
    //
    // For parenthesized declarators like `void (*x)[3]`, the pointer is
    // *inside* the parenthesized inner declarator and is handled recursively
    // by apply_direct_declarator's Parenthesized branch.
    let modified_base = if let Some(ref pointer) = declarator.pointer {
        apply_pointer_layers(base, pointer)
    } else {
        base
    };
    apply_direct_declarator(modified_base, &declarator.direct, name_table)
}

fn apply_pointer_layers(base: CType, pointer: &ast::Pointer) -> CType {
    let quals = crate::common::types::TypeQualifiers::default();
    let mut current = CType::Pointer(Box::new(base), quals);
    if let Some(ref inner) = pointer.inner {
        current = apply_pointer_layers(current, inner);
    }
    current
}

fn apply_direct_declarator(
    base: CType,
    direct: &ast::DirectDeclarator,
    name_table: &[String],
) -> CType {
    match direct {
        ast::DirectDeclarator::Identifier(_, _) => base,
        ast::DirectDeclarator::Parenthesized(inner) => {
            apply_declarator_type(base, inner, name_table)
        }
        ast::DirectDeclarator::Array {
            base: inner_dd,
            size,
            ..
        } => {
            let array_size = size.as_ref().and_then(|e| evaluate_const_int_expr(e));
            match inner_dd.as_ref() {
                // Parenthesized base: `void (*x)[3]`
                // The array wraps the base type: Array(void, 3)
                // Then the inner declarator (which has the pointer) wraps the
                // array type: Pointer(Array(void, 3))
                ast::DirectDeclarator::Parenthesized(inner_decl) => {
                    let array_type = CType::Array(Box::new(base), array_size);
                    apply_declarator_type(array_type, inner_decl, name_table)
                }
                // Non-parenthesized: e.g. `int x[3]` or `int x[3][4]`
                // Build Array(base, current_size) first, then recurse into
                // inner_dd so that multi-dimensional arrays nest correctly:
                //   `int a[3][4]` → Array(Array(int, 4), 3)
                // The outermost AST node's size (4, parsed last) corresponds
                // to the innermost C dimension, so we wrap base with it
                // first and let the recursion wrap with the outer dimension.
                _ => {
                    let arr_type = CType::Array(Box::new(base), array_size);
                    apply_direct_declarator(arr_type, inner_dd, name_table)
                }
            }
        }
        ast::DirectDeclarator::Function {
            base: inner_dd,
            params,
            is_variadic,
            ..
        } => {
            let param_types: Vec<CType> = params
                .iter()
                .map(|p| resolve_param_type(p, &Target::X86_64, name_table))
                .collect();
            // C declarator syntax is inside-out. For function pointers like
            // `int (*op)(int, int)`, the AST has:
            //   Function { base: Parenthesized(Declarator{pointer:*, direct:Ident("op")}), params }
            //
            // The pointer `*` wraps the FUNCTION type, NOT the return type.
            // So we must first construct the function type with `base` as its
            // return type, then apply the inner declarator's modifiers (pointer,
            // array, etc.) to the completed function type.
            match inner_dd.as_ref() {
                ast::DirectDeclarator::Parenthesized(inner_decl) => {
                    let func_type = CType::Function {
                        return_type: Box::new(base),
                        params: param_types,
                        variadic: *is_variadic,
                    };
                    apply_declarator_type(func_type, inner_decl, name_table)
                }
                _ => {
                    let return_type = apply_direct_declarator(base, inner_dd, name_table);
                    CType::Function {
                        return_type: Box::new(return_type),
                        params: param_types,
                        variadic: *is_variadic,
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Parameter and name helpers
// ===========================================================================

fn extract_function_params(declarator: &ast::Declarator) -> (Vec<ast::ParameterDeclaration>, bool) {
    extract_function_params_from_dd(&declarator.direct)
}

fn extract_function_params_from_dd(
    dd: &ast::DirectDeclarator,
) -> (Vec<ast::ParameterDeclaration>, bool) {
    match dd {
        ast::DirectDeclarator::Function {
            base,
            params,
            is_variadic,
            ..
        } => {
            // For function-returning-function-pointer patterns like:
            //   void (*memdbDlSym(real_params))(return_type_params)
            // The outer Function's params belong to the returned function pointer,
            // NOT the actual function parameters. The real params are in the inner
            // nested Function declarator. Detect and recurse.
            if let ast::DirectDeclarator::Parenthesized(inner_decl) = base.as_ref() {
                if dd_has_nested_function(&inner_decl.direct) {
                    return extract_function_params(inner_decl);
                }
            }
            (params.clone(), *is_variadic)
        }
        ast::DirectDeclarator::Parenthesized(inner) => extract_function_params(inner),
        _ => (Vec::new(), false),
    }
}

/// Check if a direct declarator contains a nested Function declarator.
fn dd_has_nested_function(dd: &ast::DirectDeclarator) -> bool {
    match dd {
        ast::DirectDeclarator::Function { .. } => true,
        ast::DirectDeclarator::Parenthesized(inner) => dd_has_nested_function(&inner.direct),
        ast::DirectDeclarator::Array { base, .. } => dd_has_nested_function(base),
        ast::DirectDeclarator::Identifier(..) => false,
    }
}

fn extract_param_name(param: &ast::ParameterDeclaration, name_table: &[String]) -> String {
    if let Some(ref declarator) = param.declarator {
        extract_declarator_name(declarator, name_table).unwrap_or_default()
    } else {
        String::new()
    }
}

fn resolve_param_type(
    param: &ast::ParameterDeclaration,
    _target: &Target,
    name_table: &[String],
) -> CType {
    let base = resolve_base_type_fast(&param.specifiers, name_table);
    if let Some(ref declarator) = param.declarator {
        apply_declarator_type(base, declarator, name_table)
    } else {
        base
    }
}

pub fn extract_declarator_name(
    declarator: &ast::Declarator,
    name_table: &[String],
) -> Option<String> {
    extract_name_from_dd(&declarator.direct, name_table)
}

fn extract_name_from_dd(dd: &ast::DirectDeclarator, name_table: &[String]) -> Option<String> {
    match dd {
        ast::DirectDeclarator::Identifier(sym, _) => Some(resolve_sym(name_table, sym).to_string()),
        ast::DirectDeclarator::Parenthesized(inner) => extract_declarator_name(inner, name_table),
        ast::DirectDeclarator::Array { base, .. } => extract_name_from_dd(base, name_table),
        ast::DirectDeclarator::Function { base, .. } => extract_name_from_dd(base, name_table),
    }
}

// ===========================================================================
// Constant evaluation utilities
// ===========================================================================

/// Evaluate a constant integer expression at compile time.
/// Returns None if the expression cannot be evaluated as a constant.
fn evaluate_const_int_expr(expr: &ast::Expression) -> Option<usize> {
    match expr {
        ast::Expression::IntegerLiteral { value, .. } => Some(*value as usize),
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Negate,
            operand,
            ..
        } => {
            let inner = evaluate_const_int_expr(operand)?;
            Some((-(inner as i64)) as usize)
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::BitwiseNot,
            operand,
            ..
        } => {
            let inner = evaluate_const_int_expr(operand)?;
            Some(!inner)
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Plus,
            operand,
            ..
        } => evaluate_const_int_expr(operand),
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::LogicalNot,
            operand,
            ..
        } => {
            let inner = evaluate_const_int_expr(operand)?;
            Some(if inner == 0 { 1 } else { 0 })
        }
        ast::Expression::Binary {
            op, left, right, ..
        } => {
            let l = evaluate_const_int_expr(left)?;
            let r = evaluate_const_int_expr(right)?;
            match op {
                ast::BinaryOp::Add => Some(l.wrapping_add(r)),
                ast::BinaryOp::Sub => Some(l.wrapping_sub(r)),
                ast::BinaryOp::Mul => Some(l.wrapping_mul(r)),
                ast::BinaryOp::Div if r != 0 => Some(l / r),
                ast::BinaryOp::Mod if r != 0 => Some(l % r),
                ast::BinaryOp::ShiftLeft => Some(l << (r & 63)),
                ast::BinaryOp::ShiftRight => Some(l >> (r & 63)),
                ast::BinaryOp::BitwiseAnd => Some(l & r),
                ast::BinaryOp::BitwiseOr => Some(l | r),
                ast::BinaryOp::BitwiseXor => Some(l ^ r),
                ast::BinaryOp::Equal => Some(if l == r { 1 } else { 0 }),
                ast::BinaryOp::NotEqual => Some(if l != r { 1 } else { 0 }),
                ast::BinaryOp::Less => Some(if (l as i64) < (r as i64) { 1 } else { 0 }),
                ast::BinaryOp::Greater => Some(if (l as i64) > (r as i64) { 1 } else { 0 }),
                ast::BinaryOp::LessEqual => Some(if (l as i64) <= (r as i64) { 1 } else { 0 }),
                ast::BinaryOp::GreaterEqual => Some(if (l as i64) >= (r as i64) { 1 } else { 0 }),
                ast::BinaryOp::LogicalAnd => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                ast::BinaryOp::LogicalOr => Some(if l != 0 || r != 0 { 1 } else { 0 }),
                _ => None,
            }
        }
        ast::Expression::Cast { operand, .. } => evaluate_const_int_expr(operand),
        ast::Expression::Parenthesized { inner, .. } => evaluate_const_int_expr(inner),
        ast::Expression::Conditional {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            let c = evaluate_const_int_expr(condition)?;
            if c != 0 {
                then_expr.as_ref().and_then(|e| evaluate_const_int_expr(e))
            } else {
                evaluate_const_int_expr(else_expr)
            }
        }
        // Handle sizeof(type) — critical for correct struct layout when
        // array dimensions use sizeof expressions (e.g. `int arr[sizeof(T)]`).
        ast::Expression::SizeofType { type_name, .. } => {
            // Access the target from the thread-local set during lowering.
            super::LOWERING_TARGET.with(|tl| {
                let target = tl.borrow();
                let target = target.as_ref()?;
                let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
                let cty = if let Some(ref abs) = type_name.abstract_declarator {
                    apply_abstract_declarator_to_type(cty, abs)
                } else {
                    cty
                };
                // Resolve forward-referenced struct types using the
                // thread-local struct_defs registry.
                let resolved = resolve_sizeof_struct_ref(cty, target);
                Some(crate::common::types::sizeof_ctype(&resolved, target))
            })
        }
        // Handle sizeof(expr) — infer the expression's type and compute size.
        ast::Expression::SizeofExpr { operand, .. } => {
            super::LOWERING_TARGET.with(|tl| {
                let target = tl.borrow();
                let target = target.as_ref()?;
                // Strip parentheses from the operand. In C, sizeof(arr) is
                // parsed as SizeofExpr { operand: Parenthesized { inner:
                // Identifier } } because the parser sees (arr) as a
                // parenthesized expression.
                let mut op = operand.as_ref();
                while let ast::Expression::Parenthesized { inner, .. } = op {
                    op = inner.as_ref();
                }
                // For sizeof applied to an expression, attempt to determine
                // the expression's type from context.  Common patterns:
                //   sizeof(*ptr) — dereference → pointee type
                //   sizeof(literal) — integer literal → int
                //   sizeof(identifier) — look up type from TYPEOF_CONTEXT
                match op {
                    ast::Expression::IntegerLiteral { .. } => {
                        Some(crate::common::types::sizeof_ctype(&CType::Int, target))
                    }
                    ast::Expression::StringLiteral { segments, .. } => {
                        // sizeof("string") = total length of segments + 1 (null terminator)
                        let total: usize = segments.iter().map(|s| s.value.len()).sum();
                        Some(total + 1)
                    }
                    ast::Expression::Identifier { name, .. } => {
                        // Look up the identifier's C type from the
                        // TYPEOF_CONTEXT (populated with globals and locals
                        // during lowering).  This correctly handles
                        // sizeof(array_var) returning the full array size
                        // instead of defaulting to pointer width.
                        let result = super::TYPEOF_CONTEXT.with(|ctx| {
                            let ctx_borrow = ctx.borrow();
                            if let Some(ref map) = *ctx_borrow {
                                // name is a Symbol — resolve to string via name_table
                                let name_str = super::NAME_TABLE.with(|nt| {
                                    let nt_borrow = nt.borrow();
                                    if let Some(ref table) = *nt_borrow {
                                        let idx = name.as_u32() as usize;
                                        if idx < table.len() {
                                            Some(table[idx].clone())
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    }
                                });
                                if let Some(name_s) = name_str {
                                    if let Some(cty) = map.get(&name_s) {
                                        return Some(crate::common::types::sizeof_ctype(
                                            cty, target,
                                        ));
                                    }
                                }
                            }
                            None
                        });
                        if result.is_some() {
                            return result;
                        }
                        // Fallback: pointer width
                        Some(target.pointer_width())
                    }
                    ast::Expression::UnaryOp {
                        op: ast::UnaryOp::Deref,
                        operand: inner,
                        ..
                    } => {
                        // sizeof(*ptr) — try to resolve the pointee type
                        if let ast::Expression::Identifier { name, .. } = inner.as_ref() {
                            let result = super::TYPEOF_CONTEXT.with(|ctx| {
                                let ctx_borrow = ctx.borrow();
                                if let Some(ref map) = *ctx_borrow {
                                    let name_str = super::NAME_TABLE.with(|nt| {
                                        let nt_borrow = nt.borrow();
                                        if let Some(ref table) = *nt_borrow {
                                            let idx = name.as_u32() as usize;
                                            if idx < table.len() {
                                                Some(table[idx].clone())
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    });
                                    if let Some(ns) = name_str {
                                        if let Some(cty) = map.get(&ns) {
                                            match cty {
                                                CType::Pointer(pointee, _) => {
                                                    return Some(
                                                        crate::common::types::sizeof_ctype(
                                                            pointee, target,
                                                        ),
                                                    );
                                                }
                                                CType::Array(elem, _) => {
                                                    return Some(
                                                        crate::common::types::sizeof_ctype(
                                                            elem, target,
                                                        ),
                                                    );
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                                None
                            });
                            if result.is_some() {
                                return result;
                            }
                        }
                        Some(target.pointer_width())
                    }
                    _ => {
                        // Default: pointer width for unresolvable expressions.
                        Some(target.pointer_width())
                    }
                }
            })
        }
        // Handle _Alignof(type) / __alignof__(type)
        ast::Expression::AlignofType { type_name, .. } => super::LOWERING_TARGET.with(|tl| {
            let target = tl.borrow();
            let target = target.as_ref()?;
            let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
            let cty = if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(cty, abs)
            } else {
                cty
            };
            let resolved = resolve_sizeof_struct_ref(cty, target);
            Some(crate::common::types::alignof_ctype(&resolved, target))
        }),
        // Handle __builtin_offsetof(type, member)
        ast::Expression::BuiltinCall {
            builtin: ast::BuiltinKind::Offsetof,
            args,
            ..
        } => {
            // args[0] is SizeofType wrapping the type name, args[1] is the member expr
            if args.len() < 2 {
                return None;
            }
            let type_name = match &args[0] {
                ast::Expression::SizeofType { type_name, .. } => type_name,
                _ => return None,
            };
            super::LOWERING_TARGET.with(|tl| {
                let target = tl.borrow();
                let target = target.as_ref()?;
                let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
                let cty = if let Some(ref abs) = type_name.abstract_declarator {
                    apply_abstract_declarator_to_type(cty, abs)
                } else {
                    cty
                };
                let resolved = resolve_sizeof_struct_ref(cty, target);
                // Extract the chain of member designators from args[1]
                let chain = extract_offsetof_member_chain(&args[1]);
                compute_struct_field_offset(&resolved, &chain, target)
            })
        }
        // Handle identifiers — enum constants used as array sizes or in
        // constant expressions (e.g., `int arr[NUM_ITEMS]` where NUM_ITEMS
        // is an enum constant).  Look up the identifier in the
        // ENUM_CONSTANTS_TL thread-local populated during early enum
        // collection (before struct definition collection).
        ast::Expression::Identifier { name, .. } => {
            super::ENUM_CONSTANTS_TL.with(|ec| {
                let ec_borrow = ec.borrow();
                if let Some(ref map) = *ec_borrow {
                    // Resolve the symbol to a string
                    let name_str = super::NAME_TABLE.with(|nt| {
                        let nt_borrow = nt.borrow();
                        if let Some(ref table) = *nt_borrow {
                            let idx = name.as_u32() as usize;
                            if idx < table.len() {
                                Some(table[idx].clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });
                    if let Some(ref name_s) = name_str {
                        if let Some(&val) = map.get(name_s) {
                            return Some(val as usize);
                        }
                    }
                }
                None
            })
        }
        _ => None,
    }
}

/// A step in an offsetof member-designator chain.
enum OffsetofMemberStep {
    /// `.field_name`
    Field(String),
    /// `[index]`
    Index(usize),
}

/// Extract the member designator chain from an offsetof member expression.
///
/// Walks the AST expression tree to build a flat list of field accesses and
/// array subscripts.  For example, `a.b[0].c` produces
/// `[Field("a"), Field("b"), Index(0), Field("c")]`.
fn extract_offsetof_member_chain(expr: &ast::Expression) -> Vec<OffsetofMemberStep> {
    let mut chain = Vec::new();
    extract_offsetof_chain_recursive(expr, &mut chain);
    chain
}

fn extract_offsetof_chain_recursive(expr: &ast::Expression, chain: &mut Vec<OffsetofMemberStep>) {
    let name_table_rc__ = super::INTERNER_SNAPSHOT.with(|snap| {
        snap.borrow()
            .as_ref()
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(Vec::new()))
    });
    let name_table: &[String] = &name_table_rc__;
    match expr {
        ast::Expression::Identifier { name, .. } => {
            let field_name = if (name.as_u32() as usize) < name_table.len() {
                name_table[name.as_u32() as usize].clone()
            } else {
                format!("sym_{}", name.as_u32())
            };
            chain.push(OffsetofMemberStep::Field(field_name));
        }
        ast::Expression::MemberAccess { object, member, .. }
        | ast::Expression::PointerMemberAccess { object, member, .. } => {
            extract_offsetof_chain_recursive(object, chain);
            let member_name = if (member.as_u32() as usize) < name_table.len() {
                name_table[member.as_u32() as usize].clone()
            } else {
                format!("sym_{}", member.as_u32())
            };
            chain.push(OffsetofMemberStep::Field(member_name));
        }
        ast::Expression::ArraySubscript { base, index, .. } => {
            extract_offsetof_chain_recursive(base, chain);
            if let Some(idx) = evaluate_const_int_expr(index) {
                chain.push(OffsetofMemberStep::Index(idx));
            }
        }
        ast::Expression::Parenthesized { inner, .. } => {
            extract_offsetof_chain_recursive(inner, chain);
        }
        _ => {}
    }
}

/// Compute the byte offset of a member within a struct/union type,
/// following a chain of field accesses and array subscripts.
///
/// This mirrors the layout computation in `compute_struct_size` from
/// `src/common/types.rs`, including bitfield packing, alignment, and
/// anonymous struct/union member lookup.
fn compute_struct_field_offset(
    ctype: &CType,
    chain: &[OffsetofMemberStep],
    target: &Target,
) -> Option<usize> {
    if chain.is_empty() {
        return Some(0);
    }

    // Unwrap typedef/qualified wrappers
    let unwrapped = unwrap_ctype_for_offsetof(ctype);

    // Resolve forward-declared structs via thread-local struct_defs
    let resolved = resolve_sizeof_struct_ref(unwrapped.clone(), target);
    let unwrapped = unwrap_ctype_for_offsetof(&resolved);

    match &chain[0] {
        OffsetofMemberStep::Index(idx) => {
            if let CType::Array(elem, _) = unwrapped {
                let elem_sz = crate::common::types::sizeof_ctype(elem, target);
                let rest = compute_struct_field_offset(elem, &chain[1..], target)?;
                Some(idx * elem_sz + rest)
            } else {
                None
            }
        }
        OffsetofMemberStep::Field(target_name) => {
            let (fields, is_union, packed) = match unwrapped {
                CType::Struct { fields, packed, .. } => (fields, false, *packed),
                CType::Union { fields, packed, .. } => (fields, true, *packed),
                _ => return None,
            };

            let mut bit_offset: usize = 0;
            if std::env::var("BCC_DEBUG_OFFSETOF").is_ok() {
                for (fi, f) in fields.iter().enumerate() {
                    eprintln!(
                        "[OFFSETOF] field[{}] name={:?} ty={:?} bw={:?}",
                        fi, f.name, f.ty, f.bit_width
                    );
                }
            }
            for field in fields {
                // Handle bitfield alignment
                if let Some(bits) = field.bit_width {
                    let bits = bits as usize;
                    let unit_bytes = crate::common::types::sizeof_ctype(&field.ty, target);
                    let unit_bits = unit_bytes * 8;
                    let fa = if packed {
                        1
                    } else {
                        crate::common::types::alignof_ctype(&field.ty, target)
                    };

                    if bits == 0 {
                        let align_bits = fa * 8;
                        if align_bits > 0 {
                            bit_offset = (bit_offset + align_bits - 1) / align_bits * align_bits;
                        }
                        continue;
                    }

                    let unit_align_bits = if packed { 8 } else { fa * 8 };
                    let unit_start = if unit_align_bits > 0 {
                        (bit_offset / unit_align_bits) * unit_align_bits
                    } else {
                        bit_offset
                    };
                    let bits_used = bit_offset - unit_start;
                    let start = if bits_used + bits <= unit_bits {
                        bit_offset
                    } else {
                        unit_start + unit_bits
                    };

                    let field_name = field.name.as_deref().unwrap_or("");
                    if field_name == target_name.as_str() {
                        return Some(start / 8);
                    }
                    if !is_union {
                        bit_offset = start + bits;
                    }
                    continue;
                }

                // Non-bitfield: round up to byte boundary
                if bit_offset % 8 != 0 {
                    bit_offset = (bit_offset + 7) / 8 * 8;
                }
                let byte_offset = bit_offset / 8;
                let fa = if packed {
                    1
                } else {
                    crate::common::types::alignof_ctype(&field.ty, target)
                };
                let aligned = if fa > 1 {
                    (byte_offset + fa - 1) & !(fa - 1)
                } else {
                    byte_offset
                };

                let field_name = field.name.as_deref().unwrap_or("");
                if field_name == target_name.as_str() {
                    if chain.len() == 1 {
                        return Some(aligned);
                    }
                    return compute_struct_field_offset(&field.ty, &chain[1..], target)
                        .map(|rest| aligned + rest);
                }

                // Check anonymous struct/union — recurse into it
                if field_name.is_empty() {
                    if let Some(inner_off) = compute_struct_field_offset(&field.ty, chain, target) {
                        return Some(aligned + inner_off);
                    }
                }

                if !is_union {
                    let field_sz = crate::common::types::sizeof_ctype(&field.ty, target);
                    bit_offset = aligned * 8 + field_sz * 8;
                }
            }
            None
        }
    }
}

/// Strip Typedef, Qualified, and Atomic wrappers to get the underlying type.
fn unwrap_ctype_for_offsetof(ctype: &CType) -> &CType {
    match ctype {
        CType::Typedef { underlying, .. } => unwrap_ctype_for_offsetof(underlying),
        CType::Qualified(inner, _) => unwrap_ctype_for_offsetof(inner),
        CType::Atomic(inner) => unwrap_ctype_for_offsetof(inner),
        other => other,
    }
}

/// Public wrapper for `evaluate_const_int_expr` so that other lowering
/// modules (e.g. `expr_lowering`) can evaluate non-literal array sizes.
pub fn evaluate_const_int_expr_pub(expr: &ast::Expression) -> Option<usize> {
    evaluate_const_int_expr(expr)
}

/// Evaluate `&array[index]` to a (symbol_name, byte_offset) pair.
///
/// Handles patterns like `&sqlite3UpperToLower[210]` which produce a
/// GlobalRefOffset relocation.  Recursively handles nested subscripts
/// for multi-dimensional arrays.
/// Given a nested `ArraySubscript` base and its inner subscript index,
/// determine the element size at the *outer* subscript level.
///
/// For example, given `a[i][j]` where `a : Array(Array(Array(Char,28),3),2)`:
///   - `a[i]` peels one layer → type becomes `Array(Array(Char,28),3)`
///   - elements of `a[i]` are `Array(Char,28)` with sizeof=28
///   - so `j` multiplies by 28
///
/// This walks up to the root identifier, counts how many subscript layers
/// have been peeled, then strips that many `Array(...)` layers from the
/// type stored in TYPEOF_CONTEXT and returns `sizeof(elem)` at that depth.
fn resolve_nested_array_elem_size(
    inner_base: &ast::Expression,
    _inner_index: &ast::Expression,
    target: &Target,
    name_table: &[String],
) -> i64 {
    // Walk up through nested ArraySubscript/Parenthesized/Cast to find the
    // root Identifier and count the total subscript depth.
    fn find_root_and_depth(
        expr: &ast::Expression,
        depth: usize,
        name_table: &[String],
    ) -> Option<(String, usize)> {
        match expr {
            ast::Expression::Identifier { name, .. } => {
                let name_str = resolve_sym(name_table, name).to_string();
                Some((name_str, depth))
            }
            ast::Expression::ArraySubscript { base, .. } => {
                find_root_and_depth(base, depth + 1, name_table)
            }
            ast::Expression::Parenthesized { inner, .. } => {
                find_root_and_depth(inner, depth, name_table)
            }
            ast::Expression::Cast { operand, .. } => {
                find_root_and_depth(operand, depth, name_table)
            }
            _ => None,
        }
    }

    // The inner_base already accounts for one subscript, so start depth=1.
    let root_info = find_root_and_depth(inner_base, 1, name_table);
    if let Some((root_name, depth)) = root_info {
        let result = super::TYPEOF_CONTEXT.with(|ctx| {
            let borrow = ctx.borrow();
            if let Some(ref map) = *borrow {
                if let Some(raw_ty) = map.get(&root_name) {
                    let mut ty = crate::common::types::resolve_and_strip(raw_ty);
                    // Peel `depth` Array layers to reach the type at
                    // the current subscript nesting level.
                    for _ in 0..depth {
                        if let CType::Array(elem, _) = ty {
                            ty = crate::common::types::resolve_and_strip(elem);
                        } else {
                            return None;
                        }
                    }
                    // `ty` is the array type at this subscript level.
                    // We need the *element* size (one more peel) since
                    // subscript `[j]` strides by sizeof(element), not
                    // sizeof(the-whole-row).
                    if let CType::Array(elem, _) = ty {
                        ty = crate::common::types::resolve_and_strip(elem);
                    }
                    return Some(crate::common::types::sizeof_ctype(ty, target) as i64);
                }
            }
            None
        });
        if let Some(sz) = result {
            return sz;
        }
    }
    // Fallback for byte arrays or when type info is unavailable.
    1
}

fn evaluate_address_of_subscript(
    base: &ast::Expression,
    index: &ast::Expression,
    target: &Target,
    name_table: &[String],
) -> Option<(String, i64)> {
    let idx_val = evaluate_const_int_expr(index)? as i64;

    match base {
        ast::Expression::Identifier { name, .. } => {
            let name_str = resolve_sym(name_table, name).to_string();
            // Determine element size.  For a bare identifier used as an
            // array base, look up its type from the TYPEOF_CONTEXT.
            // Must strip Qualified/Typedef/Atomic wrappers before matching
            // the Array variant.
            let elem_size = super::TYPEOF_CONTEXT.with(|ctx| {
                let borrow = ctx.borrow();
                if let Some(ref map) = *borrow {
                    if let Some(raw_ty) = map.get(&name_str) {
                        let stripped = crate::common::types::resolve_and_strip(raw_ty);
                        if let CType::Array(elem, _) = stripped {
                            return Some(crate::common::types::sizeof_ctype(elem, target) as i64);
                        }
                    }
                }
                // Default: try sizeof_ctype on the type itself.
                // For `const unsigned char arr[]`, element size is 1.
                None
            });
            let elem_sz = elem_size.unwrap_or(1);
            Some((name_str, idx_val * elem_sz))
        }
        ast::Expression::ArraySubscript {
            base: inner_base,
            index: inner_index,
            ..
        } => {
            // Nested subscript: &arr[i][j] — compute inner offset first.
            let (sym, inner_off) =
                evaluate_address_of_subscript(inner_base, inner_index, target, name_table)?;
            // Determine element size at THIS subscript level by resolving
            // the base identifier and peeling off array layers.
            let elem_sz =
                resolve_nested_array_elem_size(inner_base, inner_index, target, name_table);
            Some((sym, inner_off + idx_val * elem_sz))
        }
        ast::Expression::StringLiteral {
            segments, prefix, ..
        } => {
            // &("X"[0]) — create an anonymous string global and return
            // a reference to it at the computed byte offset.
            let char_width: usize = match prefix {
                ast::StringPrefix::None | ast::StringPrefix::U8 => 1,
                ast::StringPrefix::L | ast::StringPrefix::U32 => 4,
                ast::StringPrefix::U16 => 2,
            };
            let mut raw_bytes = Vec::new();
            for seg in segments {
                raw_bytes.extend_from_slice(&seg.value);
            }
            let bytes = super::decl_lowering::expand_string_bytes_for_width(&raw_bytes, char_width);
            let anon_name = create_anonymous_string_global(&bytes, target);
            Some((anon_name, idx_val * char_width as i64))
        }
        ast::Expression::Parenthesized { inner, .. } => {
            // &(("X")[0]) — unwrap parentheses and recurse.
            evaluate_address_of_subscript(inner, index, target, name_table)
        }
        ast::Expression::MemberAccess { .. } | ast::Expression::PointerMemberAccess { .. } => {
            // &struct_var.array_member[idx] — evaluate the member access
            // chain to get the base symbol + offset, then add the array
            // element offset.
            let (sym, base_off) = evaluate_address_of_member_chain(base, target, name_table)?;
            // Determine element size from the member expression's type.
            // The member is expected to be an array type — its element
            // size is needed to compute the byte offset for `[idx]`.
            let elem_sz = infer_member_array_element_size(base, target, name_table);
            Some((sym, base_off + idx_val * elem_sz))
        }
        ast::Expression::Cast { operand, .. } => {
            // &((type)expr)[idx] — unwrap cast and recurse.
            evaluate_address_of_subscript(operand, index, target, name_table)
        }
        _ => None,
    }
}

/// Evaluate address-of-member chains like `&((struct T *)0)->field` or
/// `&ptr->field`.  Returns `(symbol_name, byte_offset)` if the expression
/// is a compile-time address constant.
fn evaluate_address_of_member_chain(
    expr: &ast::Expression,
    target: &Target,
    name_table: &[String],
) -> Option<(String, i64)> {
    // Handle &identifier (base case)
    if let ast::Expression::Identifier { name, .. } = expr {
        let name_str = resolve_sym(name_table, name).to_string();
        return Some((name_str, 0));
    }
    // Handle parenthesized: &(expr)
    if let ast::Expression::Parenthesized { inner, .. } = expr {
        return evaluate_address_of_member_chain(inner, target, name_table);
    }
    // Handle cast: &((type)expr) — cast doesn't change address
    if let ast::Expression::Cast { operand, .. } = expr {
        return evaluate_address_of_member_chain(operand, target, name_table);
    }
    // Handle array subscript: &arr[N]
    if let ast::Expression::ArraySubscript { base, index, .. } = expr {
        return evaluate_address_of_subscript(base, index, target, name_table);
    }
    // Handle member access: expr.member
    if let ast::Expression::MemberAccess { object, member, .. } = expr {
        let member_name = resolve_sym(name_table, member).to_string();
        let (sym, base_off) = evaluate_address_of_member_chain(object, target, name_table)?;
        let field_off = evaluate_member_field_offset(&member_name, object, target, name_table);
        return Some((sym, base_off + field_off.unwrap_or(0)));
    }
    // Handle pointer member access: expr->member
    if let ast::Expression::PointerMemberAccess { object, member, .. } = expr {
        let member_name = resolve_sym(name_table, member).to_string();
        let (sym, base_off) = evaluate_address_of_member_chain(object, target, name_table)?;
        let field_off = evaluate_member_field_offset(&member_name, object, target, name_table);
        return Some((sym, base_off + field_off.unwrap_or(0)));
    }
    // Handle pointer arithmetic: ptr + N or ptr - N
    if let ast::Expression::Binary {
        op, left, right, ..
    } = expr
    {
        match op {
            ast::BinaryOp::Add => {
                // Try left as pointer, right as integer
                if let Some((sym, base_off)) =
                    evaluate_address_of_member_chain(left, target, name_table)
                {
                    if let Some(n) = evaluate_const_int_expr(right) {
                        let elem_sz = infer_element_size_for_ptr_arith(left, target, name_table);
                        return Some((sym, base_off + (n as i64) * elem_sz));
                    }
                }
                // Try right as pointer, left as integer
                if let Some((sym, base_off)) =
                    evaluate_address_of_member_chain(right, target, name_table)
                {
                    if let Some(n) = evaluate_const_int_expr(left) {
                        let elem_sz = infer_element_size_for_ptr_arith(right, target, name_table);
                        return Some((sym, base_off + (n as i64) * elem_sz));
                    }
                }
            }
            ast::BinaryOp::Sub => {
                if let Some((sym, base_off)) =
                    evaluate_address_of_member_chain(left, target, name_table)
                {
                    if let Some(n) = evaluate_const_int_expr(right) {
                        let elem_sz = infer_element_size_for_ptr_arith(left, target, name_table);
                        return Some((sym, base_off - (n as i64) * elem_sz));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Infer the element size when a member-access expression is used as an
/// array base in a static initializer — e.g., for `g.arr[1]` where `g` is
/// `struct Outer { struct Inner arr[3]; }`, this returns `sizeof(struct Inner)`.
///
/// Walks the member chain to find the struct field type, which should be an
/// array, and returns the element size of that array.
/// Infer the array element size when a member-access expression is used
/// as an array base in a static initializer — e.g., for `g.arr[1]` where
/// `g` is `struct Outer { struct Inner arr[3]; }`, returns `sizeof(struct Inner)`.
fn infer_member_array_element_size(
    member_expr: &ast::Expression,
    target: &Target,
    name_table: &[String],
) -> i64 {
    // For MemberAccess { object, member }, look up the struct layout from
    // TYPEOF_CONTEXT, find the member field, and if it is an array,
    // return the size of one element.
    if let ast::Expression::MemberAccess { object, member, .. }
    | ast::Expression::PointerMemberAccess { object, member, .. } = member_expr
    {
        let member_name = resolve_sym(name_table, member).to_string();
        // Get the struct type of the object (e.g. `g` → struct Outer).
        if let Some(struct_ty) = infer_struct_type_from_expr(object, name_table) {
            let resolved = resolve_sizeof_struct_ref(struct_ty, target);
            if let CType::Struct { ref fields, .. } = resolved {
                for f in fields {
                    if f.name.as_deref() == Some(&member_name) {
                        // The field type is the full type of the member.
                        // If it is an array, we want the element size.
                        match &f.ty {
                            CType::Array(elem, _) => {
                                let elem_resolved =
                                    resolve_sizeof_struct_ref(elem.as_ref().clone(), target);
                                let esz = crate::common::types::sizeof_ctype(&elem_resolved, target)
                                    as i64;
                                if esz > 0 {
                                    return esz;
                                }
                            }
                            CType::Pointer(pointee, _) => {
                                let p_resolved =
                                    resolve_sizeof_struct_ref(pointee.as_ref().clone(), target);
                                return crate::common::types::sizeof_ctype(&p_resolved, target)
                                    as i64;
                            }
                            _ => {
                                return crate::common::types::sizeof_ctype(&f.ty, target) as i64;
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: use infer_struct_type_from_expr on the full expression.
    // This returns the *result type* after the member access, which may
    // be an array type or an element type depending on how MemberAccess
    // is modeled.  If it is an array, unwrap it.
    if let Some(member_ty) = infer_struct_type_from_expr(member_expr, name_table) {
        let resolved = resolve_sizeof_struct_ref(member_ty, target);
        match &resolved {
            CType::Array(elem, _) => {
                let elem_resolved = resolve_sizeof_struct_ref(elem.as_ref().clone(), target);
                let esz = crate::common::types::sizeof_ctype(&elem_resolved, target) as i64;
                if esz > 0 {
                    return esz;
                }
            }
            _ => {
                let sz = crate::common::types::sizeof_ctype(&resolved, target) as i64;
                if sz > 0 {
                    return sz;
                }
            }
        }
    }

    // Ultimate fallback — pointer size.
    target.pointer_width() as i64
}

/// Infer the element size for pointer arithmetic on a global variable.
/// E.g., for `items + 1` where items is `struct foo[]`, returns sizeof(struct foo).
fn infer_element_size_for_ptr_arith(
    ptr_expr: &ast::Expression,
    target: &Target,
    name_table: &[String],
) -> i64 {
    let sym_name = match ptr_expr {
        ast::Expression::Identifier { name, .. } => Some(resolve_sym(name_table, name).to_string()),
        ast::Expression::Parenthesized { inner, .. } => match inner.as_ref() {
            ast::Expression::Identifier { name, .. } => {
                Some(resolve_sym(name_table, name).to_string())
            }
            _ => None,
        },
        _ => None,
    };
    if let Some(ref name) = sym_name {
        let elem_sz = super::TYPEOF_CONTEXT.with(|ctx| {
            let borrow = ctx.borrow();
            if let Some(ref map) = *borrow {
                if let Some(cty) = map.get(name) {
                    match cty {
                        CType::Array(elem, _) => {
                            let resolved = resolve_sizeof_struct_ref(elem.as_ref().clone(), target);
                            return Some(
                                crate::common::types::sizeof_ctype(&resolved, target) as i64
                            );
                        }
                        CType::Pointer(pointee, _) => {
                            let resolved =
                                resolve_sizeof_struct_ref(pointee.as_ref().clone(), target);
                            return Some(
                                crate::common::types::sizeof_ctype(&resolved, target) as i64
                            );
                        }
                        _ => {}
                    }
                }
            }
            None
        });
        if let Some(sz) = elem_sz {
            return sz;
        }
    }
    // Default to pointer size for unknown types
    target.pointer_width() as i64
}

/// Try to evaluate the byte offset of a struct/union member field.
/// Uses TYPEOF_CONTEXT to look up the struct type from the base expression,
/// then SIZEOF_STRUCT_DEFS to get the struct layout.
fn evaluate_member_field_offset(
    member_name: &str,
    base_expr: &ast::Expression,
    target: &Target,
    name_table: &[String],
) -> Option<i64> {
    // Try to find the struct type from the base expression's type in TYPEOF_CONTEXT
    let struct_type = infer_struct_type_from_expr(base_expr, name_table);
    if let Some(sty) = struct_type {
        // Resolve forward references
        let resolved = resolve_sizeof_struct_ref(sty, target);
        // Get fields from the resolved type
        if let CType::Struct {
            ref fields,
            packed,
            aligned,
            ..
        } = &resolved
        {
            // Use TypeBuilder::compute_struct_layout_with_fields to get
            // accurate byte offsets that correctly handle bitfield packing,
            // alignment, and storage unit sharing.  The naive manual walk
            // that previously lived here treated every bitfield as occupying
            // sizeof(underlying_type) bytes, which over-counted when
            // non-bitfield members shared a storage unit's tail bytes.
            let tb = crate::common::type_builder::TypeBuilder::new(*target);
            let layout = tb.compute_struct_layout_with_fields(fields, *packed, *aligned);
            // Walk fields and layout entries in parallel to match by name.
            for (field, fl) in fields.iter().zip(layout.fields.iter()) {
                if let Some(ref fname) = field.name {
                    if fname == member_name {
                        return Some(fl.offset as i64);
                    }
                }
            }
        } else if let CType::Union { .. } = &resolved {
            // Union members are all at offset 0
            return Some(0);
        }
    }
    // If we can't determine the offset, return 0 (best-effort for field at start)
    None
}

/// Infer the struct/union type that the base expression points to.
/// E.g. for `items + 1` where items: struct foo[], returns struct foo.
fn infer_struct_type_from_expr(expr: &ast::Expression, name_table: &[String]) -> Option<CType> {
    match expr {
        ast::Expression::Identifier { name, .. } => {
            let sym_name = resolve_sym(name_table, name).to_string();
            super::TYPEOF_CONTEXT.with(|ctx| {
                let borrow = ctx.borrow();
                if let Some(ref map) = *borrow {
                    if let Some(cty) = map.get(&sym_name) {
                        match cty {
                            CType::Array(elem, _) => return Some(elem.as_ref().clone()),
                            CType::Pointer(pointee, _) => return Some(pointee.as_ref().clone()),
                            CType::Struct { .. } | CType::Union { .. } => return Some(cty.clone()),
                            _ => {}
                        }
                    }
                }
                None
            })
        }
        ast::Expression::Parenthesized { inner, .. } => {
            infer_struct_type_from_expr(inner, name_table)
        }
        ast::Expression::Cast { operand, .. } => infer_struct_type_from_expr(operand, name_table),
        ast::Expression::Binary { left, right, .. } => {
            // For ptr + N, infer from the pointer side
            infer_struct_type_from_expr(left, name_table)
                .or_else(|| infer_struct_type_from_expr(right, name_table))
        }
        ast::Expression::PointerMemberAccess { object, member, .. }
        | ast::Expression::MemberAccess { object, member, .. } => {
            // After member access, the result type is the member's type.
            // Must handle both Struct and Union containers — a union's
            // member fields are looked up the same way as a struct's.
            let struct_ty = infer_struct_type_from_expr(object, name_table)?;
            let resolved =
                resolve_sizeof_struct_ref(struct_ty, &crate::common::target::Target::X86_64);
            let fields_opt = match &resolved {
                CType::Struct { ref fields, .. } | CType::Union { ref fields, .. } => Some(fields),
                _ => None,
            };
            if let Some(fields) = fields_opt {
                let member_name = resolve_sym(name_table, member).to_string();
                for f in fields {
                    if f.name.as_deref() == Some(&member_name) {
                        return Some(f.ty.clone());
                    }
                }
            }
            None
        }
        ast::Expression::ArraySubscript { base, .. } => {
            // For arr[i], the result type is the element type of arr.
            // Recursively get the array's type from its base, then peel
            // one Array layer to get the element type.
            let base_ty = infer_struct_type_from_expr(base, name_table)?;
            let resolved = crate::common::types::resolve_and_strip(&base_ty);
            match resolved {
                CType::Array(elem, _) => {
                    let elem_resolved = crate::common::types::resolve_and_strip(elem);
                    Some(elem_resolved.clone())
                }
                // If already peeled to a struct (base is already array-subscripted),
                // return it as-is.
                CType::Struct { .. } | CType::Union { .. } => Some(resolved.clone()),
                _ => Some(resolved.clone()),
            }
        }
        _ => None,
    }
}

/// Evaluate sizeof for a type expression (simplified heuristic).
fn evaluate_sizeof_expr(
    expr: &ast::Expression,
    target: &Target,
    type_builder: &TypeBuilder,
) -> u64 {
    match expr {
        ast::Expression::SizeofType { type_name, .. } => {
            let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
            let cty = if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(cty, abs)
            } else {
                cty
            };
            // Resolve forward-referenced struct/union using thread-local struct_defs.
            let resolved = resolve_sizeof_struct_ref(cty, target);
            type_builder.sizeof_type(&resolved) as u64
        }
        ast::Expression::SizeofExpr { operand, .. } => {
            // For sizeof(expr), infer the expression type from context.
            // Uses TYPEOF_CONTEXT thread-local to resolve variable types
            // so that sizeof(array_variable) returns the full array size
            // rather than decaying to pointer width.
            evaluate_sizeof_operand(operand, target, type_builder)
        }
        _ => target.pointer_width() as u64,
    }
}

/// Resolve the size of a sizeof(expr) operand in a global initializer context.
/// Uses TYPEOF_CONTEXT to look up variable types so that sizeof(array_var)
/// returns the full array byte-count instead of decaying to pointer width.
fn evaluate_sizeof_operand(
    operand: &ast::Expression,
    target: &Target,
    type_builder: &TypeBuilder,
) -> u64 {
    match operand {
        ast::Expression::IntegerLiteral { .. } => type_builder.sizeof_type(&CType::Int) as u64,
        ast::Expression::StringLiteral { segments, .. } => {
            let total: usize = segments.iter().map(|s| s.value.len()).sum();
            (total + 1) as u64
        }
        ast::Expression::Identifier { name, .. } => {
            // Look up the identifier in TYPEOF_CONTEXT — this preserves
            // array types (C11 §6.3.2.1p3: sizeof suppresses decay).
            let result = super::TYPEOF_CONTEXT.with(|ctx| {
                let ctx_borrow = ctx.borrow();
                if let Some(ref map) = *ctx_borrow {
                    let name_str = super::NAME_TABLE.with(|nt| {
                        let nt_borrow = nt.borrow();
                        if let Some(ref table) = *nt_borrow {
                            let idx = name.as_u32() as usize;
                            if idx < table.len() {
                                Some(table[idx].clone())
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });
                    if let Some(ns) = name_str {
                        if let Some(cty) = map.get(&ns) {
                            let resolved = resolve_sizeof_struct_ref(cty.clone(), target);
                            return Some(
                                crate::common::types::sizeof_ctype(&resolved, target) as u64
                            );
                        }
                    }
                }
                None
            });
            result.unwrap_or(target.pointer_width() as u64)
        }
        ast::Expression::ArraySubscript { base, index: _, .. } => {
            // sizeof(arr[i]) — resolve arr's element type.
            // The index value is irrelevant for sizeof; we need the
            // element type of the base expression.
            let base_type = resolve_expr_type_from_context(base, target);
            if let Some(cty) = base_type {
                let elem_ty = match cty {
                    CType::Array(elem, _) => *elem,
                    CType::Pointer(pointee, _) => *pointee,
                    _ => cty,
                };
                let resolved = resolve_sizeof_struct_ref(elem_ty, target);
                crate::common::types::sizeof_ctype(&resolved, target) as u64
            } else {
                // Try recursive: if index is integer literal and base is
                // a known array, get element size
                target.pointer_width() as u64
            }
        }
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            operand: inner,
            ..
        } => {
            // sizeof(*ptr) — resolve pointee type
            let base_type = resolve_expr_type_from_context(inner, target);
            if let Some(cty) = base_type {
                let pointee = match cty {
                    CType::Pointer(p, _) => *p,
                    CType::Array(elem, _) => *elem,
                    _ => cty,
                };
                let resolved = resolve_sizeof_struct_ref(pointee, target);
                crate::common::types::sizeof_ctype(&resolved, target) as u64
            } else {
                target.pointer_width() as u64
            }
        }
        ast::Expression::MemberAccess { object, member, .. } => {
            // sizeof(expr.field) — resolve field type
            let base_type = resolve_expr_type_from_context(object, target);
            if let Some(cty) = base_type {
                let resolved = resolve_sizeof_struct_ref(cty, target);
                if let Some(field_ty) = find_struct_field_type(&resolved, member) {
                    let resolved_field = resolve_sizeof_struct_ref(field_ty, target);
                    crate::common::types::sizeof_ctype(&resolved_field, target) as u64
                } else {
                    target.pointer_width() as u64
                }
            } else {
                target.pointer_width() as u64
            }
        }
        ast::Expression::PointerMemberAccess {
            object: pointer,
            member,
            ..
        } => {
            // sizeof(ptr->field) — resolve pointer→struct→field type
            let base_type = resolve_expr_type_from_context(pointer, target);
            if let Some(cty) = base_type {
                let pointee = match cty {
                    CType::Pointer(p, _) => *p,
                    _ => cty,
                };
                let resolved = resolve_sizeof_struct_ref(pointee, target);
                if let Some(field_ty) = find_struct_field_type(&resolved, member) {
                    let resolved_field = resolve_sizeof_struct_ref(field_ty, target);
                    crate::common::types::sizeof_ctype(&resolved_field, target) as u64
                } else {
                    target.pointer_width() as u64
                }
            } else {
                target.pointer_width() as u64
            }
        }
        ast::Expression::Parenthesized { inner, .. } => {
            evaluate_sizeof_operand(inner, target, type_builder)
        }
        ast::Expression::Cast { type_name, .. } => {
            // sizeof((type)expr) — the sizeof of the cast target type
            let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
            let cty = if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(cty, abs)
            } else {
                cty
            };
            let resolved = resolve_sizeof_struct_ref(cty, target);
            crate::common::types::sizeof_ctype(&resolved, target) as u64
        }
        _ => target.pointer_width() as u64,
    }
}

/// Resolve the C type of an expression by looking up identifiers in
/// TYPEOF_CONTEXT. Returns None if the type cannot be determined.
fn resolve_expr_type_from_context(expr: &ast::Expression, target: &Target) -> Option<CType> {
    match expr {
        ast::Expression::Identifier { name, .. } => super::TYPEOF_CONTEXT.with(|ctx| {
            let ctx_borrow = ctx.borrow();
            if let Some(ref map) = *ctx_borrow {
                let name_str = super::NAME_TABLE.with(|nt| {
                    let nt_borrow = nt.borrow();
                    if let Some(ref table) = *nt_borrow {
                        let idx = name.as_u32() as usize;
                        if idx < table.len() {
                            Some(table[idx].clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                });
                if let Some(ns) = name_str {
                    return map.get(&ns).cloned();
                }
            }
            None
        }),
        ast::Expression::Parenthesized { inner, .. } => {
            resolve_expr_type_from_context(inner, target)
        }
        _ => None,
    }
}

/// Find a named field's type within a struct/union CType.
fn find_struct_field_type(
    cty: &CType,
    member: &crate::common::string_interner::Symbol,
) -> Option<CType> {
    let fields = match cty {
        CType::Struct { fields, .. } | CType::Union { fields, .. } => fields,
        _ => return None,
    };
    super::NAME_TABLE.with(|nt| {
        let nt_borrow = nt.borrow();
        if let Some(ref table) = *nt_borrow {
            let member_idx = member.as_u32() as usize;
            let member_name = if member_idx < table.len() {
                &table[member_idx]
            } else {
                return None;
            };
            for f in fields {
                if let Some(ref fname) = f.name {
                    if fname == member_name {
                        return Some(f.ty.clone());
                    }
                }
            }
        }
        None
    })
}

/// Apply abstract declarator (pointer/array/function) layers to a base type
/// for constant expression evaluation (global initializers).
fn apply_abstract_declarator_to_type(base: CType, abs: &ast::AbstractDeclarator) -> CType {
    let mut result = if let Some(ref pointer) = abs.pointer {
        let quals = crate::common::types::TypeQualifiers::default();
        let mut current = CType::Pointer(Box::new(base), quals);
        fn apply_ptr(current: CType, inner: &Option<Box<ast::Pointer>>) -> CType {
            if let Some(ref next) = inner {
                let quals = crate::common::types::TypeQualifiers::default();
                let wrapped = CType::Pointer(Box::new(current), quals);
                apply_ptr(wrapped, &next.inner)
            } else {
                current
            }
        }
        current = apply_ptr(current, &pointer.inner);
        current
    } else {
        base
    };

    if let Some(ref direct) = abs.direct {
        result = apply_direct_abstract_decl_to_type(result, direct);
    }
    result
}

fn apply_direct_abstract_decl_to_type(
    base: CType,
    direct: &ast::DirectAbstractDeclarator,
) -> CType {
    match direct {
        ast::DirectAbstractDeclarator::Parenthesized(inner) => {
            apply_abstract_declarator_to_type(base, inner)
        }
        ast::DirectAbstractDeclarator::Array { size, .. } => {
            let len = size.as_ref().and_then(|e| evaluate_const_int_expr(e));
            CType::Array(Box::new(base), len)
        }
        ast::DirectAbstractDeclarator::Function { .. } => CType::Pointer(
            Box::new(base),
            crate::common::types::TypeQualifiers::default(),
        ),
    }
}

/// Resolve a struct/union forward reference for sizeof evaluation.
/// Uses SIZEOF_STRUCT_DEFS thread-local when available, otherwise
/// falls back to the struct_defs from the lowering context.
/// Public wrapper for resolve_sizeof_struct_ref for use in stmt_lowering.
pub fn resolve_sizeof_struct_ref_pub(cty: CType, target: &Target) -> CType {
    resolve_sizeof_struct_ref(cty, target)
}

fn resolve_sizeof_struct_ref(mut cty: CType, _target: &Target) -> CType {
    // Use thread-local struct defs registry if available
    super::SIZEOF_STRUCT_DEFS.with(|cell| {
        let guard = cell.borrow();
        if let Some(ref map) = *guard {
            resolve_sizeof_forward_ref(&mut cty, map);
        }
    });
    cty
}

/// Recursively resolve forward-referenced struct/union types in a CType tree.
fn resolve_sizeof_forward_ref(ctype: &mut CType, struct_defs: &FxHashMap<String, CType>) {
    match ctype {
        CType::Struct {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = struct_defs.get(&tag_owned) {
                *ctype = full_def.clone();
            }
        }
        CType::Union {
            name: Some(ref tag),
            ref fields,
            ..
        } if fields.is_empty() => {
            let tag_owned = tag.clone();
            if let Some(full_def) = struct_defs.get(&tag_owned) {
                *ctype = full_def.clone();
            }
        }
        CType::Pointer(inner, _) => resolve_sizeof_forward_ref(inner, struct_defs),
        CType::Array(inner, _) => resolve_sizeof_forward_ref(inner, struct_defs),
        CType::Typedef { underlying, .. } => resolve_sizeof_forward_ref(underlying, struct_defs),
        CType::Qualified(inner, _) => resolve_sizeof_forward_ref(inner, struct_defs),
        CType::Atomic(inner) => resolve_sizeof_forward_ref(inner, struct_defs),
        // Also resolve struct fields that contain forward-referenced types.
        CType::Struct { ref mut fields, .. } if !fields.is_empty() => {
            for field in fields.iter_mut() {
                resolve_sizeof_forward_ref(&mut field.ty, struct_defs);
            }
        }
        CType::Union { ref mut fields, .. } if !fields.is_empty() => {
            for field in fields.iter_mut() {
                resolve_sizeof_forward_ref(&mut field.ty, struct_defs);
            }
        }
        _ => {}
    }
}

/// Evaluate alignof for a type expression.
fn evaluate_alignof_expr(
    expr: &ast::Expression,
    target: &Target,
    type_builder: &TypeBuilder,
) -> u64 {
    match expr {
        ast::Expression::AlignofType { type_name, .. } => {
            let cty = resolve_base_type_from_sqlist(&type_name.specifier_qualifiers);
            let cty = if let Some(ref abs) = type_name.abstract_declarator {
                apply_abstract_declarator_to_type(cty, abs)
            } else {
                cty
            };
            let resolved = resolve_sizeof_struct_ref(cty, target);
            type_builder.alignof_type(&resolved) as u64
        }
        _ => target.pointer_width() as u64,
    }
}

// ===========================================================================
// Designator resolution helpers
// ===========================================================================

/// Resolve array index from designator, or fall back to default index.
/// Recursively zero-initialize a field at the given byte offset.  For scalar
/// fields, emits a single store of 0.  For struct fields, recurses into each
/// sub-field so that the entire aggregate is zeroed.
/// Count the number of leaf scalar fields in an aggregate type for brace
/// elision detection.  Returns 1 for non-aggregate types.
#[allow(clippy::only_used_in_recursion)]
fn count_aggregate_scalar_fields(ty: &CType, tb: &TypeBuilder) -> usize {
    let resolved = crate::common::types::resolve_typedef(ty);
    match resolved {
        CType::Struct { fields, .. } => {
            let mut total = 0;
            for f in fields {
                total += count_aggregate_scalar_fields(&f.ty, tb);
            }
            if total == 0 {
                1
            } else {
                total
            }
        }
        CType::Array(elem, size_opt) => {
            let len = size_opt.unwrap_or(1);
            len * count_aggregate_scalar_fields(elem, tb)
        }
        _ => 1,
    }
}

/// Lower flat scalar initializers into the fields of an aggregate (struct),
/// consuming `init_list[cursor..]` and returning the updated cursor.
fn lower_brace_elision_into_aggregate(
    base_ptr: Value,
    init_list: &[ast::DesignatedInitializer],
    cursor: usize,
    target_type: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) -> usize {
    let resolved = crate::common::types::resolve_typedef(target_type);
    match resolved {
        CType::Struct {
            fields,
            packed,
            aligned,
            ..
        } => {
            // Use the proper struct layout computation which handles
            // bitfield packing, zero-width bitfields, and alignment.
            let layout = ctx
                .type_builder
                .compute_struct_layout_with_fields(fields, *packed, *aligned);
            let mut cur = cursor;
            for (fi, field) in fields.iter().enumerate() {
                if cur >= init_list.len() {
                    break;
                }
                let fl = &layout.fields[fi];
                let byte_offset = fl.offset as i64;

                // Skip zero-width bitfields (they produce no storage).
                if let Some((_, 0)) = fl.bitfield_info {
                    continue;
                }

                let field_resolved = crate::common::types::resolve_typedef(&field.ty);

                if let Some((bit_offset, bit_width)) = fl.bitfield_info {
                    // ----- Bitfield field -----
                    // Must use read-modify-write pattern: load the storage
                    // unit, mask out the old bits, shift the new value into
                    // position, OR, store back.
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (storage_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);

                    // Lower the initializer expression to get the value.
                    let val = match &init_list[cur].initializer {
                        ast::Initializer::Expression(expr) => {
                            expr_lowering::lower_expression(ctx, expr)
                        }
                        ast::Initializer::List {
                            designators_and_initializers,
                            ..
                        } => {
                            // Scalar bitfield with braced initializer, e.g. {1}
                            if let Some(first) = designators_and_initializers.first() {
                                if let ast::Initializer::Expression(expr) = &first.initializer {
                                    expr_lowering::lower_expression(ctx, expr)
                                } else {
                                    expr_lowering::emit_int_const_for_zero(ctx, IrType::I32)
                                }
                            } else {
                                expr_lowering::emit_int_const_for_zero(ctx, IrType::I32)
                            }
                        }
                    };

                    expr_lowering::lower_bitfield_store(
                        ctx,
                        storage_ptr,
                        val,
                        &field.ty,
                        bit_offset,
                        bit_width,
                        span,
                    );
                    cur += 1;
                } else if matches!(field_resolved, CType::Struct { .. } | CType::Array(..)) {
                    // Recurse into nested aggregate
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (field_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    cur = lower_brace_elision_into_aggregate(
                        field_ptr,
                        init_list,
                        cur,
                        field_resolved,
                        ctx,
                        span,
                    );
                } else {
                    // Scalar field — consume one initializer
                    let idx_val = make_index_value(ctx, byte_offset);
                    let (field_ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    lower_single_init_element(
                        field_ptr,
                        &init_list[cur].initializer,
                        &field.ty,
                        ctx,
                    );
                    cur += 1;
                }
            }
            cur
        }
        CType::Array(elem, size_opt) => {
            let len = size_opt.unwrap_or(1);
            let elem_size = ctx.type_builder.sizeof_type(elem);
            let mut cur = cursor;
            for i in 0..len {
                if cur >= init_list.len() {
                    break;
                }
                let byte_offset = (i * elem_size) as i64;
                let elem_resolved = crate::common::types::resolve_typedef(elem);
                let idx_val = make_index_value(ctx, byte_offset);
                let (elem_ptr, gep_inst) =
                    ctx.builder
                        .build_gep(base_ptr, vec![idx_val], IrType::Ptr, span);
                emit_inst_to_ctx(ctx, gep_inst);
                if matches!(elem_resolved, CType::Struct { .. } | CType::Array(..)) {
                    cur = lower_brace_elision_into_aggregate(
                        elem_ptr,
                        init_list,
                        cur,
                        elem_resolved,
                        ctx,
                        span,
                    );
                } else {
                    lower_single_init_element(elem_ptr, &init_list[cur].initializer, elem, ctx);
                    cur += 1;
                }
            }
            cur
        }
        _ => {
            // Scalar — consume one initializer
            if cursor < init_list.len() {
                lower_single_init_element(
                    base_ptr,
                    &init_list[cursor].initializer,
                    target_type,
                    ctx,
                );
                cursor + 1
            } else {
                cursor
            }
        }
    }
}

fn zero_init_field(
    base_alloca: Value,
    byte_offset: i64,
    field_ty: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) {
    let resolved = crate::common::types::resolve_typedef(field_ty);
    match resolved {
        CType::Struct {
            ref fields,
            packed,
            aligned,
            ..
        } => {
            // Use the full layout computation that correctly handles
            // bitfield packing (multiple bitfields share one storage
            // unit at the same byte offset).  The old
            // `compute_struct_field_offsets` assigned sequential offsets
            // per field ignoring bitfields, causing buffer overflows
            // when zero-initialising bitfield structs.
            let layout = ctx
                .type_builder
                .compute_struct_layout_with_fields(fields, *packed, *aligned);
            // Track which storage-unit byte offsets we have already
            // zeroed so we don't emit redundant stores (many bitfield
            // fields may share offset 0 in the same allocation unit).
            let mut zeroed_offsets = std::collections::HashSet::<i64>::new();
            for (i, sub_field) in fields.iter().enumerate() {
                let fl = &layout.fields[i];
                let sub_off = byte_offset + fl.offset as i64;
                // For bitfield fields, zero the storage unit exactly
                // once instead of once per bitfield member.
                if sub_field.bit_width.is_some() {
                    if fl.size == 0 {
                        // Zero-width padding bitfield — nothing to store.
                        continue;
                    }
                    if zeroed_offsets.insert(sub_off) {
                        // First bitfield at this offset — zero the
                        // storage unit.
                        let idx_val = make_index_value(ctx, sub_off);
                        let (ptr, gep_inst) =
                            ctx.builder
                                .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                        emit_inst_to_ctx(ctx, gep_inst);
                        let ir_ty = expr_lowering::ctype_to_ir(&sub_field.ty, ctx.target);
                        let zero_val = expr_lowering::emit_int_const_for_zero(ctx, ir_ty);
                        let store_inst = ctx.builder.build_store(zero_val, ptr, span);
                        emit_inst_to_ctx(ctx, store_inst);
                    }
                } else {
                    zero_init_field(base_alloca, sub_off, &sub_field.ty, ctx, span);
                }
            }
        }
        CType::Union { ref fields, .. } => {
            // Zero ALL bytes of the union (not just the first member)
            // so that reading through any member after zero-init sees
            // zeroes.  The first member alone may be smaller than the
            // entire union, leaving trailing bytes uninitialized.
            let union_size = crate::common::types::sizeof_ctype(resolved, ctx.target);
            if union_size > 0 {
                for byte_idx in 0..union_size {
                    let off = byte_offset + byte_idx as i64;
                    let idx_val = make_index_value(ctx, off);
                    let (ptr, gep_inst) =
                        ctx.builder
                            .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
                    emit_inst_to_ctx(ctx, gep_inst);
                    let zero_val = expr_lowering::emit_int_const_for_zero(ctx, IrType::I8);
                    let store_inst = ctx.builder.build_store(zero_val, ptr, span);
                    emit_inst_to_ctx(ctx, store_inst);
                }
            } else if let Some(first) = fields.first() {
                // Fallback: if sizeof fails, at least zero the first
                // member (matches previous behavior).
                zero_init_field(base_alloca, byte_offset, &first.ty, ctx, span);
            }
        }
        CType::Array(ref elem, Some(len)) => {
            let elem_size = ctx.type_builder.sizeof_type(elem);
            for i in 0..*len {
                let arr_off = byte_offset + (i * elem_size) as i64;
                zero_init_field(base_alloca, arr_off, elem, ctx, span);
            }
        }
        CType::Array(_, None) => {
            // Flexible array member — zero-length by definition (C99 §6.7.2.1/18).
            // Do NOT emit any stores; the FAM contributes zero bytes to
            // the struct's sizeof and any writes would overflow into
            // adjacent stack variables.
        }
        _ => {
            // Scalar: emit GEP + store 0 with the correct width for the
            // field type so that we don't clobber adjacent struct fields.
            let idx_val = make_index_value(ctx, byte_offset);
            let (ptr, gep_inst) =
                ctx.builder
                    .build_gep(base_alloca, vec![idx_val], IrType::Ptr, span);
            emit_inst_to_ctx(ctx, gep_inst);
            // Create a zero constant matching the field's IR type width.
            let ir_ty = expr_lowering::ctype_to_ir(resolved, ctx.target);
            let zero_val = expr_lowering::emit_int_const_for_zero(ctx, ir_ty);
            let store_inst = ctx.builder.build_store(zero_val, ptr, span);
            emit_inst_to_ctx(ctx, store_inst);
        }
    }
}

/// Lower a single designated struct initializer, handling nested designators
/// like `.origin.x = 1` by recursively navigating field types.
fn lower_designated_struct_init(
    base_alloca: Value,
    designators: &[ast::Designator],
    initializer: &ast::Initializer,
    fields: &[crate::common::types::StructField],
    field_offsets: &[usize],
    field_layouts: &[crate::common::type_builder::FieldLayout],
    accumulated_offset: i64,
    default_idx: usize,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
    span: Span,
) {
    // Resolve the first designator to a field index.
    let field_idx = resolve_field_designator(designators, fields, default_idx, ctx.name_table);
    if field_idx >= fields.len() {
        return;
    }
    let field = &fields[field_idx];
    let byte_offset =
        accumulated_offset + field_offsets.get(field_idx).copied().unwrap_or(0) as i64;

    // If there are remaining designators (nested designation), recurse into
    // the sub-struct.
    if designators.len() > 1 {
        let remaining_designators = &designators[1..];
        let resolved_field_ty = crate::common::types::resolve_typedef(&field.ty);
        if let CType::Struct {
            fields: ref sub_fields,
            packed: sub_packed,
            aligned: sub_aligned,
            ..
        } = resolved_field_ty
        {
            let sub_layout = ctx.type_builder.compute_struct_layout_with_fields(
                sub_fields,
                *sub_packed,
                *sub_aligned,
            );
            let inner_offsets: Vec<usize> = sub_layout.fields.iter().map(|fl| fl.offset).collect();
            lower_designated_struct_init(
                base_alloca,
                remaining_designators,
                initializer,
                sub_fields,
                &inner_offsets,
                &sub_layout.fields,
                byte_offset,
                0,
                ctx,
                span,
            );
            return;
        }
        // If the field type is not a struct but there are more designators,
        // fall through to treat as a simple store at the current offset.
    }

    // Check if this field is a bitfield — if so, use read-modify-write.
    let bf_info = field_layouts.get(field_idx).and_then(|fl| fl.bitfield_info);
    if let Some((bit_offset, bit_width)) = bf_info {
        if bit_width > 0 {
            // Bitfield field — GEP to the storage unit offset and use
            // lower_bitfield_store for correct read-modify-write masking.
            let field_idx_val = make_index_value(ctx, byte_offset);
            let (storage_ptr, gep_inst) =
                ctx.builder
                    .build_gep(base_alloca, vec![field_idx_val], IrType::Ptr, span);
            emit_inst_to_ctx(ctx, gep_inst);

            // Lower the initializer to a value.
            let val = match initializer {
                ast::Initializer::Expression(expr) => expr_lowering::lower_expression(ctx, expr),
                ast::Initializer::List {
                    designators_and_initializers,
                    ..
                } => {
                    if let Some(first) = designators_and_initializers.first() {
                        if let ast::Initializer::Expression(expr) = &first.initializer {
                            expr_lowering::lower_expression(ctx, expr)
                        } else {
                            expr_lowering::emit_int_const_for_zero(ctx, IrType::I32)
                        }
                    } else {
                        expr_lowering::emit_int_const_for_zero(ctx, IrType::I32)
                    }
                }
            };

            expr_lowering::lower_bitfield_store(
                ctx,
                storage_ptr,
                val,
                &field.ty,
                bit_offset,
                bit_width,
                span,
            );
            return;
        }
    }

    // Leaf case: GEP to the byte offset and lower the initializer.
    let field_idx_val = make_index_value(ctx, byte_offset);
    let (field_ptr, gep_inst) =
        ctx.builder
            .build_gep(base_alloca, vec![field_idx_val], IrType::Ptr, span);
    emit_inst_to_ctx(ctx, gep_inst);
    lower_single_init_element(field_ptr, initializer, &field.ty, ctx);
}

fn resolve_designator_index(designators: &[ast::Designator], default_idx: usize) -> usize {
    if let Some(di) = designators.first() {
        match di {
            ast::Designator::Index(expr, _) => evaluate_const_int_expr(expr).unwrap_or(default_idx),
            _ => default_idx,
        }
    } else {
        default_idx
    }
}

/// Resolve struct/union field index from designator, or fall back to default.
/// Count the number of flat scalar elements an aggregate type needs for
/// brace-elided initialization (e.g. `struct S { int a[3]; }` → 3).
fn flat_scalar_count(ty: &CType, type_builder: &TypeBuilder) -> usize {
    let resolved = crate::common::types::resolve_typedef(ty);
    match resolved {
        CType::Array(elem, Some(n)) => {
            let inner = flat_scalar_count(elem.as_ref(), type_builder);
            inner * n
        }
        CType::Struct { ref fields, .. } => fields
            .iter()
            .filter(|f| f.name.is_some()) // skip anonymous fields
            .map(|f| flat_scalar_count(&f.ty, type_builder))
            .sum(),
        _ => 1,
    }
}

fn resolve_field_designator(
    designators: &[ast::Designator],
    fields: &[crate::common::types::StructField],
    default_idx: usize,
    name_table: &[String],
) -> usize {
    if let Some(di) = designators.first() {
        match di {
            ast::Designator::Field(sym, _) => {
                let field_name = resolve_sym(name_table, sym);
                fields
                    .iter()
                    .position(|f| f.name.as_deref() == Some(field_name))
                    .unwrap_or(default_idx)
            }
            _ => default_idx,
        }
    } else {
        default_idx
    }
}

// ===========================================================================
// Instruction emission helpers
// ===========================================================================

/// Push an instruction into the entry block of the given function.
fn push_inst_to_entry(function: &mut IrFunction, inst: Instruction) {
    function.entry_block_mut().push_instruction(inst);
}

/// Emit an instruction into the current insertion point of the builder context.
/// Check whether an rvalue expression, when lowered, produces a *pointer*
/// to aggregate data rather than the aggregate data itself.
/// Mirrors `rhs_expr_returns_aggregate_pointer` in expr_lowering.
fn expr_returns_aggregate_pointer(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::Assignment { .. } => true,
        ast::Expression::Parenthesized { inner, .. } => expr_returns_aggregate_pointer(inner),
        ast::Expression::Comma { exprs, .. } => {
            if let Some(last) = exprs.last() {
                expr_returns_aggregate_pointer(last)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn emit_inst_to_ctx(ctx: &mut expr_lowering::ExprLoweringContext<'_>, inst: Instruction) {
    if let Some(block_id) = ctx.builder.get_insert_block() {
        let block_idx = block_id.0 as usize;
        if let Some(block) = ctx.function.get_block_mut(block_idx) {
            block.push_instruction(inst);
        }
    }
}

// =========================================================================
// String literal → char array initialization helpers
// =========================================================================

/// Check if a resolved type is a character array and return (element_type, array_size).
/// Matches `char`, `signed char`, `unsigned char` arrays.
/// Return the byte size of a single element for wide character array types.
/// Used when converting `Constant::String` byte lengths (which are multi-byte
/// encoded for wide strings) into element counts for array sizing.
///
/// - `Int` / `UInt` → 4 (wchar_t / char32_t on Linux)
/// - `Short` / `UShort` → 2 (char16_t)
/// - Everything else → 1 (char / signed char / unsigned char)
/// Return the byte size of a single element for wide character array types.
/// Used when converting `Constant::String` byte lengths (which are multi-byte
/// encoded for wide strings) into element counts for array sizing.
///
/// - `Int` / `UInt` → 4 (wchar_t / char32_t on Linux)
/// - `Short` / `UShort` → 2 (char16_t)
/// - Everything else → 1 (char / signed char / unsigned char)
pub fn wide_char_elem_size(elem: &CType) -> usize {
    let stripped = crate::common::types::resolve_typedef(elem);
    match stripped {
        CType::Int | CType::UInt => 4,
        CType::Short | CType::UShort => 2,
        _ => 1,
    }
}

/// Decode raw bytes (potentially UTF-8 encoded) into Unicode code points.
/// For each valid multi-byte UTF-8 sequence, yields one u32 code point.
/// Invalid or PUA-range bytes (0x80–0xFF that aren't valid UTF-8 lead bytes)
/// are treated as individual Latin-1 code points for backward compatibility.
pub fn decode_bytes_to_codepoints(bytes: &[u8]) -> Vec<u32> {
    let mut codepoints = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            // ASCII
            codepoints.push(b as u32);
            i += 1;
        } else if b & 0xE0 == 0xC0 && i + 1 < bytes.len() && bytes[i + 1] & 0xC0 == 0x80 {
            // 2-byte UTF-8
            let cp = ((b as u32 & 0x1F) << 6) | (bytes[i + 1] as u32 & 0x3F);
            codepoints.push(cp);
            i += 2;
        } else if b & 0xF0 == 0xE0
            && i + 2 < bytes.len()
            && bytes[i + 1] & 0xC0 == 0x80
            && bytes[i + 2] & 0xC0 == 0x80
        {
            // 3-byte UTF-8
            let cp = ((b as u32 & 0x0F) << 12)
                | ((bytes[i + 1] as u32 & 0x3F) << 6)
                | (bytes[i + 2] as u32 & 0x3F);
            codepoints.push(cp);
            i += 3;
        } else if b & 0xF8 == 0xF0
            && i + 3 < bytes.len()
            && bytes[i + 1] & 0xC0 == 0x80
            && bytes[i + 2] & 0xC0 == 0x80
            && bytes[i + 3] & 0xC0 == 0x80
        {
            // 4-byte UTF-8
            let cp = ((b as u32 & 0x07) << 18)
                | ((bytes[i + 1] as u32 & 0x3F) << 12)
                | ((bytes[i + 2] as u32 & 0x3F) << 6)
                | (bytes[i + 3] as u32 & 0x3F);
            codepoints.push(cp);
            i += 4;
        } else {
            // Invalid UTF-8 or bare high byte — treat as Latin-1 code point
            codepoints.push(b as u32);
            i += 1;
        }
    }
    codepoints
}

/// Expand raw segment bytes (UTF-8 encoded) into wide character bytes.
/// Each byte in narrow strings is treated as-is. For wide strings (char_width > 1),
/// UTF-8 sequences are decoded to Unicode code points, then each code point is
/// encoded at the specified width in little-endian byte order.
pub fn expand_string_bytes_for_width(raw_bytes: &[u8], char_width: usize) -> Vec<u8> {
    if char_width == 1 {
        let mut out = raw_bytes.to_vec();
        out.push(0); // null terminator
        out
    } else {
        let codepoints = decode_bytes_to_codepoints(raw_bytes);
        let mut bytes = Vec::with_capacity((codepoints.len() + 1) * char_width);
        for cp in &codepoints {
            for i in 0..char_width {
                bytes.push(((*cp >> (i * 8)) & 0xFF) as u8);
            }
        }
        // Null terminator
        for _ in 0..char_width {
            bytes.push(0);
        }
        bytes
    }
}

fn is_char_array_type(ty: &CType) -> Option<(CType, usize)> {
    let resolved = crate::common::types::resolve_typedef(ty);
    match resolved {
        CType::Array(elem, Some(size)) => {
            let stripped = crate::common::types::resolve_typedef(elem.as_ref());
            match stripped {
                CType::Char
                | CType::SChar
                | CType::UChar
                | CType::Int
                | CType::UInt
                | CType::Short
                | CType::UShort => Some((stripped.clone(), *size)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract raw bytes from a StringLiteral expression (including null terminator).
/// For wide string literals (L, U, u), each source byte is expanded to
/// the appropriate width (4 bytes for wchar_t/char32_t, 2 for char16_t).
fn extract_string_literal_bytes(expr: &ast::Expression) -> Option<Vec<u8>> {
    match expr {
        ast::Expression::StringLiteral {
            segments, prefix, ..
        } => {
            let char_width: usize = match prefix {
                crate::frontend::parser::ast::StringPrefix::None
                | crate::frontend::parser::ast::StringPrefix::U8 => 1,
                crate::frontend::parser::ast::StringPrefix::L
                | crate::frontend::parser::ast::StringPrefix::U32 => 4,
                crate::frontend::parser::ast::StringPrefix::U16 => 2,
            };
            let mut raw_bytes = Vec::new();
            for seg in segments {
                raw_bytes.extend_from_slice(&seg.value);
            }
            Some(expand_string_bytes_for_width(&raw_bytes, char_width))
        }
        _ => None,
    }
}

/// Emit stores to copy string literal bytes into a local char array variable.
/// Handles zero-filling of remaining array elements beyond the string length
/// and truncation when the string is longer than the array.
fn lower_char_array_from_string(
    var_alloca: Value,
    str_bytes: &[u8],
    arr_size: usize,
    _elem_ty: &CType,
    ctx: &mut expr_lowering::ExprLoweringContext<'_>,
) {
    let span = Span::dummy();

    // Number of bytes to copy from the string (may be truncated to fit)
    let copy_len = std::cmp::min(str_bytes.len(), arr_size);

    // Build a full byte buffer: string bytes + zero-fill
    let mut buf = vec![0u8; arr_size];
    buf[..copy_len].copy_from_slice(&str_bytes[..copy_len]);

    // Emit stores coalesced into the widest possible chunks (8/4/2/1 bytes)
    // for efficiency.  We process the array from left to right.
    let mut offset: usize = 0;
    while offset < arr_size {
        let remaining = arr_size - offset;

        // Determine the largest power-of-two chunk we can store at this
        // alignment.  We only coalesce if offset is naturally aligned for
        // the chosen width.
        let (chunk_sz, ir_ty): (usize, IrType) = if remaining >= 8 && offset % 8 == 0 {
            (8, IrType::I64)
        } else if remaining >= 4 && offset % 4 == 0 {
            (4, IrType::I32)
        } else if remaining >= 2 && offset % 2 == 0 {
            (2, IrType::I16)
        } else {
            (1, IrType::I8)
        };

        // Read the chunk value from our buffer (little-endian).
        let chunk_val: i128 = {
            let mut v: u64 = 0;
            for i in 0..chunk_sz {
                v |= (buf[offset + i] as u64) << (i * 8);
            }
            v as i128
        };

        // Create the constant value **with the correct chunk width**.
        // Previous bug: `emit_int_const_for_index` always produced an I64
        // constant, so the backend would emit an 8-byte store regardless of
        // chunk_sz.  This overwrote adjacent stack memory when two local
        // char arrays were next to each other (only the last survived).
        let const_val = expr_lowering::emit_int_const(ctx, chunk_val, ir_ty.clone(), Span::dummy());

        // Compute the destination address: var_alloca + offset
        let dst_addr = if offset == 0 {
            var_alloca
        } else {
            let off_val = expr_lowering::emit_int_const_for_index(ctx, offset as i64);
            let (gep_val, gep_inst) =
                ctx.builder
                    .build_gep(var_alloca, vec![off_val], ir_ty.clone(), span);
            emit_inst_to_ctx(ctx, gep_inst);
            gep_val
        };

        // Emit the store instruction.
        let si = ctx.builder.build_store(const_val, dst_addr, span);
        emit_inst_to_ctx(ctx, si);

        offset += chunk_sz;
    }
}

/// Create a Value representing an integer constant for GEP indices.
/// Uses the builder to allocate a fresh value identifier.
/// Create an IR Value containing the given byte offset for use as a GEP index.
///
/// Previous bug: this function called `fresh_value()` without registering
/// the constant, producing a dangling Value that the backend defaulted to
/// 0, causing ALL initializer stores to target the same base address.
/// Check if an AST expression is a "natural lvalue" — i.e., it refers to
/// a memory location that can be addressed directly (variable, member
/// access, dereference, array subscript, compound literal).  Expressions
/// like function calls, conditional expressions, and casts produce rvalues
/// (struct data), not addressable memory locations.
fn is_natural_lvalue_expr(expr: &ast::Expression) -> bool {
    match expr {
        ast::Expression::Identifier { .. }
        | ast::Expression::MemberAccess { .. }
        | ast::Expression::PointerMemberAccess { .. }
        | ast::Expression::ArraySubscript { .. }
        | ast::Expression::CompoundLiteral { .. } => true,
        ast::Expression::UnaryOp {
            op: ast::UnaryOp::Deref,
            ..
        } => true,
        ast::Expression::Parenthesized { inner, .. } => is_natural_lvalue_expr(inner),
        ast::Expression::Cast { operand, .. } => is_natural_lvalue_expr(operand),
        _ => false,
    }
}

fn make_index_value(ctx: &mut expr_lowering::ExprLoweringContext<'_>, byte_offset: i64) -> Value {
    expr_lowering::emit_int_const_for_index(ctx, byte_offset)
}

// NOTE: compute_struct_field_offsets removed — it did not handle
// bitfield packing and caused buffer-overflow zeroing in
// zero_init_field.  All callers now use
// TypeBuilder::compute_struct_layout_with_fields instead.
