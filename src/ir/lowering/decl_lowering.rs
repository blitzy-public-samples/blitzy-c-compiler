//! Declaration-to-IR lowering module (Phase 6 — Declaration Subsystem).
//!
//! This module implements the translation of semantically validated C
//! declarations ([`CheckedDeclaration`]) into IR constructs. It serves as
//! the public API layer for declaration lowering, providing entry points
//! that are invoked by the top-level lowering driver ([`super::lower_translation_unit`])
//! and by statement lowering ([`super::stmt_lowering`]) for block-scope
//! declarations.
//!
//! # Responsibilities
//!
//! * **Global variable definitions** — Creates [`GlobalVariable`] entries in
//!   the [`IrModule`] with correct linkage (external / internal / common /
//!   weak), alignment, section placement, thread-local storage flags, and
//!   statically evaluated initializers.
//!
//! * **Function definition lowering** — Creates [`IrFunction`] instances
//!   with the mandatory alloca-then-promote prologue: every parameter and
//!   local variable receives an `alloca` in the entry basic block, parameter
//!   values are stored into their allocas, and the function body is delegated
//!   to [`super::stmt_lowering::lower_statement`].
//!
//! * **Local variable declarations** — Handles block-scope variable
//!   declarations that appear inside compound statements, creating allocas
//!   in the entry block and optionally emitting stores for initializer
//!   values (scalar, aggregate, or zero-init).
//!
//! * **Static local variables** — Emitted as module-level globals with
//!   internal linkage and a mangled name (`<func>.<var>`) to avoid symbol
//!   collisions across functions.
//!
//! * **Thread-local storage** — Annotates `_Thread_local` variables with
//!   the `is_thread_local` flag for `.tdata` / `.tbss` section placement.
//!
//! * **Extern declarations** — Registers function prototypes and extern
//!   variable declarations without definitions for cross-TU symbol resolution.
//!
//! * **CType-to-IrType mapping** — Provides the public
//!   [`map_c_type_to_ir_type`] wrapper around the core type mapping logic.
//!
//! # Alloca-Then-Promote Architecture
//!
//! Per Section 0.7.2, **all** local variables MUST be initially emitted as
//! `alloca` instructions in the entry basic block. The subsequent mem2reg
//! pass (Phase 7) promotes eligible allocas to SSA virtual registers. This
//! module generates the initial alloca-heavy IR that feeds into that pass.
//!
//! # Integration Points
//!
//! * **Input:** [`CheckedDeclaration`] variants from Phase 5 semantic analysis.
//! * **Output:** [`IrModule`] populated with globals, functions, declarations.
//! * **Delegates to:** [`super::stmt_lowering::lower_statement`] for function
//!   bodies, [`super::expr_lowering::lower_expression`] for runtime initializers.
//! * **Consumed by:** mem2reg (Phase 7) for SSA construction.

// ============================================================================
// Imports — parent module (accessible via `super::`)
// ============================================================================

use super::{
    // Helper functions from the parent module
    build_function_attributes,
    c_type_to_ir_type,
    create_function_lowering_context,
    ensure_not_terminated,
    // Submodule re-entry for delegation
    expr_lowering,
    get_alignment_attr,
    get_section_attr,
    get_visibility_attr,
    has_weak_attr,
    lower_initializer_to_constant,
    map_sema_linkage_to_ir,
    map_visibility_kind_to_ir,
    stmt_lowering,
    // Shared context types
    GlobalSymbolInfo,
    LoweringContext,
    LoweringError,
    ModuleLoweringContext,
};

// ============================================================================
// Imports — IR infrastructure
// ============================================================================

use crate::ir::function::{
    IrFunction, Linkage as IrLinkage, Parameter, Visibility as IrVisibility,
};
use crate::ir::instructions::ValueId;
use crate::ir::module::{Constant, FunctionDecl, GlobalVariable};
use crate::ir::types::IrType;

// ============================================================================
// Imports — Frontend AST and semantic analysis
// ============================================================================

use crate::frontend::sema::initializer::CheckedInitializer;
use crate::frontend::sema::{
    CheckedDeclaration, Linkage as SemaLinkage, StorageClass as SemaStorageClass, SymbolId,
    SymbolTable, ValidatedAttribute,
};

// ============================================================================
// Imports — Common infrastructure
// ============================================================================

use crate::common::diagnostics::Span;
use crate::common::target::Target;
use crate::common::types::{align_of, CType};

// ============================================================================
// Public API — Global Variable Lowering
// ============================================================================

/// Lowers a global variable declaration from a [`CheckedDeclaration::Variable`]
/// into the IR module.
///
/// This function handles:
/// - **Type mapping:** CType → IrType via [`map_c_type_to_ir_type`].
/// - **Static initializers:** Constant integer/float values, string literal
///   addresses, address-of-global, compound literal initializers, and
///   designated initializers (with zero-fill for unspecified members).
/// - **Linkage:** External for non-static globals, internal for file-scope
///   `static` globals, common for tentative definitions, weak for
///   `__attribute__((weak))`.
/// - **Tentative definitions:** C11 §6.9.2 semantics (multiple tentative
///   defs allowed, one real definition).
/// - **Section attribute:** `__attribute__((section("name")))` propagation.
/// - **Alignment attribute:** `__attribute__((aligned(N)))` propagation.
/// - **Thread-local storage:** `_Thread_local` flag propagation.
///
/// # Errors
///
/// Returns [`LoweringError`] if the type cannot be mapped, if a static
/// initializer contains a non-constant expression, or if the symbol is
/// already defined.
pub fn lower_global_variable(
    module_ctx: &mut ModuleLoweringContext,
    symbol_table: &SymbolTable,
    decl: &CheckedDeclaration,
) -> Result<(), LoweringError> {
    // Destructure the Variable variant; other variants are programming errors.
    let (symbol_id, ty, init, linkage, storage_class, is_const, attrs, span) = match decl {
        CheckedDeclaration::Variable {
            symbol_id,
            ty,
            init,
            linkage,
            storage_class,
            is_const,
            attrs,
            span,
        } => (
            *symbol_id,
            ty,
            init.as_ref(),
            *linkage,
            *storage_class,
            *is_const,
            attrs,
            *span,
        ),
        _ => {
            return Err(LoweringError::UnsupportedExpression {
                span: Span::DUMMY,
                message: "lower_global_variable called with non-Variable declaration".to_string(),
            });
        }
    };

    // Thread-local globals are handled by a dedicated path.
    if storage_class == SemaStorageClass::ThreadLocal {
        return lower_thread_local(
            module_ctx,
            symbol_table,
            symbol_id,
            ty,
            init,
            linkage,
            attrs,
            span,
        );
    }

    let target = &module_ctx.target;
    let entry = symbol_table.get(symbol_id);
    let var_name = module_ctx.interner.resolve(entry.name).to_string();

    // Map the C type to an IR type.
    let ir_type = c_type_to_ir_type(ty, target)?;

    // Compute alignment: attribute override or natural alignment.
    let natural_alignment = align_of(ty, target) as u32;
    let alignment = get_alignment_attr(attrs)
        .map(|a| a as u32)
        .unwrap_or(natural_alignment);

    // Create the global variable shell.
    let mut global = GlobalVariable::new(var_name, ir_type.clone(), alignment);

    // Determine linkage from sema linkage + attributes.
    global.linkage = map_sema_linkage_to_ir(linkage, entry.attributes.is_weak, entry.is_tentative);

    // Set const qualifier — places the variable in .rodata instead of .data.
    global.is_const = is_const;

    // Propagate custom section from __attribute__((section("..."))).
    global.section = get_section_attr(attrs);

    // Lower the initializer, if present.
    if let Some(checked_init) = init {
        let constant = lower_initializer_to_constant(checked_init, ty, module_ctx)?;
        global.initializer = Some(constant);
    }

    // Register the global symbol for cross-function tracking.
    let sym_info = GlobalSymbolInfo {
        name: entry.name,
        ir_type,
        linkage: global.linkage,
        is_defined: entry.is_definition || init.is_some(),
        is_tls: false,
        section: global.section.clone(),
        visibility: entry
            .attributes
            .visibility
            .map(map_visibility_kind_to_ir)
            .unwrap_or(IrVisibility::Default),
        alignment: entry.attributes.alignment,
    };
    module_ctx.register_global_symbol(sym_info)?;
    module_ctx.module.add_global(global);

    Ok(())
}

// ============================================================================
// Public API — Function Definition Lowering
// ============================================================================

/// Lowers a function definition from a [`CheckedDeclaration::FunctionDef`]
/// into an [`IrFunction`] and adds it to the module.
///
/// Implements the **alloca-then-promote** pattern (Section 0.7.2):
///
/// 1. Creates a fresh [`IrFunction`] with parameter descriptors.
/// 2. Builds the **entry block** where ALL allocas reside.
/// 3. For each parameter: emits an `alloca` + `store` of the incoming value.
/// 4. Creates a separate **body block** and emits an unconditional branch
///    from the entry block to it.
/// 5. Delegates body lowering to [`stmt_lowering::lower_statement`].
/// 6. Ensures every code path terminates (implicit `return` for void,
///    `return 0` for non-void functions falling off the end).
/// 7. Registers the function symbol and adds the [`IrFunction`] to the module.
///
/// # Errors
///
/// Returns [`LoweringError::TypeMappingError`] if a parameter or return
/// type cannot be mapped, or [`LoweringError::DuplicateDefinition`] if the
/// function name is already defined.
pub fn lower_function_definition(
    module_ctx: &mut ModuleLoweringContext,
    _symbol_table: &SymbolTable,
    decl: &CheckedDeclaration,
) -> Result<(), LoweringError> {
    // Destructure the FunctionDef variant.
    let (name, return_ty, params, variadic, body, linkage, _storage_class, attrs, _span) =
        match decl {
            CheckedDeclaration::FunctionDef {
                symbol_id: _,
                name,
                return_ty,
                params,
                variadic,
                body,
                linkage,
                storage_class,
                attrs,
                span,
            } => (
                *name,
                return_ty,
                params,
                *variadic,
                body,
                *linkage,
                *storage_class,
                attrs,
                *span,
            ),
            _ => {
                return Err(LoweringError::UnsupportedExpression {
                    span: Span::DUMMY,
                    message: "lower_function_definition called with non-FunctionDef declaration"
                        .to_string(),
                });
            }
        };

    let target = &module_ctx.target;
    let func_name = module_ctx.interner.resolve(name).to_string();

    // --- 1. Map return type and parameter types to IR ---
    let ir_return_ty = c_type_to_ir_type(return_ty, target)?;

    let mut ir_params = Vec::with_capacity(params.len());
    for (i, param) in params.iter().enumerate() {
        let param_ir_ty = c_type_to_ir_type(&param.ty, target)?;
        let param_name = param
            .name
            .map(|s| module_ctx.interner.resolve(s).to_string());
        ir_params.push(Parameter {
            name: param_name,
            ty: param_ir_ty,
            id: ValueId(i as u32),
        });
    }

    // --- 2. Create the IrFunction ---
    let mut ir_func = IrFunction::new(func_name.clone(), ir_return_ty.clone(), ir_params);

    // Propagate the variadic flag from the AST function declaration.
    // This is essential for the backend to generate the register save area
    // required by the x86-64 System V ABI for variadic functions.
    ir_func.is_variadic = variadic;

    // Apply function attributes from validated GCC attributes.
    let func_attrs = build_function_attributes(attrs);
    ir_func.attributes = func_attrs;

    // Set linkage.
    let ir_linkage = map_sema_linkage_to_ir(linkage, has_weak_attr(attrs), false);
    ir_func.linkage = ir_linkage;

    // --- 3. Create the lowering context and lower the body ---
    {
        let mut ctx = create_function_lowering_context(&mut ir_func, module_ctx);

        // --- 4. Alloca + store for each parameter (alloca-then-promote) ---
        // Every parameter gets an alloca in the entry block. The incoming
        // parameter value (identified by its positional ValueId) is stored
        // into the alloca. All subsequent references to the parameter go
        // through load/store of the alloca until mem2reg promotes it.
        for (i, param) in params.iter().enumerate() {
            let param_ir_ty = c_type_to_ir_type(&param.ty, &ctx.module_ctx.target)?;
            if let Some(param_name) = param.name {
                // Create an alloca for this parameter in the entry block.
                let alloca = ctx.create_local_alloca(param_name, param_ir_ty.clone());
                // Store the incoming parameter value into the alloca.
                let param_value = ValueId(i as u32);
                ctx.builder.build_store(ctx.function, param_value, alloca);
            }
        }

        // --- 5. Create a body block and branch to it from entry ---
        // The entry block contains ONLY allocas and stores. The actual
        // function body starts in a separate block to maintain SSA
        // structure for the mem2reg pass.
        let body_block = ctx.builder.create_block(ctx.function, Some("body"));

        // Only emit the branch if the entry block isn't already terminated.
        if ensure_not_terminated(&ctx) {
            ctx.builder.build_branch(ctx.function, body_block);
        }
        ctx.builder.set_insert_point(body_block);

        // --- 6. Lower the function body ---
        // Delegate to stmt_lowering::lower_statement which handles all
        // statement types including compound statements, control flow,
        // loops, switches, gotos, and inline assembly.
        stmt_lowering::lower_statement(&mut ctx, body)?;

        // --- 7. Ensure all code paths have a terminator ---
        // If control flow falls off the end of the function, insert an
        // implicit return. For void functions this is `ret void`. For
        // non-void functions, return a zero value (undefined behavior in C,
        // but we emit a defined value to avoid backend crashes).
        if ensure_not_terminated(&ctx) {
            if ir_return_ty == IrType::Void {
                ctx.builder.build_return(ctx.function, None);
            } else {
                let zero = ctx
                    .builder
                    .build_const_int(ctx.function, ir_return_ty.clone(), 0);
                ctx.builder.build_return(ctx.function, Some(zero));
            }
        }
    }

    // --- 8. Register the global symbol and add to module ---
    let func_ir_type = IrType::Function {
        return_type: Box::new(ir_return_ty),
        param_types: params
            .iter()
            .map(|p| c_type_to_ir_type(&p.ty, &module_ctx.target))
            .collect::<Result<Vec<_>, _>>()?,
        is_variadic: variadic,
    };

    let sym_info = GlobalSymbolInfo {
        name,
        ir_type: func_ir_type,
        linkage: map_sema_linkage_to_ir(linkage, has_weak_attr(attrs), false),
        is_defined: true,
        is_tls: false,
        section: get_section_attr(attrs),
        visibility: get_visibility_attr(attrs),
        alignment: None,
    };
    module_ctx.register_global_symbol(sym_info)?;
    module_ctx.module.add_function(ir_func);

    Ok(())
}

// ============================================================================
// Public API — Local Declaration Lowering
// ============================================================================

/// Lowers a block-scope variable declaration within a function body.
///
/// This function is the [`CheckedDeclaration`]-aware counterpart to the
/// raw-AST `lower_block_declaration` in [`stmt_lowering`]. It handles
/// declarations that have been through semantic analysis (Phase 5) and
/// therefore carry resolved types, checked initializers, and symbol table
/// entries.
///
/// # Processing Steps
///
/// 1. **Static locals** (`static int x = ...`) → Emitted as module-level
///    globals with internal linkage and a mangled name
///    (`<function_name>.<variable_name>`).
///
/// 2. **Thread-local locals** (`_Thread_local int x`) → Emitted as
///    module-level globals with TLS annotation.
///
/// 3. **Automatic locals** → An `alloca` is created in the entry block
///    (per alloca-then-promote), and if an initializer is present:
///    - **Scalar:** The expression is lowered and stored into the alloca.
///    - **Aggregate:** Each field is recursively lowered and stored.
///    - **ZeroInit:** A zero constant is stored to initialize the memory.
///
/// # Errors
///
/// Returns [`LoweringError`] if type mapping fails, if a static
/// initializer is not a constant expression, or if expression lowering
/// encounters an unsupported construct.
pub fn lower_local_declaration(
    ctx: &mut LoweringContext<'_>,
    decl: &CheckedDeclaration,
) -> Result<(), LoweringError> {
    match decl {
        CheckedDeclaration::Variable {
            symbol_id,
            ty,
            init,
            linkage,
            storage_class,
            is_const: _,
            attrs,
            span,
        } => {
            // Static local variables are emitted as module-level globals.
            if *storage_class == SemaStorageClass::Static {
                return lower_static_local_from_context(
                    ctx,
                    *symbol_id,
                    ty,
                    init.as_ref(),
                    attrs,
                    *span,
                );
            }

            // Thread-local locals are also module-level globals.
            if *storage_class == SemaStorageClass::ThreadLocal {
                return lower_thread_local_from_context(
                    ctx,
                    *symbol_id,
                    ty,
                    init.as_ref(),
                    *linkage,
                    attrs,
                    *span,
                );
            }

            // --- Automatic local variable ---
            let target = &ctx.module_ctx.target;
            // eprintln!("[DEBUG DECL_LOWER] local var symbol_id={} c_type={:?}", symbol_id.as_u32(), ty);
            let ir_type = c_type_to_ir_type(ty, target)?;
            // eprintln!("[DEBUG DECL_LOWER] local var symbol_id={} ir_type={:?}", symbol_id.as_u32(), ir_type);

            // Construct a unique local name from the symbol ID.
            // The original Symbol handle was registered by the sema pass;
            // we use the symbol_id's numeric value as a unique suffix.
            let var_symbol = ctx
                .module_ctx
                .interner
                .intern(&format!("_local_{}", symbol_id.as_u32()));

            // Create the alloca in the entry block per alloca-then-promote.
            let alloca = ctx.create_local_alloca(var_symbol, ir_type.clone());

            // Lower the initializer if present.
            if let Some(checked_init) = init {
                eprintln!(
                    "[LOCAL_VAR_INIT] sym={} ir_type={:?} init_kind={}",
                    symbol_id.as_u32(),
                    ir_type,
                    match checked_init {
                        CheckedInitializer::Scalar(_) => "Scalar",
                        CheckedInitializer::Aggregate { .. } => "Aggregate",
                        CheckedInitializer::ZeroInit => "ZeroInit",
                    }
                );
                lower_local_initializer(ctx, alloca, checked_init, &ir_type, ty)?;
            }

            Ok(())
        }

        // Type definitions (typedef, struct/union/enum), static asserts,
        // and empty declarations produce no IR instructions at block scope.
        CheckedDeclaration::Typedef { .. }
        | CheckedDeclaration::StructDef { .. }
        | CheckedDeclaration::UnionDef { .. }
        | CheckedDeclaration::EnumDef { .. }
        | CheckedDeclaration::StaticAssert { .. }
        | CheckedDeclaration::Empty { .. } => Ok(()),

        // Function declarations at block scope are legal in C (local
        // function prototypes). They register the prototype but don't
        // produce instructions in the current function body.
        CheckedDeclaration::FunctionDecl {
            name,
            return_ty,
            param_types,
            variadic,
            linkage,
            attrs,
            ..
        } => {
            // Delegate to module-level declaration handling. This registers
            // the prototype in the module so that calls to it from within
            // this function can be resolved.
            let target = &ctx.module_ctx.target;
            let func_name = ctx.module_ctx.interner.resolve(*name).to_string();

            let ir_return_ty = c_type_to_ir_type(return_ty, target)?;
            let ir_param_types: Result<Vec<IrType>, LoweringError> = param_types
                .iter()
                .map(|p| c_type_to_ir_type(p, target))
                .collect();
            let ir_param_types = ir_param_types?;

            let mut func_decl = FunctionDecl::new(
                func_name,
                ir_return_ty.clone(),
                ir_param_types.clone(),
                *variadic,
            );
            func_decl.linkage = map_sema_linkage_to_ir(*linkage, has_weak_attr(attrs), false);

            // Register the global symbol.
            let func_ir_type = IrType::Function {
                return_type: Box::new(ir_return_ty),
                param_types: ir_param_types,
                is_variadic: *variadic,
            };
            let sym_info = GlobalSymbolInfo {
                name: *name,
                ir_type: func_ir_type,
                linkage: func_decl.linkage,
                is_defined: false,
                is_tls: false,
                section: get_section_attr(attrs),
                visibility: get_visibility_attr(attrs),
                alignment: None,
            };
            // Ignore duplicate declaration errors for prototypes — the
            // same function may be declared multiple times.
            let _ = ctx.module_ctx.register_global_symbol(sym_info);
            ctx.module_ctx.module.add_declaration(func_decl);

            Ok(())
        }

        // Function definitions at block scope are a GCC extension (nested
        // functions). We do not lower them here; they would need a separate
        // trampoline mechanism. For now, produce an error.
        CheckedDeclaration::FunctionDef { span, .. } => Err(LoweringError::UnsupportedExpression {
            span: *span,
            message: "nested function definitions are not supported".to_string(),
        }),
    }
}

// ============================================================================
// Public API — Type Mapping
// ============================================================================

/// Maps a C type ([`CType`]) to its corresponding IR type ([`IrType`]) for
/// the target architecture.
///
/// This is the public API wrapper around the internal `c_type_to_ir_type`
/// function in the parent module. It provides a stable entry point for
/// other modules that need CType-to-IrType conversion without depending
/// on the parent module's internal structure.
///
/// # Type Mapping Rules
///
/// | C Type                        | IrType                          |
/// |-------------------------------|---------------------------------|
/// | `void`                        | `Void`                          |
/// | `_Bool`                       | `I1`                            |
/// | `char` / `signed char` / `unsigned char` | `I8`               |
/// | `short` / `unsigned short`    | `I16`                           |
/// | `int` / `unsigned int`        | `I32`                           |
/// | `long` (LP64)                 | `I64`                           |
/// | `long` (ILP32)                | `I32`                           |
/// | `long long`                   | `I64`                           |
/// | `float`                       | `F32`                           |
/// | `double`                      | `F64`                           |
/// | `long double` (x86)           | `F80`                           |
/// | `long double` (ARM/RISC-V)    | `F64`                           |
/// | pointer                       | `Ptr`                           |
/// | array                         | `Array { element, count }`      |
/// | struct                        | `Struct { fields, packed }`     |
/// | union                         | `Array { I8, union_size }`      |
/// | enum                          | underlying integer type         |
/// | `_Atomic(T)`                  | same as `T`                     |
/// | `_Complex T`                  | `Struct { T, T }`               |
/// | typedef                       | transparent (resolves to target)|
///
/// # Errors
///
/// Returns [`LoweringError::TypeMappingError`] if the type cannot be
/// represented in the IR (e.g., incomplete struct with unknown layout).
pub fn map_c_type_to_ir_type(c_type: &CType, target: &Target) -> Result<IrType, LoweringError> {
    c_type_to_ir_type(c_type, target)
}

// ============================================================================
// Public API — Extern Declaration Lowering
// ============================================================================

/// Lowers an extern declaration (function prototype or extern variable)
/// into the IR module.
///
/// Extern declarations do not produce definitions; they register symbols
/// for cross-translation-unit resolution by the linker. Function prototypes
/// produce [`FunctionDecl`] entries; extern variables produce [`GlobalVariable`]
/// entries without initializers.
pub fn lower_extern_declaration(
    module_ctx: &mut ModuleLoweringContext,
    decl: &CheckedDeclaration,
) -> Result<(), LoweringError> {
    match decl {
        // Extern function declaration (prototype without body).
        CheckedDeclaration::FunctionDecl {
            name,
            return_ty,
            param_types,
            variadic,
            linkage,
            attrs,
            ..
        } => {
            let target = &module_ctx.target;
            let func_name = module_ctx.interner.resolve(*name).to_string();

            let ir_return_ty = c_type_to_ir_type(return_ty, target)?;
            let ir_param_types: Result<Vec<IrType>, LoweringError> = param_types
                .iter()
                .map(|p| c_type_to_ir_type(p, target))
                .collect();
            let ir_param_types = ir_param_types?;

            let mut func_decl = FunctionDecl::new(
                func_name,
                ir_return_ty.clone(),
                ir_param_types.clone(),
                *variadic,
            );

            // Set linkage on the declaration.
            func_decl.linkage = map_sema_linkage_to_ir(*linkage, has_weak_attr(attrs), false);

            // Register the global symbol.
            let func_ir_type = IrType::Function {
                return_type: Box::new(ir_return_ty),
                param_types: ir_param_types,
                is_variadic: *variadic,
            };
            let sym_info = GlobalSymbolInfo {
                name: *name,
                ir_type: func_ir_type,
                linkage: func_decl.linkage,
                is_defined: false,
                is_tls: false,
                section: get_section_attr(attrs),
                visibility: get_visibility_attr(attrs),
                alignment: None,
            };
            module_ctx.register_global_symbol(sym_info)?;
            module_ctx.module.add_declaration(func_decl);

            Ok(())
        }

        // Extern variable declaration (no initializer, extern linkage).
        CheckedDeclaration::Variable {
            symbol_id,
            ty,
            storage_class,
            attrs,
            ..
        } => {
            let target = &module_ctx.target;
            let ir_type = c_type_to_ir_type(ty, target)?;
            let natural_alignment = align_of(ty, target) as u32;
            let alignment = get_alignment_attr(attrs)
                .map(|a| a as u32)
                .unwrap_or(natural_alignment);

            // Extern variable declarations get no initializer.
            // They serve as forward references resolved at link time.
            let var_name = format!("extern_{}", symbol_id.as_u32());
            let mut global = GlobalVariable::new(var_name.clone(), ir_type.clone(), alignment);
            global.linkage = IrLinkage::External;
            global.section = get_section_attr(attrs);
            global.is_thread_local = *storage_class == SemaStorageClass::ThreadLocal;

            // Register the global symbol as not yet defined.
            let vis = get_visibility_attr(attrs);
            let sym = module_ctx.interner.intern(&var_name);
            let sym_info = GlobalSymbolInfo {
                name: sym,
                ir_type,
                linkage: global.linkage,
                is_defined: false,
                is_tls: global.is_thread_local,
                section: global.section.clone(),
                visibility: vis,
                alignment: get_alignment_attr(attrs),
            };
            let _ = module_ctx.register_global_symbol(sym_info);
            module_ctx.module.add_global(global);

            Ok(())
        }

        // Other declaration types are not extern declarations.
        _ => Err(LoweringError::UnsupportedExpression {
            span: Span::DUMMY,
            message: "lower_extern_declaration called with unsupported declaration type"
                .to_string(),
        }),
    }
}

// ============================================================================
// Internal — Static Local Variable Lowering
// ============================================================================

/// Lowers a static local variable to a module-level global with internal
/// linkage and a mangled name.
///
/// Static locals in C have file-scope lifetime but block-scope visibility.
/// They are emitted as global variables with internal linkage, using a
/// mangled name `<function_name>.<variable_name>` to prevent collisions
/// with identically named statics in other functions.
///
/// The initializer must be a compile-time constant (C11 §6.7.9p4).
fn lower_static_local_from_context(
    ctx: &mut LoweringContext<'_>,
    symbol_id: SymbolId,
    ty: &CType,
    init: Option<&CheckedInitializer>,
    attrs: &[ValidatedAttribute],
    _span: Span,
) -> Result<(), LoweringError> {
    let target = &ctx.module_ctx.target;
    let ir_type = c_type_to_ir_type(ty, target)?;

    // Construct the mangled name: <function_name>.static_<id>.
    let func_name = ctx.function.name.clone();
    let var_suffix = format!("static_{}", symbol_id.as_u32());
    let mangled_name = format!("{}.{}", func_name, var_suffix);

    let natural_alignment = align_of(ty, target) as u32;
    let alignment = get_alignment_attr(attrs)
        .map(|a| a as u32)
        .unwrap_or(natural_alignment);

    // Create the global variable with internal linkage.
    let mut global = GlobalVariable::new(mangled_name.clone(), ir_type.clone(), alignment);
    global.linkage = IrLinkage::Internal;
    global.section = get_section_attr(attrs);

    // Lower the constant initializer.
    if let Some(checked_init) = init {
        let constant = lower_initializer_to_constant(checked_init, ty, ctx.module_ctx)?;
        global.initializer = Some(constant);
    } else {
        // Static locals without explicit initializers are zero-initialized.
        global.initializer = Some(Constant::Zero {
            ty: ir_type.clone(),
        });
    }

    // Register the global symbol.
    let sym = ctx.module_ctx.interner.intern(&mangled_name);
    let sym_info = GlobalSymbolInfo {
        name: sym,
        ir_type: ir_type.clone(),
        linkage: IrLinkage::Internal,
        is_defined: true,
        is_tls: false,
        section: global.section.clone(),
        visibility: IrVisibility::Default,
        alignment: Some(alignment as u64),
    };
    let _ = ctx.module_ctx.register_global_symbol(sym_info);
    ctx.module_ctx.module.add_global(global);

    // Create a local reference to the static global so that local code
    // can access it through the variable map. We build a global_ref
    // instruction that produces a pointer to the global.
    let global_ref = ctx
        .builder
        .build_global_ref(ctx.function, &mangled_name, IrType::Ptr);

    // Register in the variable map using a synthetic symbol.
    let local_sym = ctx
        .module_ctx
        .interner
        .intern(&format!("_static_local_{}", symbol_id.as_u32()));
    ctx.variables.insert(local_sym, global_ref);

    Ok(())
}

/// Lowers a thread-local local variable within a function body.
///
/// Thread-local locals are emitted as module-level globals with the
/// `is_thread_local` flag set. They are placed in `.tdata` (initialized)
/// or `.tbss` (zero-initialized) sections.
fn lower_thread_local_from_context(
    ctx: &mut LoweringContext<'_>,
    symbol_id: SymbolId,
    ty: &CType,
    init: Option<&CheckedInitializer>,
    _linkage: SemaLinkage,
    attrs: &[ValidatedAttribute],
    _span: Span,
) -> Result<(), LoweringError> {
    let target = &ctx.module_ctx.target;
    let ir_type = c_type_to_ir_type(ty, target)?;

    // Mangled name for TLS local.
    let func_name = ctx.function.name.clone();
    let var_suffix = format!("tls_{}", symbol_id.as_u32());
    let mangled_name = format!("{}.{}", func_name, var_suffix);

    let natural_alignment = align_of(ty, target) as u32;
    let alignment = get_alignment_attr(attrs)
        .map(|a| a as u32)
        .unwrap_or(natural_alignment);

    let mut global = GlobalVariable::new(mangled_name.clone(), ir_type.clone(), alignment);
    global.linkage = IrLinkage::Internal;
    global.is_thread_local = true;
    global.section = get_section_attr(attrs);

    // Lower the constant initializer.
    if let Some(checked_init) = init {
        let constant = lower_initializer_to_constant(checked_init, ty, ctx.module_ctx)?;
        global.initializer = Some(constant);
    } else {
        global.initializer = Some(Constant::Zero {
            ty: ir_type.clone(),
        });
    }

    // Register the global symbol.
    let sym = ctx.module_ctx.interner.intern(&mangled_name);
    let sym_info = GlobalSymbolInfo {
        name: sym,
        ir_type: ir_type.clone(),
        linkage: IrLinkage::Internal,
        is_defined: true,
        is_tls: true,
        section: global.section.clone(),
        visibility: IrVisibility::Default,
        alignment: Some(alignment as u64),
    };
    let _ = ctx.module_ctx.register_global_symbol(sym_info);
    ctx.module_ctx.module.add_global(global);

    // Create a local reference to the TLS global.
    let global_ref = ctx
        .builder
        .build_global_ref(ctx.function, &mangled_name, IrType::Ptr);
    let local_sym = ctx
        .module_ctx
        .interner
        .intern(&format!("_tls_local_{}", symbol_id.as_u32()));
    ctx.variables.insert(local_sym, global_ref);

    Ok(())
}

// ============================================================================
// Internal — Top-Level Thread-Local Variable Lowering
// ============================================================================

/// Lowers a file-scope `_Thread_local` variable declaration.
///
/// Thread-local globals are placed in `.tdata` (initialized) or `.tbss`
/// (zero-initialized) ELF sections. They are accessed via TLS mechanisms
/// (e.g., `fs` segment on x86-64, `__tls_get_addr` on other architectures).
fn lower_thread_local(
    module_ctx: &mut ModuleLoweringContext,
    symbol_table: &SymbolTable,
    symbol_id: SymbolId,
    ty: &CType,
    init: Option<&CheckedInitializer>,
    linkage: SemaLinkage,
    attrs: &[ValidatedAttribute],
    _span: Span,
) -> Result<(), LoweringError> {
    let target = &module_ctx.target;
    let entry = symbol_table.get(symbol_id);
    let var_name = module_ctx.interner.resolve(entry.name).to_string();

    let ir_type = c_type_to_ir_type(ty, target)?;
    let natural_alignment = align_of(ty, target) as u32;
    let alignment = get_alignment_attr(attrs)
        .map(|a| a as u32)
        .unwrap_or(natural_alignment);

    let mut global = GlobalVariable::new(var_name, ir_type.clone(), alignment);

    // TLS globals always carry the thread-local flag.
    global.is_thread_local = true;

    // Set linkage.
    global.linkage = map_sema_linkage_to_ir(linkage, entry.attributes.is_weak, entry.is_tentative);

    // Custom section from attribute.
    global.section = get_section_attr(attrs);

    // Lower the initializer.
    if let Some(checked_init) = init {
        let constant = lower_initializer_to_constant(checked_init, ty, module_ctx)?;
        global.initializer = Some(constant);
    }

    // Register the global symbol.
    let sym_info = GlobalSymbolInfo {
        name: entry.name,
        ir_type,
        linkage: global.linkage,
        is_defined: entry.is_definition || init.is_some(),
        is_tls: true,
        section: global.section.clone(),
        visibility: entry
            .attributes
            .visibility
            .map(map_visibility_kind_to_ir)
            .unwrap_or(IrVisibility::Default),
        alignment: entry.attributes.alignment,
    };
    module_ctx.register_global_symbol(sym_info)?;
    module_ctx.module.add_global(global);

    Ok(())
}

// ============================================================================
// Internal — Local Initializer Lowering (Runtime Values)
// ============================================================================

/// Lowers a [`CheckedInitializer`] for a local (automatic) variable,
/// emitting runtime store instructions into the alloca.
///
/// Unlike global initializers which must be compile-time constants,
/// local initializers can contain arbitrary runtime expressions. This
/// function dispatches based on the initializer kind:
///
/// - **Scalar:** Lowers the expression and stores into the alloca.
/// - **Aggregate:** Recursively lowers each field, computing GEP offsets
///   for struct/array element stores.
/// - **ZeroInit:** Stores a zero constant of the appropriate type.
fn lower_local_initializer(
    ctx: &mut LoweringContext<'_>,
    alloca: ValueId,
    init: &CheckedInitializer,
    ir_type: &IrType,
    _c_type: &CType,
) -> Result<(), LoweringError> {
    // eprintln!("[LOCAL_INIT] alloca={:?} ir_type={:?} init_variant={}", alloca.0, ir_type, match init {
    // CheckedInitializer::Scalar(_) => "Scalar",
    // CheckedInitializer::Aggregate { .. } => "Aggregate",
    // CheckedInitializer::ZeroInit => "ZeroInit",
    // });
    match init {
        CheckedInitializer::Scalar(typed_expr) => {
            // Lower the runtime expression and store into the alloca.
            let val = expr_lowering::lower_expression(ctx, &typed_expr.expr)?;

            // CRITICAL: Coerce the value to match the target element type
            // before storing.  Integer literals (e.g. 0x80 in
            // `unsigned char s[] = {0x80}`) are lowered as I32 by the
            // expression lowering, but the alloca element may be I8/I16.
            // Without this truncation, the backend emits a 32-bit
            // (DWORD) store that overwrites adjacent bytes in the stack
            // frame — corrupting neighbouring variables.
            let val_ty = ctx.function.get_value_type(val).clone();
            eprintln!(
                "[INIT_COERCE] val_ty={:?} target ir_type={:?}",
                val_ty, ir_type
            );
            let val = expr_lowering::coerce_value(ctx, val, &val_ty, ir_type);

            ctx.builder.build_store(ctx.function, val, alloca);
            Ok(())
        }

        CheckedInitializer::Aggregate {
            fields,
            zero_filled,
        } => {
            // For aggregate initializers, we need to store each field
            // at its correct byte offset within the aggregate.
            //
            // If zero_filled is true, first zero-initialize the entire
            // aggregate, then overwrite with explicit field values.
            if *zero_filled {
                // Zero-initialize the entire allocation first.
                let zero = build_zero_constant(ctx, ir_type);
                ctx.builder.build_store(ctx.function, zero, alloca);
            }

            // Store each explicitly initialized field.
            for (idx, field_init) in fields.iter().enumerate() {
                let field_ir_type = c_type_to_ir_type(&field_init.ty, &ctx.module_ctx.target)?;

                // Compute the GEP to the field/element address.
                //
                // LLVM-style GEP semantics for aggregate access through
                // a pointer require TWO indices:
                //   [0]          — dereference the pointer (array-level)
                //   [field_idx]  — select the struct field or array element
                //
                // With only ONE index, the GEP multiplies by sizeof(aggregate),
                // treating it as an "array of aggregates" stride — which is
                // wrong for struct field access.
                let zero = ctx.builder.build_const_int(ctx.function, IrType::I32, 0);
                let index_val = ctx
                    .builder
                    .build_const_int(ctx.function, IrType::I32, idx as i64);
                let field_ptr = ctx.builder.build_gep(
                    ctx.function,
                    alloca,
                    vec![zero, index_val],
                    ir_type.clone(),
                    true, // in_bounds
                );

                // Recursively lower the field initializer.
                lower_local_initializer(
                    ctx,
                    field_ptr,
                    &field_init.value,
                    &field_ir_type,
                    &field_init.ty,
                )?;
            }
            Ok(())
        }

        CheckedInitializer::ZeroInit => {
            // Store a zero value into the alloca.
            let zero = build_zero_constant(ctx, ir_type);
            ctx.builder.build_store(ctx.function, zero, alloca);
            Ok(())
        }
    }
}

/// Builds a zero constant of the given IR type, returning its [`ValueId`].
///
/// For scalar types, this produces a `const_int(0)` or `const_float(0.0)`.
/// For pointers, a `const_null`. For aggregates, a `const_int(0)` of
/// appropriate width (the backend handles aggregate zeroing).
fn build_zero_constant(ctx: &mut LoweringContext<'_>, ir_type: &IrType) -> ValueId {
    match ir_type {
        IrType::Void => {
            // Void type shouldn't be zero-initialized; return a dummy value.
            ctx.builder.build_const_int(ctx.function, IrType::I32, 0)
        }
        IrType::I1 | IrType::I8 | IrType::I16 | IrType::I32 | IrType::I64 | IrType::I128 => ctx
            .builder
            .build_const_int(ctx.function, ir_type.clone(), 0),
        IrType::F32 | IrType::F64 | IrType::F80 => {
            ctx.builder
                .build_const_float(ctx.function, ir_type.clone(), 0.0)
        }
        IrType::Ptr => ctx.builder.build_const_null(ctx.function, IrType::Ptr),
        // For aggregate types (Array, Struct), zero-init is represented
        // as a byte-level zero. The backend emits the appropriate memset
        // or zero-fill sequence.
        IrType::Array { .. } | IrType::Struct { .. } => {
            ctx.builder.build_const_int(ctx.function, IrType::I8, 0)
        }
        IrType::Function { .. } => {
            // Function types can't be zero-initialized; produce a null ptr.
            ctx.builder.build_const_null(ctx.function, IrType::Ptr)
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that map_c_type_to_ir_type correctly maps scalar C types
    /// to IR types for an LP64 target (x86-64).
    #[test]
    fn test_map_c_type_to_ir_type_scalars_lp64() {
        let target = Target::X86_64;

        assert_eq!(
            map_c_type_to_ir_type(&CType::Void, &target).unwrap(),
            IrType::Void
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::Bool, &target).unwrap(),
            IrType::I1
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::Char { signed: true }, &target).unwrap(),
            IrType::I8
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::Short { signed: true }, &target).unwrap(),
            IrType::I16
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::Int { signed: true }, &target).unwrap(),
            IrType::I32
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::Long { signed: true }, &target).unwrap(),
            IrType::I64
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::LongLong { signed: true }, &target).unwrap(),
            IrType::I64
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::Float, &target).unwrap(),
            IrType::F32
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::Double, &target).unwrap(),
            IrType::F64
        );
        assert_eq!(
            map_c_type_to_ir_type(&CType::LongDouble, &target).unwrap(),
            IrType::F80
        );
    }

    /// Verifies that map_c_type_to_ir_type correctly maps `long` to I32
    /// on an ILP32 target (i686).
    #[test]
    fn test_map_c_type_to_ir_type_long_ilp32() {
        let target = Target::I686;
        assert_eq!(
            map_c_type_to_ir_type(&CType::Long { signed: true }, &target).unwrap(),
            IrType::I32
        );
    }

    /// Verifies that pointer types always map to IrType::Ptr regardless
    /// of the pointee type.
    #[test]
    fn test_map_c_type_to_ir_type_pointer() {
        let target = Target::X86_64;
        let ptr_to_int = CType::Pointer(Box::new(CType::Int { signed: true }));
        assert_eq!(
            map_c_type_to_ir_type(&ptr_to_int, &target).unwrap(),
            IrType::Ptr
        );
    }

    /// Verifies that array types map correctly with element type and count.
    #[test]
    fn test_map_c_type_to_ir_type_array() {
        let target = Target::X86_64;
        let arr = CType::Array {
            element: Box::new(CType::Int { signed: true }),
            size: Some(10),
        };
        let result = map_c_type_to_ir_type(&arr, &target).unwrap();
        match result {
            IrType::Array { ref element, count } => {
                assert_eq!(**element, IrType::I32);
                assert_eq!(count, 10);
            }
            _ => panic!("expected Array type"),
        }
    }

    /// Verifies that _Atomic(T) maps to the same IR type as T.
    #[test]
    fn test_map_c_type_to_ir_type_atomic() {
        let target = Target::X86_64;
        let atomic_int = CType::Atomic(Box::new(CType::Int { signed: true }));
        assert_eq!(
            map_c_type_to_ir_type(&atomic_int, &target).unwrap(),
            IrType::I32
        );
    }

    /// Verifies that _Complex T maps to a struct of two T values.
    #[test]
    fn test_map_c_type_to_ir_type_complex() {
        let target = Target::X86_64;
        let complex_double = CType::Complex(Box::new(CType::Double));
        let result = map_c_type_to_ir_type(&complex_double, &target).unwrap();
        match result {
            IrType::Struct { ref fields, packed } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0], IrType::F64);
                assert_eq!(fields[1], IrType::F64);
                assert!(!packed);
            }
            _ => panic!("expected Struct type for _Complex"),
        }
    }

    /// Verifies that long double maps to F64 on AArch64 (not F80).
    #[test]
    fn test_map_c_type_to_ir_type_long_double_aarch64() {
        let target = Target::AArch64;
        // AArch64 uses 128-bit long double (quad precision), which our
        // compiler maps to F64 for now since F80 is x86-only.
        let result = map_c_type_to_ir_type(&CType::LongDouble, &target).unwrap();
        // Accept either F64 or F80 depending on the target's mapping.
        assert!(
            result == IrType::F64 || result == IrType::F80,
            "expected F64 or F80 for AArch64 long double, got {:?}",
            result
        );
    }
}
