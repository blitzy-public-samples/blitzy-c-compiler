// src/frontend/sema/initializer.rs
//
// Designated initializer semantic analysis for Phase 5 of the BCC compiler.
//
// This module implements C99/C11 initializer analysis (§6.7.9), transforming
// raw AST initializer trees into a linearized, type-checked representation
// (`CheckedInitializer`) consumed by IR lowering.
//
// Capabilities:
//   - Out-of-order field designation (`.field = val`)
//   - Nested designation (`.field.subfield = val`, `[i].field = val`)
//   - Array index designation (`[idx] = val`)
//   - Brace elision for nested struct/array aggregates
//   - Implicit zero-initialization of unspecified members (C11 §6.7.9p21)
//   - Union initialization (first member or designated member)
//   - String literal initialization of char/wchar_t arrays
//   - Duplicate field initialization detection (warns, last wins)
//   - Type compatibility validation via the type checker
//
// Architecture:
//   The analysis proceeds in two phases for struct/array initialization:
//
//   Phase 1 (Collection): Items are iterated, designators resolved at the
//     top level, and nested sub-designations accumulated per-field.
//   Phase 2 (Assembly): Accumulated sub-designations are converted into
//     synthetic initializer lists and recursively analyzed, producing the
//     hierarchical CheckedInitializer tree.
//
//   Brace elision is handled by sharing the outer item list with a mutable
//   position index; the inner aggregate consumes items from the flat list
//   until it is full or a designator is encountered.

// ---------------------------------------------------------------------------
// Imports
// ---------------------------------------------------------------------------

use crate::common::diagnostics::{DiagnosticEngine, Span};
use crate::common::fx_hash::FxHashSet;
use crate::common::string_interner::Interner;
use crate::common::target::Target;
use crate::common::type_builder::{
    self, compute_struct_layout, compute_union_layout, StructLayout,
};
use crate::common::types::{align_of, size_of, CType, FieldDef};
use crate::frontend::parser::ast::{Designator, Expression, Initializer, InitializerItem};
use crate::frontend::sema::constant_eval;
use crate::frontend::sema::scope::ScopeStack;
use crate::frontend::sema::symbol_table::SymbolTable;
use crate::frontend::sema::type_checker::{
    check_expression, insert_implicit_conversion, is_assignment_compatible, TypedExpression,
};

// ===========================================================================
// Public Types
// ===========================================================================

/// A fully type-checked and linearized initializer, ready for IR lowering.
///
/// Each variant maps to a different IR lowering strategy:
/// - `Scalar`: a single typed expression emitted as a store.
/// - `Aggregate`: a collection of field-level initializations with byte offsets.
/// - `ZeroInit`: emits zeroed memory (memset or zero-filled data section).
#[derive(Clone, Debug)]
pub enum CheckedInitializer {
    /// A single scalar value, fully type-checked and implicitly converted.
    Scalar(TypedExpression),

    /// An aggregate (struct, union, or array) initializer with per-field data.
    ///
    /// `fields` contains one entry per explicitly or implicitly initialized
    /// member, sorted by byte offset.  `zero_filled` is `true` when at least
    /// one member was implicitly zero-initialized.
    Aggregate {
        /// Per-member initialization entries, ordered by byte offset.
        fields: Vec<FieldInit>,
        /// Whether any member was implicitly zero-initialized.
        zero_filled: bool,
    },

    /// Implicit zero-initialization for an entire object.
    ZeroInit,
}

/// A single field/element initialization within an aggregate.
///
/// Produced during initializer analysis and consumed by IR lowering to emit
/// stores at the correct byte offsets within the aggregate's memory.
#[derive(Clone, Debug)]
pub struct FieldInit {
    /// Byte offset from the start of the containing aggregate.
    pub offset: usize,
    /// The C type of this field/element.
    pub ty: CType,
    /// The checked initializer value for this field/element.
    pub value: CheckedInitializer,
}

// ===========================================================================
// Internal Context
// ===========================================================================

/// Bundles all context needed during initializer analysis, avoiding excessive
/// parameter lists through the recursive call tree.
struct InitContext<'a> {
    diagnostics: &'a mut DiagnosticEngine,
    target: &'a Target,
    interner: &'a Interner,
    scopes: &'a ScopeStack,
    symbols: &'a SymbolTable,
}

/// Tracks the initialization state for a single struct/union field during
/// the collection phase of aggregate initialization.
enum FieldInitState {
    /// A complete, directly-provided initializer for this field.
    Direct(CheckedInitializer),
    /// Accumulated nested sub-designations (remaining designator chain,
    /// initializer value, source span) to be processed as a synthetic
    /// initializer list in the assembly phase.
    SubDesignations(Vec<(Vec<Designator>, Initializer, Span)>),
}

// ===========================================================================
// Public API — analyze_initializer
// ===========================================================================

/// Analyzes an initializer against its target type, producing a fully
/// type-checked `CheckedInitializer` tree.
///
/// This is the primary entry point for initializer semantic analysis.
/// It handles all C11 initializer forms: scalar expressions, brace-enclosed
/// aggregate initializers with optional designators, string literal
/// initialization of character arrays, and implicit zero-initialization.
///
/// # Parameters
///
/// * `init`        — the raw AST initializer to analyze.
/// * `target_type` — the declared type of the object being initialized.
/// * `diagnostics` — diagnostic engine for error/warning reporting.
/// * `target`      — compilation target for architecture-dependent layout.
/// * `interner`    — string interner for resolving field name symbols.
/// * `scopes`      — scope stack for expression type-checking.
/// * `symbols`     — symbol table for expression type-checking.
///
/// # Returns
///
/// `Ok(CheckedInitializer)` on success, `Err(())` if unrecoverable errors
/// were encountered (errors are also reported through `diagnostics`).
pub fn analyze_initializer(
    init: &Initializer,
    target_type: &CType,
    diagnostics: &mut DiagnosticEngine,
    target: &Target,
    interner: &Interner,
    scopes: &ScopeStack,
    symbols: &SymbolTable,
) -> Result<CheckedInitializer, ()> {
    let mut ctx = InitContext {
        diagnostics,
        target,
        interner,
        scopes,
        symbols,
    };
    analyze_init_inner(init, target_type, &mut ctx)
}

// ===========================================================================
// Public API — zero_init_for_type
// ===========================================================================

/// Produces a `CheckedInitializer` representing the implicit zero value for
/// the given C type, per C11 §6.7.9p10.
///
/// - Integer types: `0`
/// - Floating-point types: `0.0`
/// - Pointers: null pointer constant
/// - Aggregates (struct/union/array): recursively zero-initialized
///
/// This function cannot fail — every C type has a well-defined zero value.
pub fn zero_init_for_type(ty: &CType) -> CheckedInitializer {
    // For all types, `ZeroInit` instructs IR lowering to emit zeroed memory.
    // This is semantically correct for integers (0), floats (0.0), pointers
    // (NULL), and aggregates (recursively zero). Using a single variant
    // avoids inflating the tree while preserving the correct semantics.
    //
    // The type information is carried by the containing `FieldInit.ty`, so
    // the IR lowering phase knows how many bytes to zero and what alignment
    // to use via `size_of(ty, target)` and `align_of(ty, target)`.
    let canonical = ty.canonical();
    match canonical {
        // For scalar types: ZeroInit is the all-zero-bits representation,
        // which is `0` for integers, `+0.0` for IEEE 754 floats, and NULL
        // for pointers.
        _ if canonical.is_integer()
            || canonical.is_floating()
            || canonical.is_pointer() =>
        {
            CheckedInitializer::ZeroInit
        }

        // Aggregates: ZeroInit covers the full extent, including padding.
        CType::Struct { .. } | CType::Union { .. } | CType::Array { .. } => {
            CheckedInitializer::ZeroInit
        }

        // All other types (enum, atomic, typedef, void): ZeroInit.
        _ => CheckedInitializer::ZeroInit,
    }
}

// ===========================================================================
// Internal — Top-level Dispatch
// ===========================================================================

/// Routes the initializer to the appropriate handler based on initializer
/// form (expression vs. list) and target type (scalar vs. aggregate).
fn analyze_init_inner(
    init: &Initializer,
    target_type: &CType,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    let canonical = target_type.canonical();

    match init {
        Initializer::Expression(expr) => {
            // Special case: string literal initializing a character array.
            if canonical.is_array() {
                if let CType::Array { element, .. } = canonical {
                    if is_char_element_type(element) {
                        if let Expression::StringLiteral { .. } = expr.as_ref() {
                            return analyze_string_literal_init(expr, canonical, ctx);
                        }
                    }
                }
            }
            // General case: type-check the expression and validate
            // assignment compatibility with the target type.
            analyze_scalar_init(expr, target_type, ctx)
        }

        Initializer::List { items, span } => {
            // Empty initializer list → full zero-initialization.
            if items.is_empty() {
                return Ok(zero_init_for_type(target_type));
            }

            match canonical {
                CType::Struct { .. } => {
                    let mut pos = 0;
                    let result = analyze_struct_from_items(
                        items, &mut pos, canonical, *span, true, ctx,
                    )?;
                    warn_excess_items(items, pos, "struct initializer", ctx);
                    Ok(result)
                }
                CType::Union { .. } => {
                    analyze_union_init(items, canonical, *span, ctx)
                }
                CType::Array { element, .. } => {
                    // Special case: braced string literal for char array,
                    // e.g. `char s[] = { "hello" };`
                    if items.len() == 1
                        && items[0].designators.is_empty()
                        && is_char_element_type(element)
                    {
                        if let Initializer::Expression(expr) = &items[0].initializer {
                            if matches!(expr.as_ref(), Expression::StringLiteral { .. }) {
                                return analyze_string_literal_init(expr, canonical, ctx);
                            }
                        }
                    }
                    let mut pos = 0;
                    let result = analyze_array_from_items(
                        items, &mut pos, canonical, *span, true, ctx,
                    )?;
                    warn_excess_items(items, pos, "array initializer", ctx);
                    Ok(result)
                }
                _ if canonical.is_scalar() => {
                    // Scalar with braced init: `int x = { 5 };`
                    if items.len() > 1 {
                        ctx.diagnostics.warning(
                            items[1].span,
                            "excess elements in scalar initializer",
                        );
                    }
                    if items[0].designators.is_empty() {
                        analyze_init_inner(&items[0].initializer, target_type, ctx)
                    } else {
                        ctx.diagnostics.error(
                            items[0].span,
                            "designator in initializer for scalar type",
                        );
                        Err(())
                    }
                }
                _ => {
                    ctx.diagnostics.error(
                        *span,
                        format!(
                            "cannot initialize non-aggregate type '{}' with initializer list",
                            target_type
                        ),
                    );
                    Err(())
                }
            }
        }
    }
}

// ===========================================================================
// Internal — Scalar Initialization
// ===========================================================================

/// Type-checks a scalar (or copy) initializer expression against the target
/// type, inserting implicit conversions as needed.
fn analyze_scalar_init(
    expr: &Expression,
    target_type: &CType,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    // Type-check the expression to produce a typed, annotated result.
    let typed_expr = check_expression(
        expr,
        ctx.scopes,
        ctx.symbols,
        ctx.target,
        ctx.diagnostics,
        None, // no return type context during initialization
    );

    let expr_span = typed_expr.span;
    let rhs_is_null = is_null_pointer_constant(&typed_expr);

    // Validate assignment compatibility between target and initializer types.
    if !is_assignment_compatible(
        target_type,
        &typed_expr.ty,
        rhs_is_null,
        expr_span,
        ctx.diagnostics,
    ) {
        ctx.diagnostics.error(
            expr_span,
            format!(
                "initializing '{}' with incompatible type '{}'",
                target_type, typed_expr.ty
            ),
        );
        return Err(());
    }

    // Insert implicit conversion if the types are not structurally identical.
    let converted =
        if type_builder::types_compatible(target_type.canonical(), typed_expr.ty.canonical()) {
            typed_expr
        } else {
            insert_implicit_conversion(typed_expr, target_type, ctx.target)
        };

    Ok(CheckedInitializer::Scalar(converted))
}

// ===========================================================================
// Internal — Struct Initialization
// ===========================================================================

/// Analyzes a struct initializer by consuming items from `items[*pos..]`.
///
/// Handles both the top-level case (called from a braced initializer list)
/// and the brace-elision case (called when an outer list fills a nested
/// struct without explicit braces).
///
/// When `allow_designators` is `false` (brace elision), encountering a
/// designator stops consumption and returns to the caller.
fn analyze_struct_from_items(
    items: &[InitializerItem],
    pos: &mut usize,
    struct_type: &CType,
    span: Span,
    allow_designators: bool,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    let (fields, _struct_name) = match struct_type {
        CType::Struct { fields, name } => (fields, name),
        _ => unreachable!("analyze_struct_from_items called with non-struct type"),
    };

    if fields.is_empty() {
        return Ok(CheckedInitializer::Aggregate {
            fields: Vec::new(),
            zero_filled: false,
        });
    }

    // Compute the complete struct layout for field offset resolution.
    let layout = compute_struct_layout(fields, ctx.target);
    let field_count = fields.len();

    // Phase 1: Collect initializations per field.
    // Each slot in `field_inits` corresponds to the field at that index.
    let mut field_inits: Vec<Option<FieldInitState>> = (0..field_count).map(|_| None).collect();
    let mut initialized: FxHashSet<usize> = FxHashSet::default();
    let mut current_field: usize = 0;

    while *pos < items.len() {
        let item = &items[*pos];

        if !item.designators.is_empty() {
            if !allow_designators {
                // Brace elision: a designator belongs to the outer scope.
                break;
            }

            // --- Designated initializer ---
            let field_idx = resolve_first_field_designator(
                &item.designators[0],
                fields,
                item.span,
                ctx,
            )?;

            // Warn on duplicate initialization (C11 allows it; last wins).
            if initialized.contains(&field_idx) {
                ctx.diagnostics.warning(
                    item.span,
                    "initializer overrides prior initialization of this field",
                );
            }
            initialized.insert(field_idx);

            if item.designators.len() > 1 {
                // Nested designation (.field.subfield or .field[idx]):
                // accumulate sub-designations for phase 2 processing.
                let remaining = item.designators[1..].to_vec();
                accumulate_sub_designation(
                    &mut field_inits[field_idx],
                    remaining,
                    item.initializer.clone(),
                    item.span,
                );
                *pos += 1;
            } else {
                // Single-level designation: analyze the initializer directly.
                *pos += 1;
                let value = analyze_init_inner(
                    &item.initializer,
                    &fields[field_idx].ty,
                    ctx,
                )?;
                field_inits[field_idx] = Some(FieldInitState::Direct(value));
            }

            // After a designator, subsequent positional items continue from
            // the field following the designated one (C11 §6.7.9p17).
            current_field = field_idx + 1;
        } else {
            // --- Positional (sequential) initialization ---
            // Skip flexible array member at the end (only valid with designator).
            while current_field < field_count
                && is_flexible_array_member(fields, current_field, &layout)
            {
                current_field += 1;
            }

            if current_field >= field_count {
                // All non-flexible fields consumed; remaining items are excess.
                break;
            }

            let item_span = item.span;
            let field_type = &fields[current_field].ty;

            // Consume one logical initializer for this field's type,
            // handling brace elision for nested aggregates.
            let value = consume_init_for_type(
                items, pos, field_type, false, span, ctx,
            )?;

            if initialized.contains(&current_field) {
                ctx.diagnostics.warning(
                    item_span,
                    "initializer overrides prior initialization of this field",
                );
            }
            initialized.insert(current_field);
            field_inits[current_field] = Some(FieldInitState::Direct(value));
            current_field += 1;
        }
    }

    // Phase 2: Build the final aggregate result, processing any accumulated
    // sub-designations and zero-filling uninitialized fields.
    build_aggregate_result(fields, &layout, field_inits, &initialized, span, ctx)
}

// ===========================================================================
// Internal — Union Initialization
// ===========================================================================

/// Analyzes a union initializer.
///
/// Per C11 §6.7.9p17:
/// - Without a designator, the first member is initialized.
/// - With a `.field` designator, the named member is initialized.
/// - Only one member may be initialized at a time.
fn analyze_union_init(
    items: &[InitializerItem],
    union_type: &CType,
    span: Span,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    let fields = match union_type {
        CType::Union { fields, .. } => fields,
        _ => unreachable!("analyze_union_init called with non-union type"),
    };

    if fields.is_empty() {
        return Ok(CheckedInitializer::Aggregate {
            fields: Vec::new(),
            zero_filled: false,
        });
    }

    // Compute the union layout for size and alignment information.
    // All union members share offset zero; the total size is the maximum
    // of all member sizes, rounded to the union alignment.
    let union_layout = compute_union_layout(fields, ctx.target);
    let _ = union_layout.total_size; // Union size used for IR allocation.
    let _ = union_layout.alignment;  // Union alignment used for IR allocation.

    // Determine which member to initialize.
    let first_item = &items[0];
    let (member_idx, remaining_desigs) = if !first_item.designators.is_empty() {
        let idx = resolve_first_field_designator(
            &first_item.designators[0],
            fields,
            first_item.span,
            ctx,
        )?;
        let remaining = if first_item.designators.len() > 1 {
            first_item.designators[1..].to_vec()
        } else {
            Vec::new()
        };
        (idx, remaining)
    } else {
        // Default: initialize the first declared member.
        (0, Vec::new())
    };

    // Warn about excess elements; only one member can be initialized.
    if items.len() > 1 {
        ctx.diagnostics.warning(
            items[1].span,
            "excess elements in union initializer; only one member may be initialized",
        );
    }

    let member_type = &fields[member_idx].ty;
    let _member_bit_width = fields[member_idx].bit_width; // Track bit-field status.

    let value = if remaining_desigs.is_empty() {
        // Direct initialization of the selected member.
        analyze_init_inner(&first_item.initializer, member_type, ctx)?
    } else {
        // Nested designation into the selected member (.member.subfield).
        let sub_items = vec![InitializerItem {
            designators: remaining_desigs,
            initializer: first_item.initializer.clone(),
            span: first_item.span,
        }];
        analyze_init_inner(
            &Initializer::List {
                items: sub_items,
                span,
            },
            member_type,
            ctx,
        )?
    };

    Ok(CheckedInitializer::Aggregate {
        fields: vec![FieldInit {
            offset: 0, // All union members start at offset 0.
            ty: member_type.clone(),
            value,
        }],
        zero_filled: false,
    })
}

// ===========================================================================
// Internal — Array Initialization
// ===========================================================================

/// Analyzes an array initializer by consuming items from `items[*pos..]`.
///
/// Handles both fixed-size arrays (`int a[3] = {1, 2, 3}`) and unsized
/// arrays whose length is deduced from the initializer count
/// (`int a[] = {1, 2, 3}`).
fn analyze_array_from_items(
    items: &[InitializerItem],
    pos: &mut usize,
    array_type: &CType,
    span: Span,
    allow_designators: bool,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    let (element_type, array_size) = match array_type {
        CType::Array { element, size } => (element.as_ref(), *size),
        _ => unreachable!("analyze_array_from_items called with non-array type"),
    };

    // Compute element size and alignment for offset calculations.
    let elem_size = size_of(element_type, ctx.target);
    let _elem_align = align_of(element_type, ctx.target);

    // Collect (index, value) pairs for initialized elements.
    let mut element_inits: Vec<(usize, CheckedInitializer)> = Vec::new();
    let mut initialized: FxHashSet<usize> = FxHashSet::default();
    let mut current_index: usize = 0;
    let mut max_index_seen: usize = 0;

    while *pos < items.len() {
        let item = &items[*pos];

        if !item.designators.is_empty() {
            if !allow_designators {
                break; // Brace elision: designator belongs to outer scope.
            }

            // --- Array index designator: [N] = val ---
            let (idx, remaining) = resolve_first_array_designator(
                &item.designators,
                array_size,
                item.span,
                ctx,
            )?;

            // Validate index against declared array bounds.
            if let Some(max) = array_size {
                if idx >= max {
                    ctx.diagnostics.error(
                        item.span,
                        format!(
                            "array designator index {} exceeds array bounds (size {})",
                            idx, max
                        ),
                    );
                    *pos += 1;
                    continue;
                }
            }

            // Warn on duplicate initialization.
            if initialized.contains(&idx) {
                ctx.diagnostics.warning(
                    item.span,
                    format!(
                        "initializer overrides prior initialization of element [{}]",
                        idx
                    ),
                );
            }

            *pos += 1;
            let value = if remaining.is_empty() {
                analyze_init_inner(&item.initializer, element_type, ctx)?
            } else {
                // Nested designation into the element (e.g. [2].field = val).
                let sub_items = vec![InitializerItem {
                    designators: remaining,
                    initializer: item.initializer.clone(),
                    span: item.span,
                }];
                analyze_init_inner(
                    &Initializer::List {
                        items: sub_items,
                        span,
                    },
                    element_type,
                    ctx,
                )?
            };

            initialized.insert(idx);
            element_inits.push((idx, value));
            if idx >= max_index_seen {
                max_index_seen = idx + 1;
            }
            // After a designator, subsequent positional items continue from
            // the index following the designated one.
            current_index = idx + 1;
        } else {
            // --- Positional element initialization ---
            if let Some(max) = array_size {
                if current_index >= max {
                    break; // Array full; excess items handled by caller.
                }
            }

            // Warn on duplicate initialization (designator set earlier).
            if initialized.contains(&current_index) {
                ctx.diagnostics.warning(
                    item.span,
                    format!(
                        "initializer overrides prior initialization of element [{}]",
                        current_index
                    ),
                );
            }

            let value = consume_init_for_type(
                items, pos, element_type, false, span, ctx,
            )?;

            initialized.insert(current_index);
            element_inits.push((current_index, value));
            if current_index >= max_index_seen {
                max_index_seen = current_index + 1;
            }
            current_index += 1;
        }
    }

    // Determine effective array size: declared size, or deduced from the
    // maximum index seen.
    let effective_size = array_size.unwrap_or(max_index_seen);

    // Build a dense map of initialized elements for O(1) lookup.
    let mut init_map: Vec<Option<CheckedInitializer>> =
        (0..effective_size).map(|_| None).collect();
    for (idx, value) in element_inits {
        if idx < effective_size {
            init_map[idx] = Some(value);
        }
    }

    // Assemble the result: create a FieldInit for each element, zero-filling
    // any uninitialized positions.
    let mut result_fields: Vec<FieldInit> = Vec::with_capacity(effective_size);
    let mut any_zero_filled = false;

    for i in 0..effective_size {
        let value = match init_map[i].take() {
            Some(v) => v,
            None => {
                any_zero_filled = true;
                zero_init_for_type(element_type)
            }
        };
        result_fields.push(FieldInit {
            offset: i * elem_size,
            ty: element_type.clone(),
            value,
        });
    }

    Ok(CheckedInitializer::Aggregate {
        fields: result_fields,
        zero_filled: any_zero_filled,
    })
}

// ===========================================================================
// Internal — String Literal Initialization of Character Arrays
// ===========================================================================

/// Handles the special case of initializing `char[]` with a string literal.
///
/// Per C11 §6.7.9p14, a string literal can initialize an array of character
/// type. The literal provides the initial values including the implicit null
/// terminator (unless the array is too short to hold it).
fn analyze_string_literal_init(
    expr: &Expression,
    array_type: &CType,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    let (_element_type, array_size) = match array_type {
        CType::Array { element, size } => (element.as_ref(), *size),
        _ => unreachable!("string literal init on non-array type"),
    };

    // Extract the string value and span from the literal.
    let (str_bytes, str_span) = match expr {
        Expression::StringLiteral { value, span, .. } => (value, *span),
        _ => unreachable!("expected string literal expression"),
    };

    // The AST string value does NOT include the null terminator (per parser
    // convention). We account for it here: effective length = bytes + 1.
    let str_len_with_null = str_bytes.len() + 1;

    // Validate against declared array size.
    if let Some(declared) = array_size {
        if str_bytes.len() > declared {
            // C11 §6.7.9p14: excess bytes beyond the array size are not
            // stored. If the array is exactly the string length (without
            // the null), the null is dropped. If shorter, we warn.
            ctx.diagnostics.warning(
                str_span,
                format!(
                    "initializer-string for char array is too long \
                     ({} chars for array of size {})",
                    str_len_with_null, declared
                ),
            );
        }
    }

    // Type-check the string literal expression as a whole.
    // The result carries the `char [N]` type of the literal itself.
    let typed_expr = check_expression(
        expr,
        ctx.scopes,
        ctx.symbols,
        ctx.target,
        ctx.diagnostics,
        None,
    );

    // For string literal initialization of char arrays, we produce a Scalar
    // containing the full typed string expression. IR lowering treats this as
    // a memcpy from the string literal's `.rodata` storage into the target
    // array, truncating or null-padding as needed per the declared size.
    Ok(CheckedInitializer::Scalar(typed_expr))
}

// ===========================================================================
// Internal — Consume Initializer for a Given Type (Brace Elision)
// ===========================================================================

/// Consumes one logical initializer from `items[*pos..]` for the given
/// target type, handling brace elision when appropriate.
///
/// This is the key function for brace elision. When the target type is an
/// aggregate and the current item is NOT a braced sub-list, the function
/// recursively fills the aggregate by consuming multiple items from the
/// flat outer list.
fn consume_init_for_type(
    items: &[InitializerItem],
    pos: &mut usize,
    target_type: &CType,
    allow_designators: bool,
    parent_span: Span,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    if *pos >= items.len() {
        // No more items: implicit zero-initialization.
        return Ok(zero_init_for_type(target_type));
    }

    let item = &items[*pos];
    let canonical = target_type.canonical();

    // If the current item is a braced list, use it directly (no elision).
    if matches!(&item.initializer, Initializer::List { .. }) {
        *pos += 1;
        return analyze_init_inner(&item.initializer, target_type, ctx);
    }

    // String literal for character array: consume directly.
    if let CType::Array { element, .. } = canonical {
        if is_char_element_type(element) {
            if let Initializer::Expression(expr) = &item.initializer {
                if matches!(expr.as_ref(), Expression::StringLiteral { .. }) {
                    *pos += 1;
                    return analyze_init_inner(&item.initializer, target_type, ctx);
                }
            }
        }
    }

    // If the target is an aggregate and the item is a bare expression,
    // apply brace elision: the inner aggregate consumes items from the
    // outer flat list.
    if canonical.is_aggregate() {
        match canonical {
            CType::Struct { .. } => {
                return analyze_struct_from_items(
                    items,
                    pos,
                    canonical,
                    parent_span,
                    allow_designators,
                    ctx,
                );
            }
            CType::Array { .. } => {
                return analyze_array_from_items(
                    items,
                    pos,
                    canonical,
                    parent_span,
                    allow_designators,
                    ctx,
                );
            }
            CType::Union { fields, .. } => {
                // Union brace elision: consume one item for the first member.
                if !fields.is_empty() {
                    let first_member_type = &fields[0].ty;
                    let value = consume_init_for_type(
                        items, pos, first_member_type, false, parent_span, ctx,
                    )?;
                    return Ok(CheckedInitializer::Aggregate {
                        fields: vec![FieldInit {
                            offset: 0,
                            ty: first_member_type.clone(),
                            value,
                        }],
                        zero_filled: false,
                    });
                }
                return Ok(zero_init_for_type(target_type));
            }
            _ => {}
        }
    }

    // Scalar type: consume one item directly.
    *pos += 1;
    analyze_init_inner(&item.initializer, target_type, ctx)
}

// ===========================================================================
// Internal — Designator Resolution
// ===========================================================================

/// Resolves a field designator (`.field`) against a struct/union field list,
/// returning the matching field index.
///
/// Searches named fields directly, then recurses into anonymous struct/union
/// members to find transparent access paths (C11 §6.7.2.1p13).
fn resolve_first_field_designator(
    designator: &Designator,
    fields: &[FieldDef],
    span: Span,
    ctx: &mut InitContext<'_>,
) -> Result<usize, ()> {
    match designator {
        Designator::Field(sym) => {
            let name = ctx.interner.resolve(*sym);
            let _sym_id = sym.as_u32(); // Symbol ID for diagnostics context.

            match find_field_index(fields, name) {
                Some(idx) => Ok(idx),
                None => {
                    ctx.diagnostics.error(
                        span,
                        format!("field '{}' does not exist in struct/union type", name),
                    );
                    Err(())
                }
            }
        }
        Designator::Index(_) => {
            ctx.diagnostics.error(
                span,
                "array index designator '[N]' cannot be used with struct/union type",
            );
            Err(())
        }
    }
}

/// Resolves an array index designator (`[N]`) at the front of a designator
/// chain, returning the integer index and any remaining designators.
fn resolve_first_array_designator(
    designators: &[Designator],
    _array_size: Option<usize>,
    span: Span,
    ctx: &mut InitContext<'_>,
) -> Result<(usize, Vec<Designator>), ()> {
    match &designators[0] {
        Designator::Index(index_expr) => {
            // Evaluate the index as a compile-time integer constant.
            let index_val = constant_eval::evaluate_integer_constant(
                index_expr,
                ctx.diagnostics,
                ctx.target,
            )?;

            if index_val < 0 {
                ctx.diagnostics.error(
                    span,
                    format!("array designator index {} is negative", index_val),
                );
                return Err(());
            }

            let index = index_val as usize;
            let remaining = if designators.len() > 1 {
                designators[1..].to_vec()
            } else {
                Vec::new()
            };
            Ok((index, remaining))
        }
        Designator::Field(_) => {
            ctx.diagnostics.error(
                span,
                "field designator '.name' cannot be used with array type",
            );
            Err(())
        }
    }
}

/// Searches for a field by name within a struct/union field list.
///
/// First checks named fields directly, then transparently searches into
/// anonymous struct/union members per C11 §6.7.2.1p13 (anonymous members
/// have `name = None`).
fn find_field_index(fields: &[FieldDef], name: &str) -> Option<usize> {
    // First pass: direct named field match.
    for (idx, field) in fields.iter().enumerate() {
        if let Some(ref field_name) = field.name {
            if field_name == name {
                return Some(idx);
            }
        }
    }

    // Second pass: search anonymous struct/union members transparently.
    for (idx, field) in fields.iter().enumerate() {
        if field.name.is_none() {
            match &field.ty {
                CType::Struct { fields: sub, .. } | CType::Union { fields: sub, .. } => {
                    for sub_field in sub {
                        if sub_field.name.as_deref() == Some(name) {
                            return Some(idx);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    None
}

// ===========================================================================
// Internal — Sub-Designation Accumulation
// ===========================================================================

/// Accumulates a nested sub-designation for a field.
///
/// When multiple designators target sub-members of the same field
/// (e.g., `.inner.x = 1, .inner.y = 2`), the sub-designations are
/// collected and later assembled into a synthetic initializer list.
fn accumulate_sub_designation(
    state: &mut Option<FieldInitState>,
    remaining_desigs: Vec<Designator>,
    init: Initializer,
    span: Span,
) {
    match state {
        Some(FieldInitState::SubDesignations(subs)) => {
            subs.push((remaining_desigs, init, span));
        }
        Some(FieldInitState::Direct(_)) => {
            // A direct init is overridden by subsequent nested designations.
            *state = Some(FieldInitState::SubDesignations(vec![(
                remaining_desigs,
                init,
                span,
            )]));
        }
        None => {
            *state = Some(FieldInitState::SubDesignations(vec![(
                remaining_desigs,
                init,
                span,
            )]));
        }
    }
}

// ===========================================================================
// Internal — Aggregate Result Builder (Phase 2)
// ===========================================================================

/// Phase 2: Processes accumulated field initialization states and builds the
/// final `CheckedInitializer::Aggregate`.
///
/// For fields with sub-designations, creates a synthetic `Initializer::List`
/// and recursively analyzes it against the field's type. Uninitialized fields
/// are zero-filled per C11 §6.7.9p21.
fn build_aggregate_result(
    fields: &[FieldDef],
    layout: &StructLayout,
    mut field_inits: Vec<Option<FieldInitState>>,
    initialized: &FxHashSet<usize>,
    span: Span,
    ctx: &mut InitContext<'_>,
) -> Result<CheckedInitializer, ()> {
    let field_count = fields.len();
    let _struct_size = layout.total_size;    // Total struct size for IR.
    let _struct_align = layout.alignment;    // Struct alignment for IR.

    let mut result_fields: Vec<FieldInit> = Vec::with_capacity(field_count);
    let mut any_zero_filled = false;

    for (idx, field) in fields.iter().enumerate() {
        // Skip flexible array members that were not explicitly initialized.
        if is_flexible_array_member(fields, idx, layout) && !initialized.contains(&idx) {
            continue;
        }

        // Retrieve the field's layout for byte offset, size, and alignment.
        let field_layout = &layout.fields[idx];
        let field_offset = field_layout.offset;
        let _field_size = field_layout.size;
        let _field_align = field_layout.alignment;
        let _field_bit_width = field.bit_width; // None for regular fields.

        let value = match field_inits[idx].take() {
            Some(FieldInitState::Direct(v)) => v,
            Some(FieldInitState::SubDesignations(subs)) => {
                // Create a synthetic initializer list from accumulated
                // sub-designations and recursively analyze.
                let sub_items: Vec<InitializerItem> = subs
                    .into_iter()
                    .map(|(desigs, init, sp)| InitializerItem {
                        designators: desigs,
                        initializer: init,
                        span: sp,
                    })
                    .collect();

                let merged_span = if sub_items.len() >= 2 {
                    Span::merge(sub_items[0].span, sub_items[sub_items.len() - 1].span)
                } else {
                    span
                };

                analyze_init_inner(
                    &Initializer::List {
                        items: sub_items,
                        span: merged_span,
                    },
                    &field.ty,
                    ctx,
                )?
            }
            None => {
                // Field not explicitly initialized → zero-initialization.
                any_zero_filled = true;
                zero_init_for_type(&field.ty)
            }
        };

        result_fields.push(FieldInit {
            offset: field_offset,
            ty: field.ty.clone(),
            value,
        });
    }

    Ok(CheckedInitializer::Aggregate {
        fields: result_fields,
        zero_filled: any_zero_filled,
    })
}

// ===========================================================================
// Internal — Utility Functions
// ===========================================================================

/// Returns `true` if the element type is a character type that can be
/// initialized with a string literal (char, signed char, unsigned char).
fn is_char_element_type(element: &CType) -> bool {
    matches!(element.canonical(), CType::Char { .. })
}

/// Returns `true` if the field at `idx` is a flexible array member
/// (the last field in a struct with `CType::Array { size: None, .. }`).
fn is_flexible_array_member(fields: &[FieldDef], idx: usize, layout: &StructLayout) -> bool {
    layout.has_flexible_array && idx == fields.len() - 1
}

/// Returns `true` if the typed expression represents a null pointer constant:
/// integer zero literal or `(void *)0` cast expression.
fn is_null_pointer_constant(typed: &TypedExpression) -> bool {
    if !typed.is_constant {
        return false;
    }
    match &typed.expr {
        Expression::IntegerLiteral { value, .. } => *value == 0,
        Expression::Cast { operand, .. } => {
            // Detect `(void *)0` pattern.
            if let Expression::IntegerLiteral { value, .. } = operand.as_ref() {
                *value == 0
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Emits a warning if there are unconsumed items past `pos` in the list.
fn warn_excess_items(
    items: &[InitializerItem],
    pos: usize,
    context: &str,
    ctx: &mut InitContext<'_>,
) {
    if pos < items.len() {
        let excess_span = items[pos].span;
        ctx.diagnostics.warning(
            excess_span,
            format!(
                "excess elements in {} ({} provided, {} consumed)",
                context,
                items.len(),
                pos
            ),
        );
    }
}
