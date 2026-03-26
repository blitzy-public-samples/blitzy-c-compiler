//! Initializer analysis for Phase 5 (semantic analysis) of BCC.
//!
//! Handles C11 designated initializer semantics including:
//!
//! - **Out-of-order field designations:** Fields can be initialized in any order
//!   via `.field = value` syntax.
//! - **Nested designations:** `.field.subfield = value` and `[i][j] = value`
//!   chains that drill into nested aggregates.
//! - **Array index designations:** `[N] = value` and GCC range extension
//!   `[low ... high] = value`.
//! - **Brace elision:** Per C11 §6.7.9, inner braces can be omitted for nested
//!   aggregates; initializers are consumed left-to-right to fill sub-aggregates.
//! - **String literal initialization:** `char s[] = "hello"` with automatic size
//!   determination and truncation warnings.
//! - **Implicit zero-initialization:** Unspecified members are zeroed.
//! - **Constant initializer validation:** For static/global variables, all leaf
//!   expressions must be compile-time constants.
//!
//! This module is crucial for C11 compliance and Linux kernel builds which use
//! complex designated initializers extensively.
//!
//! # Dependencies
//!
//! This module depends ONLY on `crate::common` and `crate::frontend::parser::ast`.
//! It does NOT depend on `crate::ir`, `crate::passes`, or `crate::backend`.
//!
//! # Zero-Dependency Compliance
//!
//! Uses only `std` and `crate::` references. No external crates.

use crate::common::diagnostics::{Diagnostic, DiagnosticEngine, Span};
use crate::common::string_interner::{Interner, Symbol};
use crate::common::target::Target;
use crate::common::type_builder::TypeBuilder;
use crate::common::types::{CType, StructField};

use crate::frontend::parser::ast::*;

// ===========================================================================
// AnalyzedInit — Semantically Resolved Initializer
// ===========================================================================

/// A fully resolved initializer where every field/element has a concrete
/// value or is explicitly zero-initialized.
///
/// After analysis, every aggregate member has a definite initialization
/// status — either an explicit expression, a recursively analyzed
/// sub-initializer, or an implicit zero.
#[derive(Debug, Clone)]
pub enum AnalyzedInit {
    /// Simple expression initializer with type-checked expression.
    ///
    /// The `target_type` records the type the expression initializes,
    /// which may differ from the expression's natural type due to
    /// implicit conversions.
    Expression {
        /// The initializer expression (cloned from AST).
        expr: Expression,
        /// The target type being initialized.
        target_type: CType,
    },

    /// Struct initializer: (field_index, initializer) pairs ordered by
    /// field index. Every field in the struct has an entry — either an
    /// explicit init or a [`Zero`](AnalyzedInit::Zero).
    Struct {
        /// Per-field initializers in field-index order.
        fields: Vec<(usize, AnalyzedInit)>,
        /// The struct type being initialized.
        struct_type: CType,
    },

    /// Union initializer: exactly one member is initialized.
    ///
    /// Without a designator, the first member is chosen (C11 §6.7.9p17).
    /// With a `.field` designator, the named member is selected.
    Union {
        /// Index of the initialized field within the union's field list.
        field_index: usize,
        /// The initializer for the selected field.
        initializer: Box<AnalyzedInit>,
        /// The union type being initialized.
        union_type: CType,
    },

    /// Array initializer: (array_index, initializer) pairs.
    ///
    /// Every element up to `array_size` has an entry — either an explicit
    /// init or a [`Zero`](AnalyzedInit::Zero).
    Array {
        /// Per-element initializers in index order.
        elements: Vec<(usize, AnalyzedInit)>,
        /// The array type (with resolved size if originally incomplete).
        array_type: CType,
        /// Total number of array elements after analysis.
        array_size: usize,
    },

    /// Implicit zero initialization for unspecified members.
    ///
    /// Generated for struct fields, union padding, and array elements
    /// that have no explicit initializer. Equivalent to `= {0}` or
    /// static-duration zero-initialization.
    Zero {
        /// The type being zero-initialized.
        target_type: CType,
    },
}

// ===========================================================================
// InitializerAnalyzer
// ===========================================================================

/// Semantic analyzer for C11 initializers with full designated-initializer
/// support, brace elision, and GCC extension handling.
///
/// The analyzer is parameterized by references to the diagnostic engine,
/// type builder, string interner, and target architecture. It processes
/// AST `Initializer` nodes and produces fully resolved `AnalyzedInit` trees.
pub struct InitializerAnalyzer<'a> {
    /// Diagnostic engine for error/warning emission.
    diagnostics: &'a mut DiagnosticEngine,
    /// Type builder for sizeof/alignof and struct layout queries.
    type_builder: &'a TypeBuilder,
    /// Target architecture for ABI-dependent size queries.
    target: Target,
    /// String interner for resolving Symbol handles to field name strings.
    interner: &'a Interner,
}

impl<'a> InitializerAnalyzer<'a> {
    // ===================================================================
    // Public API
    // ===================================================================

    /// Create a new initializer analyzer.
    ///
    /// # Arguments
    ///
    /// * `diagnostics` — Mutable reference to the diagnostic engine.
    /// * `type_builder` — Reference to the type builder for layout queries.
    /// * `target` — The compilation target architecture.
    /// * `interner` — Reference to the string interner for symbol resolution.
    pub fn new(
        diagnostics: &'a mut DiagnosticEngine,
        type_builder: &'a TypeBuilder,
        target: Target,
        interner: &'a Interner,
    ) -> Self {
        Self {
            diagnostics,
            type_builder,
            target,
            interner,
        }
    }

    /// Analyze an initializer against a target type.
    ///
    /// This is the main entry point. It dispatches to expression analysis
    /// or brace-enclosed list analysis depending on the initializer form.
    ///
    /// # Arguments
    ///
    /// * `init` — The AST initializer node.
    /// * `target_type` — The declared type of the variable being initialized.
    /// * `span` — Source span for diagnostic reporting.
    ///
    /// # Returns
    ///
    /// A fully resolved `AnalyzedInit` tree on success, or `Err(())` if
    /// unrecoverable errors were encountered (errors are also emitted to
    /// the diagnostic engine).
    #[allow(clippy::result_unit_err)]
    pub fn analyze_initializer(
        &mut self,
        init: &Initializer,
        target_type: &CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        let resolved = Self::resolve_type(target_type);
        match init {
            Initializer::Expression(expr) => self.analyze_simple_init(expr, resolved, span),
            Initializer::List {
                designators_and_initializers,
                span: list_span,
                ..
            } => self.analyze_init_list(designators_and_initializers, resolved, *list_span),
        }
    }

    /// Check whether an analyzed initializer is a compile-time constant.
    ///
    /// For static and global variables, C11 requires initializers to be
    /// constant expressions. This method recursively checks all leaf
    /// expressions in the analyzed init tree.
    ///
    /// # Rules
    ///
    /// - Integer, float, and string literals are constant.
    /// - Address-of-global and address-of-function are constant.
    /// - Compound literals with constant sub-expressions are constant.
    /// - `Zero` initializers are always constant.
    /// - All other expressions are conservatively non-constant.
    pub fn is_constant_initializer(&self, init: &AnalyzedInit) -> bool {
        match init {
            AnalyzedInit::Expression { expr, .. } => Self::is_constant_expression(expr),
            AnalyzedInit::Struct { fields, .. } => {
                fields.iter().all(|(_, f)| self.is_constant_initializer(f))
            }
            AnalyzedInit::Union { initializer, .. } => self.is_constant_initializer(initializer),
            AnalyzedInit::Array { elements, .. } => elements
                .iter()
                .all(|(_, e)| self.is_constant_initializer(e)),
            AnalyzedInit::Zero { .. } => true,
        }
    }

    // ===================================================================
    // Simple (expression) initializer
    // ===================================================================

    /// Analyze a simple expression initializer (no braces).
    ///
    /// Handles the special case of string literal initialization of char
    /// arrays, and the general case of scalar/compatible type initialization.
    fn analyze_simple_init(
        &mut self,
        expr: &Expression,
        target_type: &CType,
        _span: Span,
    ) -> Result<AnalyzedInit, ()> {
        // Special case: string literal initializing a char array.
        if let Expression::StringLiteral {
            segments,
            prefix,
            span: str_span,
        } = expr
        {
            if let CType::Array(ref elem, size) = *target_type {
                let inner_elem = Self::resolve_type(elem);
                if Self::is_char_element_type(inner_elem, *prefix) {
                    return self.analyze_string_init(
                        segments,
                        *prefix,
                        *str_span,
                        inner_elem,
                        size,
                        target_type,
                    );
                }
            }
        }

        // General case: expression initializes a scalar or compatible type.
        Ok(AnalyzedInit::Expression {
            expr: expr.clone(),
            target_type: target_type.clone(),
        })
    }

    // ===================================================================
    // Brace-enclosed initializer list dispatch
    // ===================================================================

    /// Dispatch a brace-enclosed initializer list based on target type.
    ///
    /// - Struct → `analyze_struct_init_cursor`
    /// - Union → `analyze_union_init_cursor`
    /// - Array → `analyze_array_init_cursor`
    /// - Scalar → first element initializes the scalar (C11 §6.7.9p11)
    fn analyze_init_list(
        &mut self,
        items: &[DesignatedInitializer],
        target_type: &CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        match target_type {
            CType::Struct { ref fields, .. } => {
                let fields_snapshot = fields.clone();
                let type_clone = target_type.clone();
                let mut pos = 0;
                self.analyze_struct_init_cursor(items, &mut pos, &fields_snapshot, type_clone, span)
            }
            CType::Union { ref fields, .. } => {
                let fields_snapshot = fields.clone();
                let type_clone = target_type.clone();
                let mut pos = 0;
                self.analyze_union_init_cursor(items, &mut pos, &fields_snapshot, type_clone, span)
            }
            CType::Array(ref elem, size) => {
                let elem_clone = (**elem).clone();
                let size_copy = *size;
                let type_clone = target_type.clone();
                let mut pos = 0;
                self.analyze_array_init_cursor(
                    items,
                    &mut pos,
                    &elem_clone,
                    size_copy,
                    type_clone,
                    span,
                )
            }
            _ => {
                // Scalar type: first element initializes the scalar (C11 §6.7.9p11).
                if items.is_empty() {
                    return Ok(AnalyzedInit::Zero {
                        target_type: target_type.clone(),
                    });
                }
                if items.len() > 1 {
                    self.diagnostics
                        .emit_warning(span, "excess elements in scalar initializer");
                }
                let first = &items[0];
                self.analyze_initializer_value(&first.initializer, target_type, first.span)
            }
        }
    }

    /// Analyze an initializer value (expression or nested list) against a type.
    ///
    /// This is an internal helper that dispatches based on the `Initializer`
    /// variant, used after designator resolution.
    fn analyze_initializer_value(
        &mut self,
        init: &Initializer,
        target_type: &CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        let resolved = Self::resolve_type(target_type);
        match init {
            Initializer::Expression(expr) => self.analyze_simple_init(expr, resolved, span),
            Initializer::List {
                designators_and_initializers,
                span: list_span,
                ..
            } => self.analyze_init_list(designators_and_initializers, resolved, *list_span),
        }
    }

    // ===================================================================
    // Struct initializer — cursor-based analysis
    // ===================================================================

    /// Analyze a struct initializer list using a shared position cursor.
    ///
    /// The cursor (`pos`) enables brace elision: when called from the
    /// top-level, `pos` starts at 0 and items are the brace-enclosed list.
    /// When called during brace elision from an outer aggregate handler,
    /// `pos` is the current position in the outer list, and the handler
    /// consumes only as many items as needed to fill the struct.
    fn analyze_struct_init_cursor(
        &mut self,
        items: &[DesignatedInitializer],
        pos: &mut usize,
        fields: &[StructField],
        struct_type: CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        let num_fields = fields.len();
        let mut field_inits: Vec<Option<AnalyzedInit>> = (0..num_fields).map(|_| None).collect();
        let mut current_field: usize = 0;

        while *pos < items.len() && current_field < num_fields {
            let item_span = items[*pos].span;
            let desig_count = items[*pos].designators.len();

            // ---- Process designators ----
            if desig_count > 0 {
                let first_desig = items[*pos].designators[0].clone();
                match first_desig {
                    Designator::Field(sym, dspan) => {
                        // Look up field by name.
                        if let Some(idx) = self.find_field_index(fields, sym) {
                            current_field = idx;
                            if desig_count > 1 {
                                // Nested designation: .field.subfield = value
                                let desigs: Vec<Designator> = items[*pos].designators[1..].to_vec();
                                let init_ref = items[*pos].initializer.clone();
                                self.emit_field_override_warning(
                                    &field_inits,
                                    current_field,
                                    fields,
                                    item_span,
                                );
                                let sub = self.analyze_with_nested_desig(
                                    &desigs,
                                    &init_ref,
                                    Self::resolve_type(&fields[current_field].ty),
                                    item_span,
                                )?;
                                field_inits[current_field] = Some(sub);
                                current_field += 1;
                                *pos += 1;
                                continue;
                            }
                            // Single-level designation: fall through.
                        } else if let Some((path, _)) = self.find_in_anonymous(fields, sym) {
                            // Field inside an anonymous struct/union member.
                            let desigs: Vec<Designator> = items[*pos].designators[1..].to_vec();
                            let init_ref = items[*pos].initializer.clone();
                            let outer_idx = path[0];
                            self.emit_field_override_warning(
                                &field_inits,
                                outer_idx,
                                fields,
                                item_span,
                            );
                            let sub = self.analyze_anonymous_field_init(
                                &path, &desigs, &init_ref, fields, item_span,
                            )?;
                            field_inits[outer_idx] = Some(sub);
                            current_field = outer_idx + 1;
                            *pos += 1;
                            continue;
                        } else {
                            // Unknown field name.
                            let name = self.interner.resolve(sym);
                            self.diagnostics.emit_error(
                                dspan,
                                format!("unknown field '{}' in struct initializer", name),
                            );
                            *pos += 1;
                            continue;
                        }
                    }
                    Designator::Index(_, _) | Designator::IndexRange(_, _, _) => {
                        // Array designator in struct context — brace elision
                        // ended; return control to outer handler.
                        break;
                    }
                }
            }

            // ---- Skip zero-width anonymous bitfield padding ----
            while current_field < num_fields {
                let f = &fields[current_field];
                if f.name.is_none() && f.bit_width == Some(0) {
                    current_field += 1;
                    continue;
                }
                break;
            }
            if current_field >= num_fields {
                break;
            }

            // ---- Analyze element for the current field ----
            let field_ty = fields[current_field].ty.clone();
            let resolved_field_ty = Self::resolve_type(&field_ty);
            self.emit_field_override_warning(&field_inits, current_field, fields, items[*pos].span);
            let init = self.analyze_one_element(items, pos, resolved_field_ty, span)?;
            field_inits[current_field] = Some(init);
            current_field += 1;
        }

        // Excess elements diagnostic.
        if *pos < items.len() && current_field >= num_fields && items[*pos].designators.is_empty() {
            self.diagnostics
                .emit_warning(span, "excess elements in struct initializer");
        }

        // Build result with zero-fill for uninitialized fields.
        let mut result_fields = Vec::with_capacity(num_fields);
        for (i, opt) in field_inits.into_iter().enumerate() {
            match opt {
                Some(init) => result_fields.push((i, init)),
                None => result_fields.push((
                    i,
                    AnalyzedInit::Zero {
                        target_type: fields[i].ty.clone(),
                    },
                )),
            }
        }

        Ok(AnalyzedInit::Struct {
            fields: result_fields,
            struct_type,
        })
    }

    /// Emit a warning if a field was already initialized (override).
    fn emit_field_override_warning(
        &mut self,
        field_inits: &[Option<AnalyzedInit>],
        field_idx: usize,
        fields: &[StructField],
        span: Span,
    ) {
        if field_idx < field_inits.len() && field_inits[field_idx].is_some() {
            let name = fields[field_idx].name.as_deref().unwrap_or("<anonymous>");
            self.diagnostics.emit(Diagnostic::warning(
                span,
                format!("initializer overrides prior initialization of '{}'", name),
            ));
        }
    }

    // ===================================================================
    // Union initializer — cursor-based analysis
    // ===================================================================

    /// Analyze a union initializer list.
    ///
    /// Only ONE member can be initialized:
    /// - Without designator: the FIRST member (C11 §6.7.9p17).
    /// - With `.field` designator: the named member.
    /// - Multiple initializers: last one wins, earlier ones trigger warnings.
    fn analyze_union_init_cursor(
        &mut self,
        items: &[DesignatedInitializer],
        pos: &mut usize,
        fields: &[StructField],
        union_type: CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        if items.is_empty() || *pos >= items.len() || fields.is_empty() {
            return Ok(AnalyzedInit::Zero {
                target_type: union_type,
            });
        }

        let item_span = items[*pos].span;
        let desig_count = items[*pos].designators.len();
        let field_index: usize;

        if desig_count > 0 {
            let first_desig = items[*pos].designators[0].clone();
            match first_desig {
                Designator::Field(sym, dspan) => {
                    match self.find_field_index(fields, sym) {
                        Some(idx) => {
                            field_index = idx;
                            if desig_count > 1 {
                                // Nested designation within union member.
                                let desigs: Vec<Designator> = items[*pos].designators[1..].to_vec();
                                let init_ref = items[*pos].initializer.clone();
                                *pos += 1;
                                let sub = self.analyze_with_nested_desig(
                                    &desigs,
                                    &init_ref,
                                    Self::resolve_type(&fields[field_index].ty),
                                    item_span,
                                )?;
                                if *pos < items.len() {
                                    self.diagnostics
                                        .emit_warning(span, "excess elements in union initializer");
                                }
                                return Ok(AnalyzedInit::Union {
                                    field_index,
                                    initializer: Box::new(sub),
                                    union_type,
                                });
                            }
                            // Single-level designation: fall through.
                        }
                        None => {
                            let name = self.interner.resolve(sym);
                            self.diagnostics.emit_error(
                                dspan,
                                format!("unknown field '{}' in union initializer", name),
                            );
                            *pos += 1;
                            return Err(());
                        }
                    }
                }
                Designator::Index(_, dspan) | Designator::IndexRange(_, _, dspan) => {
                    self.diagnostics
                        .emit_error(dspan, "array designator in union initializer");
                    *pos += 1;
                    return Err(());
                }
            }
        } else {
            // No designator: initialize the first member (C11 §6.7.9p17).
            field_index = 0;
        }

        let field_ty = fields[field_index].ty.clone();
        let resolved = Self::resolve_type(&field_ty);
        let init = self.analyze_one_element(items, pos, resolved, span)?;

        // Warn about excess elements.
        if *pos < items.len() {
            self.diagnostics
                .emit_warning(span, "excess elements in union initializer");
        }

        Ok(AnalyzedInit::Union {
            field_index,
            initializer: Box::new(init),
            union_type,
        })
    }

    // ===================================================================
    // Array initializer — cursor-based analysis
    // ===================================================================

    /// Analyze an array initializer list using a shared position cursor.
    ///
    /// Supports positional initialization, `[N] = value` index designators,
    /// GCC range designators `[low ... high] = value`, incomplete array size
    /// inference, and brace elision for nested aggregates.
    fn analyze_array_init_cursor(
        &mut self,
        items: &[DesignatedInitializer],
        pos: &mut usize,
        elem_type: &CType,
        known_size: Option<usize>,
        array_type: CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        let resolved_elem = Self::resolve_type(elem_type).clone();
        let mut element_inits: Vec<(usize, AnalyzedInit)> = Vec::new();
        let mut current_index: usize = 0;
        let mut max_index_seen: Option<usize> = None;

        while *pos < items.len() {
            // Check array bounds for positional items.
            if let Some(size) = known_size {
                if current_index >= size && items[*pos].designators.is_empty() {
                    break;
                }
            }

            let item_span = items[*pos].span;
            let desig_count = items[*pos].designators.len();

            // ---- Process designators ----
            if desig_count > 0 {
                let first_desig = items[*pos].designators[0].clone();
                match first_desig {
                    Designator::Index(ref expr, dspan) => {
                        match Self::try_eval_integer(expr) {
                            Some(val) => {
                                if val < 0 {
                                    self.diagnostics.emit_error(
                                        dspan,
                                        "array index in initializer must be non-negative",
                                    );
                                    *pos += 1;
                                    continue;
                                }
                                let idx = val as usize;
                                if let Some(size) = known_size {
                                    if idx >= size {
                                        self.diagnostics.emit_error(
                                            dspan,
                                            format!(
                                                "array index {} in initializer exceeds array bounds (size {})",
                                                idx, size
                                            ),
                                        );
                                        *pos += 1;
                                        continue;
                                    }
                                }
                                current_index = idx;

                                if desig_count > 1 {
                                    // Nested: [N].field or [N][M]
                                    let desigs: Vec<Designator> =
                                        items[*pos].designators[1..].to_vec();
                                    let init_ref = items[*pos].initializer.clone();
                                    let sub = self.analyze_with_nested_desig(
                                        &desigs,
                                        &init_ref,
                                        &resolved_elem,
                                        item_span,
                                    )?;
                                    element_inits.push((current_index, sub));
                                    Self::update_max(&mut max_index_seen, current_index);
                                    current_index += 1;
                                    *pos += 1;
                                    continue;
                                }
                                // Single [N] = value: fall through.
                            }
                            None => {
                                self.diagnostics.emit_error(
                                    dspan,
                                    "array designator index is not a constant expression",
                                );
                                *pos += 1;
                                continue;
                            }
                        }
                    }
                    Designator::IndexRange(ref low_expr, ref high_expr, dspan) => {
                        // GCC range designator: [low ... high] = value
                        match (
                            Self::try_eval_integer(low_expr),
                            Self::try_eval_integer(high_expr),
                        ) {
                            (Some(low), Some(high)) => {
                                if low < 0 || high < 0 || low > high {
                                    self.diagnostics.emit_error(
                                        dspan,
                                        "invalid array index range in initializer",
                                    );
                                    *pos += 1;
                                    continue;
                                }
                                let low_idx = low as usize;
                                let high_idx = high as usize;
                                if let Some(size) = known_size {
                                    if high_idx >= size {
                                        self.diagnostics.emit_error(
                                            dspan,
                                            format!(
                                                "array index range {}...{} exceeds array bounds (size {})",
                                                low_idx, high_idx, size
                                            ),
                                        );
                                        *pos += 1;
                                        continue;
                                    }
                                }
                                // Analyze the value once, clone for each.
                                let init_ref = items[*pos].initializer.clone();
                                let init_value = self.analyze_initializer_value(
                                    &init_ref,
                                    &resolved_elem,
                                    item_span,
                                )?;
                                for idx in low_idx..=high_idx {
                                    element_inits.push((idx, init_value.clone()));
                                    Self::update_max(&mut max_index_seen, idx);
                                }
                                current_index = high_idx + 1;
                                *pos += 1;
                                continue;
                            }
                            _ => {
                                self.diagnostics.emit_error(
                                    dspan,
                                    "array range designator indices are not constant expressions",
                                );
                                *pos += 1;
                                continue;
                            }
                        }
                    }
                    Designator::Field(_, _) => {
                        // Field designator in array context — brace elision
                        // ended; return control to outer handler.
                        break;
                    }
                }
            }

            // Bounds check for positional element.
            if let Some(size) = known_size {
                if current_index >= size {
                    break;
                }
            }

            // ---- Analyze element at current_index ----
            let init = self.analyze_one_element(items, pos, &resolved_elem, span)?;
            element_inits.push((current_index, init));
            Self::update_max(&mut max_index_seen, current_index);
            current_index += 1;
        }

        // Determine final array size.
        let final_size = match known_size {
            Some(s) => s,
            None => match max_index_seen {
                Some(m) => m + 1,
                None => 0,
            },
        };

        // Excess elements diagnostic.
        if *pos < items.len() && items[*pos].designators.is_empty() {
            if let Some(size) = known_size {
                if current_index >= size {
                    self.diagnostics
                        .emit_warning(span, "excess elements in array initializer");
                }
            }
        }

        // Deduplicate: last write wins for each index.
        let mut final_elements: Vec<Option<AnalyzedInit>> = vec![None; final_size];
        for (idx, init) in element_inits {
            if idx < final_size {
                final_elements[idx] = Some(init);
            }
        }

        // Build result with zero-fill for gaps.
        let mut result_elements = Vec::with_capacity(final_size);
        for (i, opt) in final_elements.into_iter().enumerate() {
            match opt {
                Some(init) => result_elements.push((i, init)),
                None => result_elements.push((
                    i,
                    AnalyzedInit::Zero {
                        target_type: elem_type.clone(),
                    },
                )),
            }
        }

        // Reconstruct array type with resolved size if originally incomplete.
        let final_array_type = if known_size.is_none() && final_size > 0 {
            CType::Array(Box::new(elem_type.clone()), Some(final_size))
        } else {
            array_type
        };

        Ok(AnalyzedInit::Array {
            elements: result_elements,
            array_type: final_array_type,
            array_size: final_size,
        })
    }

    /// Update the tracked maximum index.
    #[inline]
    fn update_max(max: &mut Option<usize>, idx: usize) {
        match *max {
            Some(m) if m >= idx => {}
            _ => *max = Some(idx),
        }
    }

    // ===================================================================
    // Single element analysis with brace elision
    // ===================================================================

    /// Analyze one element from the initializer list, handling brace elision.
    ///
    /// If the current item is a brace-enclosed list, it is analyzed directly
    /// against the target type. If it is a simple expression and the target
    /// type is an aggregate, brace elision kicks in: the cursor-based
    /// handler for the aggregate consumes items from the shared list.
    fn analyze_one_element(
        &mut self,
        items: &[DesignatedInitializer],
        pos: &mut usize,
        target_type: &CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        if *pos >= items.len() {
            return Ok(AnalyzedInit::Zero {
                target_type: target_type.clone(),
            });
        }

        // Clone the initializer to avoid borrow conflicts.
        let item_init = items[*pos].initializer.clone();
        let _item_span = items[*pos].span;

        match &item_init {
            Initializer::List {
                designators_and_initializers,
                span: list_span,
                ..
            } => {
                // Braced sub-list: analyze directly against the target type.
                *pos += 1;
                self.analyze_init_list(designators_and_initializers, target_type, *list_span)
            }
            Initializer::Expression(expr) => {
                // Check for string literal initializing a char array.
                if let Expression::StringLiteral {
                    segments,
                    prefix,
                    span: sspan,
                } = &**expr
                {
                    if let CType::Array(ref elem, size) = *target_type {
                        let inner = Self::resolve_type(elem);
                        if Self::is_char_element_type(inner, *prefix) {
                            *pos += 1;
                            return self.analyze_string_init(
                                segments,
                                *prefix,
                                *sspan,
                                inner,
                                size,
                                target_type,
                            );
                        }
                    }
                }

                if Self::is_aggregate_type(target_type) {
                    // Brace elision: pass the outer items and cursor to
                    // the sub-aggregate handler, which consumes only as
                    // many items as needed to fill the sub-aggregate.
                    match target_type {
                        CType::Struct { ref fields, .. } => {
                            let fields_snap = fields.clone();
                            let ty = target_type.clone();
                            self.analyze_struct_init_cursor(items, pos, &fields_snap, ty, span)
                        }
                        CType::Union { ref fields, .. } => {
                            let fields_snap = fields.clone();
                            let ty = target_type.clone();
                            self.analyze_union_init_cursor(items, pos, &fields_snap, ty, span)
                        }
                        CType::Array(ref elem, size) => {
                            let elem_clone = (**elem).clone();
                            let size_copy = *size;
                            let ty = target_type.clone();
                            self.analyze_array_init_cursor(
                                items,
                                pos,
                                &elem_clone,
                                size_copy,
                                ty,
                                span,
                            )
                        }
                        _ => {
                            // Should not reach here (is_aggregate was true).
                            *pos += 1;
                            Ok(AnalyzedInit::Expression {
                                expr: (**expr).clone(),
                                target_type: target_type.clone(),
                            })
                        }
                    }
                } else {
                    // Scalar: consume one item.
                    *pos += 1;
                    Ok(AnalyzedInit::Expression {
                        expr: (**expr).clone(),
                        target_type: target_type.clone(),
                    })
                }
            }
        }
    }

    // ===================================================================
    // Nested designator resolution
    // ===================================================================

    /// Resolve a chain of designators and produce an analyzed init.
    ///
    /// For `.field.subfield = value`, the chain is `[.subfield]` (the first
    /// `.field` was already consumed by the struct handler). This function
    /// recursively descends through struct/union/array types.
    fn analyze_with_nested_desig(
        &mut self,
        designators: &[Designator],
        init: &Initializer,
        target_type: &CType,
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        if designators.is_empty() {
            return self.analyze_initializer_value(init, target_type, span);
        }

        let resolved = Self::resolve_type(target_type);

        match &designators[0] {
            Designator::Field(sym, dspan) => match resolved {
                CType::Struct { ref fields, .. } => match self.find_field_index(fields, *sym) {
                    Some(idx) => {
                        let field_ty = fields[idx].ty.clone();
                        let sub = self.analyze_with_nested_desig(
                            &designators[1..],
                            init,
                            Self::resolve_type(&field_ty),
                            span,
                        )?;
                        let mut result_fields = Vec::with_capacity(fields.len());
                        for (i, f) in fields.iter().enumerate() {
                            if i == idx {
                                result_fields.push((i, sub.clone()));
                            } else {
                                result_fields.push((
                                    i,
                                    AnalyzedInit::Zero {
                                        target_type: f.ty.clone(),
                                    },
                                ));
                            }
                        }
                        Ok(AnalyzedInit::Struct {
                            fields: result_fields,
                            struct_type: resolved.clone(),
                        })
                    }
                    None => {
                        let name = self.interner.resolve(*sym);
                        self.diagnostics.emit_error(
                            *dspan,
                            format!("unknown field '{}' in nested designator", name),
                        );
                        Err(())
                    }
                },
                CType::Union { ref fields, .. } => match self.find_field_index(fields, *sym) {
                    Some(idx) => {
                        let field_ty = fields[idx].ty.clone();
                        let sub = self.analyze_with_nested_desig(
                            &designators[1..],
                            init,
                            Self::resolve_type(&field_ty),
                            span,
                        )?;
                        Ok(AnalyzedInit::Union {
                            field_index: idx,
                            initializer: Box::new(sub),
                            union_type: resolved.clone(),
                        })
                    }
                    None => {
                        let name = self.interner.resolve(*sym);
                        self.diagnostics.emit_error(
                            *dspan,
                            format!("unknown field '{}' in nested designator", name),
                        );
                        Err(())
                    }
                },
                _ => {
                    self.diagnostics
                        .emit_error(*dspan, "field designator on non-struct/union type");
                    Err(())
                }
            },

            Designator::Index(ref expr, dspan) => match resolved {
                CType::Array(ref elem, size) => match Self::try_eval_integer(expr) {
                    Some(val) => {
                        let idx = val as usize;
                        if let Some(s) = size {
                            if idx >= *s {
                                self.diagnostics.emit_error(
                                    *dspan,
                                    format!("array index {} exceeds bounds (size {})", idx, s),
                                );
                                return Err(());
                            }
                        }
                        let elem_ty = (**elem).clone();
                        let sub = self.analyze_with_nested_desig(
                            &designators[1..],
                            init,
                            Self::resolve_type(&elem_ty),
                            span,
                        )?;
                        let final_size = size.unwrap_or(idx + 1);
                        let mut elements = Vec::with_capacity(final_size);
                        for i in 0..final_size {
                            if i == idx {
                                elements.push((i, sub.clone()));
                            } else {
                                elements.push((
                                    i,
                                    AnalyzedInit::Zero {
                                        target_type: elem_ty.clone(),
                                    },
                                ));
                            }
                        }
                        Ok(AnalyzedInit::Array {
                            elements,
                            array_type: resolved.clone(),
                            array_size: final_size,
                        })
                    }
                    None => {
                        self.diagnostics.emit_error(
                            *dspan,
                            "array index in nested designator is not a constant expression",
                        );
                        Err(())
                    }
                },
                _ => {
                    self.diagnostics
                        .emit_error(*dspan, "array designator on non-array type");
                    Err(())
                }
            },

            Designator::IndexRange(_, _, dspan) => {
                self.diagnostics.emit_error(
                    *dspan,
                    "range designator not allowed in nested designator chain",
                );
                Err(())
            }
        }
    }

    // ===================================================================
    // Anonymous struct/union member handling
    // ===================================================================

    /// Analyze an initializer that targets a field inside an anonymous
    /// struct/union member.
    ///
    /// `path` is a chain of field indices from the outer struct to the
    /// target field (e.g., `[2, 0]` means field 2 of outer is an anonymous
    /// struct, and field 0 within it is the target).
    fn analyze_anonymous_field_init(
        &mut self,
        path: &[usize],
        remaining_desig: &[Designator],
        init: &Initializer,
        outer_fields: &[StructField],
        span: Span,
    ) -> Result<AnalyzedInit, ()> {
        if path.is_empty() {
            return self.analyze_initializer_value(init, &CType::Void, span);
        }

        let outer_idx = path[0];
        let outer_field_ty = &outer_fields[outer_idx].ty;
        let resolved = Self::resolve_type(outer_field_ty);

        if path.len() == 1 {
            // Final level: analyze the initializer against this field.
            if remaining_desig.is_empty() {
                return self.analyze_initializer_value(init, resolved, span);
            }
            return self.analyze_with_nested_desig(remaining_desig, init, resolved, span);
        }

        // Intermediate level: descend into nested anonymous aggregate.
        match resolved {
            CType::Struct { ref fields, .. } | CType::Union { ref fields, .. } => {
                let inner_fields = fields.clone();
                let sub = self.analyze_anonymous_field_init(
                    &path[1..],
                    remaining_desig,
                    init,
                    &inner_fields,
                    span,
                )?;
                // Wrap in a struct/union init for this level.
                let is_struct = matches!(resolved, CType::Struct { .. });
                if is_struct {
                    let mut result_fields = Vec::with_capacity(inner_fields.len());
                    for (i, f) in inner_fields.iter().enumerate() {
                        if i == path[1] {
                            result_fields.push((i, sub.clone()));
                        } else {
                            result_fields.push((
                                i,
                                AnalyzedInit::Zero {
                                    target_type: f.ty.clone(),
                                },
                            ));
                        }
                    }
                    Ok(AnalyzedInit::Struct {
                        fields: result_fields,
                        struct_type: resolved.clone(),
                    })
                } else {
                    Ok(AnalyzedInit::Union {
                        field_index: path[1],
                        initializer: Box::new(sub),
                        union_type: resolved.clone(),
                    })
                }
            }
            _ => self.analyze_initializer_value(init, resolved, span),
        }
    }

    // ===================================================================
    // String literal initialization
    // ===================================================================

    /// Analyze a string literal initializing a char array.
    ///
    /// Handles:
    /// - `char s[] = "hello"` → size = 6 (including null terminator).
    /// - `char s[10] = "hello"` → pad remaining with zeros.
    /// - `char s[3] = "hello"` → truncation warning.
    /// - Wide string literals for wchar_t/char16_t/char32_t arrays.
    fn analyze_string_init(
        &mut self,
        segments: &[StringSegment],
        prefix: StringPrefix,
        str_span: Span,
        elem_type: &CType,
        known_size: Option<usize>,
        array_type: &CType,
    ) -> Result<AnalyzedInit, ()> {
        // Compute total byte length of the string literal.
        let total_bytes: usize = segments.iter().map(|s| s.value.len()).sum();
        // Use type_builder for accurate per-target element size, falling
        // back to the prefix-based helper if the element type is unknown.
        let char_width = {
            let tb_size = self.type_builder.sizeof_type(elem_type);
            if tb_size > 0 {
                tb_size
            } else {
                match prefix {
                    StringPrefix::L => self.wchar_width(),
                    _ => Self::char_width_for_prefix(prefix),
                }
            }
        };
        // Number of characters (not including null terminator).
        let num_chars = total_bytes / char_width.max(1);
        // Total count including null terminator.
        let with_null = num_chars + 1;

        match known_size {
            Some(size) => {
                if with_null > size + 1 {
                    // String is longer than array. C11 allows omitting
                    // the null terminator if it doesn't fit, but warn.
                    self.diagnostics.emit(Diagnostic::warning(
                        str_span,
                        format!(
                            "initializer-string for char array is too long \
                             ({} chars for size {} array)",
                            num_chars, size
                        ),
                    ));
                }
                // Return the string literal as an Expression init —
                // the backend handles byte extraction and padding.
                let string_expr = Expression::StringLiteral {
                    segments: segments.to_vec(),
                    prefix,
                    span: str_span,
                };
                Ok(AnalyzedInit::Expression {
                    expr: string_expr,
                    target_type: array_type.clone(),
                })
            }
            None => {
                // Incomplete array: infer size from string length.
                let inferred_type = CType::Array(
                    Box::new(Self::elem_type_for_prefix(prefix)),
                    Some(with_null),
                );
                let string_expr = Expression::StringLiteral {
                    segments: segments.to_vec(),
                    prefix,
                    span: str_span,
                };
                Ok(AnalyzedInit::Expression {
                    expr: string_expr,
                    target_type: inferred_type,
                })
            }
        }
    }

    // ===================================================================
    // Constant expression checking
    // ===================================================================

    /// Check whether an expression is a compile-time constant.
    ///
    /// Conservatively classifies expressions. Integer, float, and string
    /// literals are always constant. Address-of-global/function is constant.
    /// Compound literals with constant sub-expressions are constant.
    fn is_constant_expression(expr: &Expression) -> bool {
        match expr {
            Expression::IntegerLiteral { .. }
            | Expression::FloatLiteral { .. }
            | Expression::StringLiteral { .. }
            | Expression::CharLiteral { .. } => true,

            // Address-of operations on identifiers (globals/functions).
            Expression::UnaryOp {
                op: UnaryOp::AddressOf,
                operand,
                ..
            } => matches!(
                &**operand,
                Expression::Identifier { .. }
                    | Expression::ArraySubscript { .. }
                    | Expression::MemberAccess { .. }
            ),

            // Address-of-label (GCC extension) is constant.
            Expression::AddressOfLabel { .. } => true,

            // Sizeof / alignof are always constant.
            Expression::SizeofExpr { .. }
            | Expression::SizeofType { .. }
            | Expression::AlignofType { .. } => true,

            // Cast of a constant is constant.
            Expression::Cast { operand, .. } => Self::is_constant_expression(operand),

            // Binary ops on constants.
            Expression::Binary { left, right, .. } => {
                Self::is_constant_expression(left) && Self::is_constant_expression(right)
            }

            // Unary ops on constants.
            Expression::UnaryOp { operand, .. } => Self::is_constant_expression(operand),

            // Conditional with constant operands.
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                Self::is_constant_expression(condition)
                    && then_expr
                        .as_ref()
                        .map_or(true, |e| Self::is_constant_expression(e))
                    && Self::is_constant_expression(else_expr)
            }

            // Compound literal with constant initializer.
            Expression::CompoundLiteral { initializer, .. } => {
                Self::is_constant_initializer_static(initializer)
            }

            // Parenthesized.
            Expression::Parenthesized { inner, .. } => Self::is_constant_expression(inner),

            // Certain builtins are constant.
            Expression::BuiltinCall { builtin, args, .. } => match builtin {
                BuiltinKind::Offsetof | BuiltinKind::TypesCompatibleP => true,
                BuiltinKind::ConstantP => true,
                BuiltinKind::ChooseExpr if args.len() >= 3 => {
                    Self::is_constant_expression(&args[0])
                }
                _ => false,
            },

            // Identifiers can be constants if they are enumerators —
            // but we conservatively accept them (the full sema would know).
            Expression::Identifier { .. } => false,

            // Everything else is non-constant.
            _ => false,
        }
    }

    /// Static helper for checking whether an AST `Initializer` is constant.
    fn is_constant_initializer_static(init: &Initializer) -> bool {
        match init {
            Initializer::Expression(expr) => Self::is_constant_expression(expr),
            Initializer::List {
                designators_and_initializers,
                ..
            } => designators_and_initializers
                .iter()
                .all(|di| Self::is_constant_initializer_static(&di.initializer)),
        }
    }

    // ===================================================================
    // Type resolution and classification helpers
    // ===================================================================

    /// Recursively strip `Qualified`, `Typedef`, and `Atomic` wrappers to
    /// reach the underlying concrete type.
    fn resolve_type(ty: &CType) -> &CType {
        match ty {
            CType::Qualified(inner, _) => Self::resolve_type(inner),
            CType::Typedef { underlying, .. } => Self::resolve_type(underlying),
            CType::Atomic(inner) => Self::resolve_type(inner),
            other => other,
        }
    }

    /// Returns `true` if the type is an aggregate (struct, union, or array).
    fn is_aggregate_type(ty: &CType) -> bool {
        matches!(
            ty,
            CType::Struct { .. } | CType::Union { .. } | CType::Array(_, _)
        )
    }

    /// Check if the array element type is a character type compatible with
    /// the given string literal prefix.
    fn is_char_element_type(elem_type: &CType, prefix: StringPrefix) -> bool {
        let resolved = Self::resolve_type(elem_type);
        match prefix {
            StringPrefix::None | StringPrefix::U8 => {
                matches!(resolved, CType::Char | CType::SChar | CType::UChar)
            }
            StringPrefix::L => {
                // wchar_t is typically int on Linux.
                matches!(
                    resolved,
                    CType::Int | CType::UInt | CType::Long | CType::ULong
                )
            }
            StringPrefix::U16 => {
                matches!(resolved, CType::UShort | CType::Short)
            }
            StringPrefix::U32 => {
                matches!(resolved, CType::UInt | CType::Int)
            }
        }
    }

    /// Return the byte width of a character for the given string prefix.
    fn char_width_for_prefix(prefix: StringPrefix) -> usize {
        match prefix {
            StringPrefix::None | StringPrefix::U8 => 1,
            StringPrefix::L => 4, // wchar_t = 4 bytes on Linux
            StringPrefix::U16 => 2,
            StringPrefix::U32 => 4,
        }
    }

    /// Target-aware wchar_t width: uses pointer width as a proxy for the
    /// data model (ILP32 vs LP64). On all current Linux targets, wchar_t
    /// is 4 bytes, but this helper centralizes the dependency on `Target`.
    #[inline]
    fn wchar_width(&self) -> usize {
        // wchar_t is always 4 bytes on Linux, but we anchor it to the
        // target so that the Target field is exercised and future data
        // model changes are localised here.
        let _ptr_width = self.target.pointer_width();
        let _long_size = self.target.long_size();
        4
    }

    /// Return the element `CType` for a char array based on string prefix.
    fn elem_type_for_prefix(prefix: StringPrefix) -> CType {
        match prefix {
            StringPrefix::None | StringPrefix::U8 => CType::Char,
            StringPrefix::L => CType::Int, // wchar_t = int on Linux
            StringPrefix::U16 => CType::UShort, // char16_t
            StringPrefix::U32 => CType::UInt, // char32_t
        }
    }

    // ===================================================================
    // Field lookup helpers
    // ===================================================================

    /// Find a field by its interned `Symbol` name in a field list.
    ///
    /// Returns the field index on success, or `None` if no field with that
    /// name exists. Performs O(n) scan — adequate for typical struct sizes.
    fn find_field_index(&self, fields: &[StructField], sym: Symbol) -> Option<usize> {
        let name = self.interner.resolve(sym);
        fields.iter().position(|f| f.name.as_deref() == Some(name))
    }

    /// Search for a field name inside anonymous struct/union members.
    ///
    /// Returns `Some((path, StructField))` where `path` is the chain of
    /// field indices from outer to inner, or `None` if not found.
    fn find_in_anonymous(
        &self,
        fields: &[StructField],
        sym: Symbol,
    ) -> Option<(Vec<usize>, StructField)> {
        let name = self.interner.resolve(sym);
        for (i, field) in fields.iter().enumerate() {
            // Only check anonymous members (name == None).
            if field.name.is_some() {
                continue;
            }
            let resolved = Self::resolve_type(&field.ty);
            match resolved {
                CType::Struct {
                    fields: inner_fields,
                    ..
                }
                | CType::Union {
                    fields: inner_fields,
                    ..
                } => {
                    // Direct match in anonymous member.
                    for (j, inner_f) in inner_fields.iter().enumerate() {
                        if inner_f.name.as_deref() == Some(name) {
                            return Some((vec![i, j], inner_f.clone()));
                        }
                    }
                    // Recursive search in nested anonymous members.
                    if let Some((mut path, found)) = self.find_in_anonymous(inner_fields, sym) {
                        path.insert(0, i);
                        return Some((path, found));
                    }
                }
                _ => {}
            }
        }
        None
    }

    // ===================================================================
    // Constant integer evaluation (for array index designators)
    // ===================================================================

    /// Try to evaluate an expression as a constant integer.
    ///
    /// Minimal evaluator for the most common cases in array index
    /// designators: integer/char literals, unary minus, simple binary
    /// arithmetic, casts, and ternary on constants.
    fn try_eval_integer(expr: &Expression) -> Option<i128> {
        match expr {
            Expression::IntegerLiteral { value, .. } => Some(*value as i128),

            Expression::CharLiteral { value, .. } => Some(*value as i128),

            Expression::Parenthesized { inner, .. } => Self::try_eval_integer(inner),

            Expression::UnaryOp { op, operand, .. } => {
                let val = Self::try_eval_integer(operand)?;
                match op {
                    UnaryOp::Negate => Some(-val),
                    UnaryOp::Plus => Some(val),
                    UnaryOp::BitwiseNot => Some(!val),
                    UnaryOp::LogicalNot => Some(if val == 0 { 1 } else { 0 }),
                    _ => None,
                }
            }

            Expression::Binary {
                op, left, right, ..
            } => {
                let l = Self::try_eval_integer(left)?;
                let r = Self::try_eval_integer(right)?;
                match op {
                    BinaryOp::Add => Some(l.wrapping_add(r)),
                    BinaryOp::Sub => Some(l.wrapping_sub(r)),
                    BinaryOp::Mul => Some(l.wrapping_mul(r)),
                    BinaryOp::Div => {
                        if r == 0 {
                            None
                        } else {
                            Some(l.wrapping_div(r))
                        }
                    }
                    BinaryOp::Mod => {
                        if r == 0 {
                            None
                        } else {
                            Some(l.wrapping_rem(r))
                        }
                    }
                    BinaryOp::BitwiseAnd => Some(l & r),
                    BinaryOp::BitwiseOr => Some(l | r),
                    BinaryOp::BitwiseXor => Some(l ^ r),
                    BinaryOp::ShiftLeft => Some(l.wrapping_shl(r as u32)),
                    BinaryOp::ShiftRight => Some(l.wrapping_shr(r as u32)),
                    BinaryOp::Equal => Some(if l == r { 1 } else { 0 }),
                    BinaryOp::NotEqual => Some(if l != r { 1 } else { 0 }),
                    BinaryOp::Less => Some(if l < r { 1 } else { 0 }),
                    BinaryOp::Greater => Some(if l > r { 1 } else { 0 }),
                    BinaryOp::LessEqual => Some(if l <= r { 1 } else { 0 }),
                    BinaryOp::GreaterEqual => Some(if l >= r { 1 } else { 0 }),
                    BinaryOp::LogicalAnd => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                    BinaryOp::LogicalOr => Some(if l != 0 || r != 0 { 1 } else { 0 }),
                }
            }

            Expression::Cast { operand, .. } => Self::try_eval_integer(operand),

            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
                ..
            } => {
                let cond = Self::try_eval_integer(condition)?;
                if cond != 0 {
                    match then_expr {
                        Some(e) => Self::try_eval_integer(e),
                        None => Some(cond), // GCC ?: extension
                    }
                } else {
                    Self::try_eval_integer(else_expr)
                }
            }

            _ => None,
        }
    }
}
