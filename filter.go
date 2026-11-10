package squirrel

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"regexp"
)

// FilterOp identifies a node in a finite filter tree.
type FilterOp string

// Logical and leaf operators accepted by CompileFilter.
//
// The string values are part of the public contract and are the only accepted
// encodings: an unknown value is rejected as an unknown node and never reaches
// the generated SQL.
const (
	OpAnd       FilterOp = "and"
	OpOr        FilterOp = "or"
	OpNot       FilterOp = "not"
	OpEq        FilterOp = "eq"
	OpNeq       FilterOp = "neq"
	OpLt        FilterOp = "lt"
	OpLte       FilterOp = "lte"
	OpGt        FilterOp = "gt"
	OpGte       FilterOp = "gte"
	OpIn        FilterOp = "in"
	OpNotIn     FilterOp = "not_in"
	OpIsNull    FilterOp = "is_null"
	OpIsNotNull FilterOp = "is_not_null"
)

// FieldType is the scalar type declared for a registered field. It decides how
// external values are coerced before they become bound args.
type FieldType int

const (
	// FieldTypeInt coerces to int64. Floating point input is accepted only when
	// it is finite, integral and inside the int64 range; strings and other
	// pseudo-numbers are never parsed.
	FieldTypeInt FieldType = iota + 1
	// FieldTypeFloat coerces to float64. NaN and Inf are rejected.
	FieldTypeFloat
	// FieldTypeString accepts strings only.
	FieldTypeString
	// FieldTypeBool accepts booleans only.
	FieldTypeBool
	// FieldTypeBytes accepts []byte. Whether a []byte is a scalar or an IN list
	// is decided by FieldSchema.BytesAsList, never by reflection.
	FieldTypeBytes
)

// Filter is a node of the external, finite filter tree. It is designed to be
// decodable directly from an API request body (for example with encoding/json).
//
// A node uses exactly one shape:
//
//   - And/Or:   Children holds one or more child nodes.
//   - Not:      Children holds exactly one child node.
//   - scalar:   Field plus Value (eq, neq, lt, lte, gt, gte).
//   - IN:       Field plus Values (in, not_in); Values may be empty.
//   - NULL:     Field only (is_null, is_not_null).
//
// Filter carries no raw SQL and no opaque expression: strings are only ever
// used as map keys against the pre-registered schema, values only ever become
// bound placeholder args.
type Filter struct {
	Op       FilterOp      `json:"op"`
	Field    string        `json:"field,omitempty"`
	Value    interface{}   `json:"value,omitempty"`
	Values   []interface{} `json:"values,omitempty"`
	Children []*Filter     `json:"children,omitempty"`
}

// FieldSchema declares one externally visible field. The declaration is owned
// by trusted server code; external callers only reference the map key used at
// registration time.
type FieldSchema struct {
	// Column is the pre-registered SQL column identity. It must be a simple
	// identifier or a dotted qualified identifier (e.g. "u.id") and is validated
	// at schema construction. External field names are never substituted into
	// SQL in its place.
	Column string
	// Type is the scalar value type used for coercion.
	Type FieldType
	// Ops is the explicit allow-list of leaf operators for this field. A nil
	// slice selects a type-specific default set.
	Ops []FilterOp
	// Nullable permits is_null/is_not_null and NULL elements inside IN lists.
	Nullable bool
	// MaxIN bounds the number of elements accepted by in/not_in. It is required
	// to be positive whenever an IN operator is allowed; zero selects the
	// package default.
	MaxIN int
	// Sensitive marks values as sensitive. Compile errors never embed values
	// regardless of this flag; sensitive fields additionally guarantee that no
	// value-derived text appears in any error message.
	Sensitive bool
	// BytesAsList is only meaningful for FieldTypeBytes. When true, a []byte
	// supplied to in/not_in expands into one integer arg per byte. When false
	// (the default) []byte is always a single scalar blob arg.
	BytesAsList bool
}

// FilterBudget bounds a single compilation. A zero value selects the package
// defaults. All limits are enforced against the actually expanded tree, args
// and SQL bytes.
type FilterBudget struct {
	MaxDepth    int
	MaxNodes    int
	MaxArgs     int
	MaxSQLBytes int
}

// Default compilation budgets.
const (
	DefaultFilterMaxDepth    = 16
	DefaultFilterMaxNodes    = 256
	DefaultFilterMaxArgs     = 1024
	DefaultFilterMaxSQLBytes = 64 * 1024
	DefaultFilterMaxIN       = 256
)

// FilterErrorCode classifies a FilterCompileError for programmatic handling.
type FilterErrorCode string

const (
	FilterErrorSchema        FilterErrorCode = "schema_invalid"
	FilterErrorMalformedNode FilterErrorCode = "malformed_node"
	FilterErrorCycle         FilterErrorCode = "cycle_detected"
	FilterErrorUnknownField  FilterErrorCode = "unknown_field"
	FilterErrorUnknownOp     FilterErrorCode = "unknown_op"
	FilterErrorOpNotAllowed  FilterErrorCode = "op_not_allowed"
	FilterErrorNotNullable   FilterErrorCode = "not_nullable"
	FilterErrorInvalidValue  FilterErrorCode = "invalid_value"
	FilterErrorOverflow      FilterErrorCode = "integer_overflow"
	FilterErrorBudget        FilterErrorCode = "budget_exceeded"
)

// Sentinel errors matched with errors.Is.
var (
	ErrFilterSchema        = errors.New("squirrel: invalid filter schema")
	ErrFilterMalformedNode = errors.New("squirrel: malformed filter node")
	ErrFilterCycle         = errors.New("squirrel: filter tree contains a cycle or a reused node")
	ErrFilterUnknownField  = errors.New("squirrel: unknown filter field")
	ErrFilterUnknownOp     = errors.New("squirrel: unknown filter operator")
	ErrFilterOpNotAllowed  = errors.New("squirrel: filter operator not allowed for field")
	ErrFilterNotNullable   = errors.New("squirrel: field is not nullable")
	ErrFilterInvalidValue  = errors.New("squirrel: invalid filter value")
	ErrFilterOverflow      = errors.New("squirrel: integer value overflows int64")
	ErrFilterBudget        = errors.New("squirrel: filter budget exceeded")
)

func codeSentinel(code FilterErrorCode) error {
	switch code {
	case FilterErrorSchema:
		return ErrFilterSchema
	case FilterErrorMalformedNode:
		return ErrFilterMalformedNode
	case FilterErrorCycle:
		return ErrFilterCycle
	case FilterErrorUnknownField:
		return ErrFilterUnknownField
	case FilterErrorUnknownOp:
		return ErrFilterUnknownOp
	case FilterErrorOpNotAllowed:
		return ErrFilterOpNotAllowed
	case FilterErrorNotNullable:
		return ErrFilterNotNullable
	case FilterErrorInvalidValue:
		return ErrFilterInvalidValue
	case FilterErrorOverflow:
		return ErrFilterOverflow
	case FilterErrorBudget:
		return ErrFilterBudget
	default:
		return nil
	}
}

// FilterCompileError is returned for every rejected compilation. It carries the
// tree path and the external field name, but never the supplied value.
type FilterCompileError struct {
	Code   FilterErrorCode
	Op     FilterOp
	Field  string
	Path   string
	reason string
	wrap   error
}

func (e *FilterCompileError) Error() string {
	var b bytes.Buffer
	b.WriteString("squirrel: filter compile failed")
	if e.Path != "" {
		fmt.Fprintf(&b, " at %s", e.Path)
	}
	if e.Field != "" {
		fmt.Fprintf(&b, " (field %q)", e.Field)
	}
	if e.Op != "" {
		fmt.Fprintf(&b, " [op %s]", e.Op)
	}
	if e.reason != "" {
		b.WriteString(": ")
		b.WriteString(e.reason)
	}
	return b.String()
}

// Unwrap exposes the class sentinel so errors.Is works.
func (e *FilterCompileError) Unwrap() error {
	if e.wrap != nil {
		return e.wrap
	}
	return codeSentinel(e.Code)
}

// OrderClause is one entry of a safe ORDER BY compilation.
type OrderClause struct {
	Field string
	Desc  bool
}

// FilterSchema is a frozen, validated field registry. Instances are created
// only with NewFilterSchema and are safe for concurrent reuse: compilation
// never mutates the schema and keeps no per-call state on it.
type FilterSchema struct {
	fields map[string]filterField
	budget FilterBudget
}

type filterField struct {
	column      string
	typ         FieldType
	ops         map[FilterOp]struct{}
	nullable    bool
	maxIN       int
	sensitive   bool
	bytesAsList bool
}

// safeColumnPattern allows plain or dotted identifiers only. No quoting,
// whitespace, comments or statement separators can pass registration.
var safeColumnPattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$`)

// NewFilterSchema validates and freezes a field registry. The input map and
// operator slices are copied defensively; later mutation of the caller's data
// does not affect the returned schema.
func NewFilterSchema(decls map[string]FieldSchema, budget FilterBudget) (*FilterSchema, error) {
	b, err := normalizeBudget(budget)
	if err != nil {
		return nil, err
	}

	if len(decls) == 0 {
		return nil, &FilterCompileError{Code: FilterErrorSchema, reason: "schema declares no fields"}
	}

	frozen := make(map[string]filterField, len(decls))
	for externalName, decl := range decls {
		if externalName == "" {
			return nil, &FilterCompileError{Code: FilterErrorSchema, reason: "field name must not be empty"}
		}
		if !safeColumnPattern.MatchString(decl.Column) {
			return nil, &FilterCompileError{
				Code:   FilterErrorSchema,
				Field:  externalName,
				reason: fmt.Sprintf("registered column %q is not a safe dotted identifier", decl.Column),
			}
		}
		if decl.Type < FieldTypeInt || decl.Type > FieldTypeBytes {
			return nil, &FilterCompileError{Code: FilterErrorSchema, Field: externalName, reason: "unknown field type"}
		}
		if decl.BytesAsList && decl.Type != FieldTypeBytes {
			return nil, &FilterCompileError{Code: FilterErrorSchema, Field: externalName, reason: "BytesAsList requires FieldTypeBytes"}
		}
		if decl.MaxIN < 0 {
			return nil, &FilterCompileError{Code: FilterErrorSchema, Field: externalName, reason: "MaxIN must not be negative"}
		}

		ops, err := normalizeOps(decl)
		if err != nil {
			return nil, &FilterCompileError{Code: FilterErrorSchema, Field: externalName, reason: err.Error()}
		}

		maxIN := 0
		if _, ok := ops[OpIn]; ok {
			maxIN = decl.MaxIN
			if maxIN == 0 {
				maxIN = DefaultFilterMaxIN
			}
			// MaxIN and MaxArgs stay independent limits; the per-compilation
			// arg budget check enforces the global expansion bound.
		}

		frozen[externalName] = filterField{
			column:      decl.Column,
			typ:         decl.Type,
			ops:         ops,
			nullable:    decl.Nullable,
			maxIN:       maxIN,
			sensitive:   decl.Sensitive,
			bytesAsList: decl.BytesAsList,
		}
	}

	return &FilterSchema{fields: frozen, budget: b}, nil
}

func normalizeBudget(b FilterBudget) (FilterBudget, error) {
	out := b
	fill := func(v int, def int) (int, bool) {
		if v == 0 {
			return def, true
		}
		return v, v > 0
	}
	var ok bool
	if out.MaxDepth, ok = fill(b.MaxDepth, DefaultFilterMaxDepth); !ok {
		return out, &FilterCompileError{Code: FilterErrorSchema, reason: "MaxDepth must be positive"}
	}
	if out.MaxNodes, ok = fill(b.MaxNodes, DefaultFilterMaxNodes); !ok {
		return out, &FilterCompileError{Code: FilterErrorSchema, reason: "MaxNodes must be positive"}
	}
	if out.MaxArgs, ok = fill(b.MaxArgs, DefaultFilterMaxArgs); !ok {
		return out, &FilterCompileError{Code: FilterErrorSchema, reason: "MaxArgs must be positive"}
	}
	if out.MaxSQLBytes, ok = fill(b.MaxSQLBytes, DefaultFilterMaxSQLBytes); !ok {
		return out, &FilterCompileError{Code: FilterErrorSchema, reason: "MaxSQLBytes must be positive"}
	}
	return out, nil
}

func defaultOps(t FieldType, nullable bool) []FilterOp {
	ops := []FilterOp{OpEq, OpNeq, OpLt, OpLte, OpGt, OpGte}
	if t == FieldTypeString || t == FieldTypeBool || t == FieldTypeBytes {
		ops = []FilterOp{OpEq, OpNeq}
	}
	ops = append(ops, OpIn, OpNotIn)
	if nullable {
		ops = append(ops, OpIsNull, OpIsNotNull)
	}
	return ops
}

func normalizeOps(decl FieldSchema) (map[FilterOp]struct{}, error) {
	declared := decl.Ops
	if declared == nil {
		declared = defaultOps(decl.Type, decl.Nullable)
	}
	if len(declared) == 0 {
		return nil, errors.New("operator allow-list must not be empty")
	}
	ops := make(map[FilterOp]struct{}, len(declared))
	for _, op := range declared {
		if !isLeafOp(op) {
			return nil, fmt.Errorf("%q is not a leaf operator", op)
		}
		if !decl.Nullable && (op == OpIsNull || op == OpIsNotNull) {
			return nil, fmt.Errorf("operator %q requires a nullable field", op)
		}
		ops[op] = struct{}{}
	}
	return ops, nil
}

func isLeafOp(op FilterOp) bool {
	switch op {
	case OpEq, OpNeq, OpLt, OpLte, OpGt, OpGte, OpIn, OpNotIn, OpIsNull, OpIsNotNull:
		return true
	}
	return false
}

func isLogicalOp(op FilterOp) bool {
	return op == OpAnd || op == OpOr || op == OpNot
}

func leafSQLToken(op FilterOp) (string, bool) {
	switch op {
	case OpEq:
		return "=", true
	case OpNeq:
		return "<>", true
	case OpLt:
		return "<", true
	case OpLte:
		return "<=", true
	case OpGt:
		return ">", true
	case OpGte:
		return ">=", true
	default:
		return "", false
	}
}

// CompileFilter compiles an external finite filter tree against the frozen
// schema into a self-contained Sqlizer.
//
// On success the returned Sqlizer holds its own copied SQL and args, keeps the
// original tree traversal order for args and can be reused with any ordinary
// SelectBuilder (e.g. via Where). On failure it returns a nil Sqlizer together
// with a *FilterCompileError: no partially executable expression is ever
// returned.
func CompileFilter(schema *FilterSchema, root *Filter) (Sqlizer, error) {
	if schema == nil {
		return nil, &FilterCompileError{Code: FilterErrorSchema, Path: "$", reason: "nil schema"}
	}
	if root == nil {
		return nil, &FilterCompileError{Code: FilterErrorMalformedNode, Path: "$", reason: "nil filter node"}
	}
	c := &filterCompiler{
		schema: schema,
		budget: schema.budget,
		seen:   make(map[*Filter]struct{}),
	}
	if err := c.compileNode(root, "$"); err != nil {
		return nil, err
	}
	sqlStr := c.sql.String()
	if len(sqlStr) > c.budget.MaxSQLBytes {
		return nil, c.budgetError(root, "$", fmt.Sprintf("generated SQL is %d bytes, limit is %d", len(sqlStr), c.budget.MaxSQLBytes))
	}
	return &compiledFilter{sql: sqlStr, args: c.args}, nil
}

// CompileFilter is the method form of the package-level entry point.
func (s *FilterSchema) CompileFilter(root *Filter) (Sqlizer, error) {
	return CompileFilter(s, root)
}

// CompileOrderBy compiles a safe ORDER BY list. Only registered field names
// are accepted; direction is a fixed ASC/DESC token, so sort text supplied by
// API callers can never inject SQL structure.
func (s *FilterSchema) CompileOrderBy(clauses []OrderClause) (Sqlizer, error) {
	if s == nil {
		return nil, &FilterCompileError{Code: FilterErrorSchema, Path: "$", reason: "nil schema"}
	}
	if len(clauses) == 0 {
		return nil, &FilterCompileError{Code: FilterErrorMalformedNode, Path: "$", reason: "empty order by list"}
	}
	if len(clauses) > s.budget.MaxNodes {
		return nil, &FilterCompileError{
			Code:   FilterErrorBudget,
			Path:   "$",
			reason: fmt.Sprintf("order by has %d entries, node limit is %d", len(clauses), s.budget.MaxNodes),
		}
	}

	parts := make([]string, 0, len(clauses))
	for i, clause := range clauses {
		field, ok := s.fields[clause.Field]
		if !ok {
			return nil, &FilterCompileError{
				Code:   FilterErrorUnknownField,
				Field:  clause.Field,
				Path:   fmt.Sprintf("$.order_by[%d]", i),
				reason: "field is not registered",
			}
		}
		direction := "ASC"
		if clause.Desc {
			direction = "DESC"
		}
		parts = append(parts, field.column+" "+direction)
	}
	sqlStr := joinStrings(parts, ", ")
	if len(sqlStr) > s.budget.MaxSQLBytes {
		return nil, &FilterCompileError{
			Code:   FilterErrorBudget,
			Path:   "$",
			reason: "order by SQL exceeds byte budget",
		}
	}
	return &compiledFilter{sql: sqlStr, args: []interface{}{}}, nil
}

func joinStrings(parts []string, sep string) string {
	var b bytes.Buffer
	for i, p := range parts {
		if i > 0 {
			b.WriteString(sep)
		}
		b.WriteString(p)
	}
	return b.String()
}

type filterCompiler struct {
	schema *FilterSchema
	budget FilterBudget
	seen   map[*Filter]struct{}
	nodes  int
	depth  int
	args   []interface{}
	sql    bytes.Buffer
}

func (c *filterCompiler) fail(node *Filter, path string, code FilterErrorCode, reason string) error {
	return &FilterCompileError{
		Code:   code,
		Op:     node.Op,
		Field:  node.Field,
		Path:   path,
		reason: reason,
	}
}

func (c *filterCompiler) budgetError(node *Filter, path string, reason string) error {
	return c.fail(node, path, FilterErrorBudget, reason)
}

func (c *filterCompiler) write(node *Filter, path, fragment string) error {
	c.sql.WriteString(fragment)
	if c.sql.Len() > c.budget.MaxSQLBytes {
		return c.budgetError(node, path, "generated SQL exceeds byte budget")
	}
	return nil
}

func (c *filterCompiler) addArgs(node *Filter, path string, extra []interface{}) error {
	if len(c.args)+len(extra) > c.budget.MaxArgs {
		return c.budgetError(node, path, fmt.Sprintf("expanded arg count would exceed limit of %d", c.budget.MaxArgs))
	}
	c.args = append(c.args, extra...)
	return nil
}

func (c *filterCompiler) compileNode(node *Filter, path string) error {
	if node == nil {
		return &FilterCompileError{Code: FilterErrorMalformedNode, Path: path, reason: "nil filter node"}
	}
	if _, seen := c.seen[node]; seen {
		return &FilterCompileError{
			Code:   FilterErrorCycle,
			Op:     node.Op,
			Field:  node.Field,
			Path:   path,
			reason: "node pointer was already visited; trees must not share nodes or contain cycles",
		}
	}
	c.seen[node] = struct{}{}

	c.nodes++
	if c.nodes > c.budget.MaxNodes {
		return c.budgetError(node, path, fmt.Sprintf("node count %d exceeds limit of %d", c.nodes, c.budget.MaxNodes))
	}
	c.depth++
	if c.depth > c.budget.MaxDepth {
		err := c.budgetError(node, path, fmt.Sprintf("depth %d exceeds limit of %d", c.depth, c.budget.MaxDepth))
		c.depth--
		return err
	}
	defer func() { c.depth-- }()

	switch {
	case node.Op == OpAnd || node.Op == OpOr:
		return c.compileLogical(node, path)
	case node.Op == OpNot:
		return c.compileNot(node, path)
	case isLeafOp(node.Op):
		return c.compileLeaf(node, path)
	default:
		return c.fail(node, path, FilterErrorUnknownOp, fmt.Sprintf("operator %q is not recognized", node.Op))
	}
}

func (c *filterCompiler) compileLogical(node *Filter, path string) error {
	if node.Field != "" || node.Value != nil || len(node.Values) > 0 {
		return c.fail(node, path, FilterErrorMalformedNode, "logical nodes only accept Children")
	}
	if len(node.Children) == 0 {
		return c.fail(node, path, FilterErrorMalformedNode, "logical nodes require at least one child")
	}
	sep := " AND "
	if node.Op == OpOr {
		sep = " OR "
	}
	if err := c.write(node, path, "("); err != nil {
		return err
	}
	for i, child := range node.Children {
		if i > 0 {
			if err := c.write(node, path, sep); err != nil {
				return err
			}
		}
		if err := c.compileNode(child, fmt.Sprintf("%s.children[%d]", path, i)); err != nil {
			return err
		}
	}
	return c.write(node, path, ")")
}

func (c *filterCompiler) compileNot(node *Filter, path string) error {
	if node.Field != "" || node.Value != nil || len(node.Values) > 0 {
		return c.fail(node, path, FilterErrorMalformedNode, "not nodes only accept one Child")
	}
	if len(node.Children) != 1 {
		return c.fail(node, path, FilterErrorMalformedNode, "not requires exactly one child")
	}
	// Structural NOT (...) preserves SQL three-valued logic: NOT UNKNOWN stays
	// UNKNOWN, it is never collapsed into a binary TRUE.
	if err := c.write(node, path, "NOT ("); err != nil {
		return err
	}
	if err := c.compileNode(node.Children[0], path+".children[0]"); err != nil {
		return err
	}
	return c.write(node, path, ")")
}

func (c *filterCompiler) compileLeaf(node *Filter, path string) error {
	if node.Field == "" {
		return c.fail(node, path, FilterErrorMalformedNode, "leaf node requires a field")
	}
	if len(node.Children) > 0 {
		return c.fail(node, path, FilterErrorMalformedNode, "leaf nodes must not carry children")
	}
	field, ok := c.schema.fields[node.Field]
	if !ok {
		return c.fail(node, path, FilterErrorUnknownField, "field is not registered")
	}
	if _, allowed := field.ops[node.Op]; !allowed {
		return c.fail(node, path, FilterErrorOpNotAllowed, fmt.Sprintf("operator %s is not allowed for this field", node.Op))
	}

	switch node.Op {
	case OpIsNull, OpIsNotNull:
		return c.compileNullCheck(node, path, field)
	case OpIn, OpNotIn:
		return c.compileIn(node, path, field)
	default:
		return c.compileScalar(node, path, field)
	}
}

func (c *filterCompiler) compileNullCheck(node *Filter, path string, field filterField) error {
	if node.Value != nil || len(node.Values) > 0 {
		return c.fail(node, path, FilterErrorMalformedNode, "null checks do not accept values")
	}
	if !field.nullable {
		return c.fail(node, path, FilterErrorNotNullable, "null checks require a nullable field")
	}
	fragment := field.column + " IS NULL"
	if node.Op == OpIsNotNull {
		fragment = field.column + " IS NOT NULL"
	}
	return c.write(node, path, fragment)
}

func (c *filterCompiler) compileScalar(node *Filter, path string, field filterField) error {
	if len(node.Values) > 0 {
		return c.fail(node, path, FilterErrorMalformedNode, "scalar operators use Value, not Values")
	}
	if node.Value == nil {
		return c.fail(node, path, FilterErrorInvalidValue, "nil is not a scalar value; use is_null/is_not_null")
	}
	arg, err := coerceScalar(node.Field, field, node.Value, path)
	if err != nil {
		return err
	}
	token, _ := leafSQLToken(node.Op)
	if err := c.write(node, path, field.column+" "+token+" ?"); err != nil {
		return err
	}
	return c.addArgs(node, path, []interface{}{arg})
}

func (c *filterCompiler) compileIn(node *Filter, path string, field filterField) error {
	if field.maxIN == 0 {
		return c.fail(node, path, FilterErrorOpNotAllowed, "IN operators are not allowed for this field")
	}

	// BytesAsList: a single []byte in Value expands into per-byte integer args.
	// This expansion is an explicit schema decision, not reflection heuristics.
	if field.bytesAsList && node.Value != nil {
		if len(node.Values) > 0 {
			return c.fail(node, path, FilterErrorMalformedNode, "use either Value ([]byte) or Values, not both")
		}
		raw, ok := node.Value.([]byte)
		if !ok {
			return c.fail(node, path, FilterErrorInvalidValue, "BytesAsList fields expect a []byte Value for IN")
		}
		if len(raw) > field.maxIN {
			return c.budgetError(node, path, fmt.Sprintf("byte list length %d exceeds MaxIN %d", len(raw), field.maxIN))
		}
		expanded := make([]interface{}, len(raw))
		for i, b := range raw {
			expanded[i] = int64(b)
		}
		return c.emitIn(node, path, field, expanded)
	}

	if node.Value != nil {
		return c.fail(node, path, FilterErrorMalformedNode, "IN operators use Values")
	}
	if len(node.Values) > field.maxIN {
		return c.budgetError(node, path, fmt.Sprintf("IN list length %d exceeds MaxIN %d", len(node.Values), field.maxIN))
	}

	args := make([]interface{}, 0, len(node.Values))
	for i, raw := range node.Values {
		elemPath := fmt.Sprintf("%s.values[%d]", path, i)
		if raw == nil {
			if !field.nullable {
				return c.fail(node, elemPath, FilterErrorNotNullable, "NULL elements require a nullable field")
			}
			args = append(args, nil)
			continue
		}
		arg, err := coerceScalar(node.Field, field, raw, elemPath)
		if err != nil {
			return err
		}
		args = append(args, arg)
	}
	return c.emitIn(node, path, field, args)
}

func (c *filterCompiler) emitIn(node *Filter, path string, field filterField, args []interface{}) error {
	// Fixed empty-set semantics under SQL three-valued logic:
	// "col IN ()" is FALSE and "col NOT IN ()" is TRUE, including for NULL cols,
	// so they compile to the same portable constants squirrel uses elsewhere.
	if len(args) == 0 {
		if node.Op == OpIn {
			return c.write(node, path, sqlFalse)
		}
		return c.write(node, path, sqlTrue)
	}
	if err := c.addArgs(node, path, args); err != nil {
		return err
	}
	op := "IN"
	if node.Op == OpNotIn {
		op = "NOT IN"
	}
	return c.write(node, path, fmt.Sprintf("%s %s (%s)", field.column, op, Placeholders(len(args))))
}

// coerceScalar converts one external value into a bound arg according to the
// declared field type. It never returns SQL fragments: Sqlizer values are
// rejected outright so opaque expressions cannot enter through the tree.
func coerceScalar(fieldName string, field filterField, raw interface{}, path string) (interface{}, error) {
	if _, ok := raw.(Sqlizer); ok {
		return nil, &FilterCompileError{
			Code:   FilterErrorInvalidValue,
			Field:  fieldName,
			Path:   path,
			reason: "SQL expressions are not accepted as filter values",
		}
	}

	rv := reflect.ValueOf(raw)
	switch rv.Kind() {
	case reflect.Ptr, reflect.Interface:
		// Covers typed nil pointers as well as non-nil pointers: the value
		// contract is closed, callers must unwrap or use is_null explicitly.
		return nil, &FilterCompileError{
			Code:   FilterErrorInvalidValue,
			Field:  fieldName,
			Path:   path,
			reason: "pointer and typed-nil values are not allowed; use is_null for SQL NULL",
		}
	case reflect.Slice, reflect.Array:
		if field.typ == FieldTypeBytes && !field.bytesAsList && rv.Kind() == reflect.Slice && rv.Type() == reflect.TypeOf([]byte{}) {
			blob := make([]byte, rv.Len())
			copy(blob, rv.Bytes())
			return blob, nil
		}
		return nil, &FilterCompileError{
			Code:   FilterErrorInvalidValue,
			Field:  fieldName,
			Path:   path,
			reason: "slice/array values are not allowed as scalars; use Values for IN",
		}
	}

	switch field.typ {
	case FieldTypeInt:
		return coerceInt(fieldName, raw, path)
	case FieldTypeFloat:
		return coerceFloat(fieldName, raw, path)
	case FieldTypeString:
		if s, ok := raw.(string); ok {
			return s, nil
		}
		return nil, invalidTypeError(fieldName, path, raw, "string")
	case FieldTypeBool:
		if b, ok := raw.(bool); ok {
			return b, nil
		}
		return nil, invalidTypeError(fieldName, path, raw, "bool")
	case FieldTypeBytes:
		if b, ok := raw.([]byte); ok {
			blob := make([]byte, len(b))
			copy(blob, b)
			return blob, nil
		}
		return nil, invalidTypeError(fieldName, path, raw, "[]byte")
	default:
		return nil, &FilterCompileError{Code: FilterErrorSchema, Field: fieldName, Path: path, reason: "field has no usable scalar type"}
	}
}

func invalidTypeError(fieldName, path string, raw interface{}, want string) error {
	return &FilterCompileError{
		Code:   FilterErrorInvalidValue,
		Field:  fieldName,
		Path:   path,
		reason: fmt.Sprintf("value of type %T is not assignable to %s field", raw, want),
	}
}

// float64 boundaries expressed explicitly: 2^63 is one past MaxInt64 and is
// the rounded float64 form of MaxInt64, so it is rejected to avoid silent
// precision loss; -2^63 is exactly MinInt64 and remains accepted.
const (
	filterInt64UpperBound = 9223372036854775808.0  // 2^63
	filterInt64LowerBound = -9223372036854775808.0 // -2^63 == math.MinInt64
)

func coerceInt(fieldName string, raw interface{}, path string) (interface{}, error) {
	switch n := raw.(type) {
	case int:
		return int64(n), nil
	case int8:
		return int64(n), nil
	case int16:
		return int64(n), nil
	case int32:
		return int64(n), nil
	case int64:
		return n, nil
	case uint:
		return uintToInt64(fieldName, uint64(n), path)
	case uint8:
		return int64(n), nil
	case uint16:
		return int64(n), nil
	case uint32:
		return int64(n), nil
	case uint64:
		return uintToInt64(fieldName, n, path)
	case uintptr:
		return uintToInt64(fieldName, uint64(n), path)
	case float32:
		return floatToInt64(fieldName, float64(n), path)
	case float64:
		return floatToInt64(fieldName, n, path)
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return i, nil
		}
		f, err := n.Float64()
		if err != nil {
			return nil, &FilterCompileError{Code: FilterErrorInvalidValue, Field: fieldName, Path: path, reason: "number literal is not parseable"}
		}
		return floatToInt64(fieldName, f, path)
	default:
		// strings (including string pseudo-numbers), bools, json.Number-less
		// text and all other types are rejected without parsing.
		return nil, invalidTypeError(fieldName, path, raw, "integer")
	}
}

func uintToInt64(fieldName string, n uint64, path string) (interface{}, error) {
	if n > math.MaxInt64 {
		return nil, &FilterCompileError{Code: FilterErrorOverflow, Field: fieldName, Path: path, reason: "unsigned integer exceeds int64 range"}
	}
	return int64(n), nil
}

func floatToInt64(fieldName string, f float64, path string) (interface{}, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil, &FilterCompileError{Code: FilterErrorInvalidValue, Field: fieldName, Path: path, reason: "non-finite number is not an integer"}
	}
	if f != math.Trunc(f) {
		return nil, &FilterCompileError{Code: FilterErrorInvalidValue, Field: fieldName, Path: path, reason: "fractional number is not an integer"}
	}
	if f >= filterInt64UpperBound || f < filterInt64LowerBound {
		return nil, &FilterCompileError{Code: FilterErrorOverflow, Field: fieldName, Path: path, reason: "number is outside the int64 range"}
	}
	return int64(f), nil
}

func coerceFloat(fieldName string, raw interface{}, path string) (interface{}, error) {
	switch n := raw.(type) {
	case float32:
		return finiteFloat(fieldName, float64(n), path)
	case float64:
		return finiteFloat(fieldName, n, path)
	case int:
		return float64(n), nil
	case int8:
		return float64(n), nil
	case int16:
		return float64(n), nil
	case int32:
		return float64(n), nil
	case int64:
		return float64(n), nil
	case uint:
		return float64(n), nil
	case uint8:
		return float64(n), nil
	case uint16:
		return float64(n), nil
	case uint32:
		return float64(n), nil
	case uint64:
		return float64(n), nil
	case uintptr:
		return float64(n), nil
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return nil, &FilterCompileError{Code: FilterErrorInvalidValue, Field: fieldName, Path: path, reason: "number literal is not parseable"}
		}
		return finiteFloat(fieldName, f, path)
	default:
		return nil, invalidTypeError(fieldName, path, raw, "float")
	}
}

func finiteFloat(fieldName string, f float64, path string) (interface{}, error) {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil, &FilterCompileError{Code: FilterErrorInvalidValue, Field: fieldName, Path: path, reason: "NaN and Inf are not allowed"}
	}
	return f, nil
}

// compiledFilter is the frozen compilation output. It is an unexported type:
// external code can obtain a Sqlizer but never inject or alter the structure.
type compiledFilter struct {
	sql  string
	args []interface{}
}

func (c *compiledFilter) ToSql() (string, []interface{}, error) {
	if len(c.args) == 0 {
		return c.sql, []interface{}{}, nil
	}
	out := make([]interface{}, len(c.args))
	copy(out, c.args)
	return c.sql, out, nil
}
