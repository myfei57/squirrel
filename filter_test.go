package squirrel

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testSchema(t *testing.T) *FilterSchema {
	t.Helper()
	s, err := NewFilterSchema(map[string]FieldSchema{
		"id": {
			Column:   "u.id",
			Type:     FieldTypeInt,
			Ops:      []FilterOp{OpEq, OpNeq, OpLt, OpLte, OpGt, OpGte, OpIn, OpNotIn, OpIsNull, OpIsNotNull},
			Nullable: true,
			MaxIN:    5,
		},
		"age": {
			Column: "u.age",
			Type:   FieldTypeInt,
			Ops:    []FilterOp{OpEq, OpNeq, OpLt, OpLte, OpGt, OpGte, OpIn, OpNotIn},
			MaxIN:  4,
		},
		"score": {
			Column: "u.score",
			Type:   FieldTypeFloat,
			Ops:    []FilterOp{OpEq, OpLt, OpGt, OpIn},
			MaxIN:  3,
		},
		"name": {
			Column:    "u.name",
			Type:      FieldTypeString,
			Ops:       []FilterOp{OpEq, OpNeq, OpIn, OpNotIn, OpIsNull, OpIsNotNull},
			Nullable:  true,
			MaxIN:     3,
			Sensitive: true,
		},
		"active": {
			Column: "u.active",
			Type:   FieldTypeBool,
			Ops:    []FilterOp{OpEq},
		},
		"blob": {
			Column:   "u.blob",
			Type:     FieldTypeBytes,
			Ops:      []FilterOp{OpEq, OpNeq, OpIn, OpNotIn, OpIsNull},
			Nullable: true,
			MaxIN:    4,
		},
		"bloblist": {
			Column:      "u.bytecode",
			Type:        FieldTypeBytes,
			Ops:         []FilterOp{OpIn, OpNotIn},
			MaxIN:       8,
			BytesAsList: true,
		},
	}, FilterBudget{})
	require.NoError(t, err)
	return s
}

func compile(t *testing.T, schema *FilterSchema, root *Filter) (string, []interface{}) {
	t.Helper()
	cond, err := CompileFilter(schema, root)
	require.NoError(t, err)
	require.NotNil(t, cond)
	sql, args, err := cond.ToSql()
	require.NoError(t, err)
	return sql, args
}

func compileErr(t *testing.T, schema *FilterSchema, root *Filter) *FilterCompileError {
	t.Helper()
	cond, err := CompileFilter(schema, root)
	require.Error(t, err)
	require.Nil(t, cond, "failed compilation must not return a Sqlizer")
	var fce *FilterCompileError
	require.True(t, errors.As(err, &fce), "expected *FilterCompileError, got %T: %v", err, err)
	return fce
}

func TestCompileFilterLeafOperators(t *testing.T) {
	schema := testSchema(t)

	cases := []struct {
		name string
		node *Filter
		sql  string
		args []interface{}
	}{
		{"eq", &Filter{Op: OpEq, Field: "age", Value: 30}, "u.age = ?", []interface{}{int64(30)}},
		{"neq", &Filter{Op: OpNeq, Field: "age", Value: 30}, "u.age <> ?", []interface{}{int64(30)}},
		{"lt", &Filter{Op: OpLt, Field: "age", Value: 30}, "u.age < ?", []interface{}{int64(30)}},
		{"lte", &Filter{Op: OpLte, Field: "age", Value: 30}, "u.age <= ?", []interface{}{int64(30)}},
		{"gt", &Filter{Op: OpGt, Field: "age", Value: 30}, "u.age > ?", []interface{}{int64(30)}},
		{"gte", &Filter{Op: OpGte, Field: "age", Value: 30}, "u.age >= ?", []interface{}{int64(30)}},
		{"in", &Filter{Op: OpIn, Field: "age", Values: []interface{}{1, 2}}, "u.age IN (?,?)", []interface{}{int64(1), int64(2)}},
		{"not in", &Filter{Op: OpNotIn, Field: "age", Values: []interface{}{1, 2}}, "u.age NOT IN (?,?)", []interface{}{int64(1), int64(2)}},
		{"is null", &Filter{Op: OpIsNull, Field: "id"}, "u.id IS NULL", []interface{}{}},
		{"is not null", &Filter{Op: OpIsNotNull, Field: "id"}, "u.id IS NOT NULL", []interface{}{}},
		{"float", &Filter{Op: OpLt, Field: "score", Value: 1.5}, "u.score < ?", []interface{}{1.5}},
		{"string", &Filter{Op: OpEq, Field: "name", Value: "alice"}, "u.name = ?", []interface{}{"alice"}},
		{"bool", &Filter{Op: OpEq, Field: "active", Value: true}, "u.active = ?", []interface{}{true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sql, args := compile(t, schema, tc.node)
			assert.Equal(t, tc.sql, sql)
			assert.Equal(t, tc.args, args)
		})
	}
}

func TestCompileFilterLogicalStructure(t *testing.T) {
	schema := testSchema(t)

	root := &Filter{Op: OpAnd, Children: []*Filter{
		{Op: OpOr, Children: []*Filter{
			{Op: OpGte, Field: "age", Value: 18},
			{Op: OpIsNull, Field: "id"},
		}},
		{Op: OpNot, Children: []*Filter{
			{Op: OpEq, Field: "name", Value: "bob"},
		}},
		{Op: OpIn, Field: "age", Values: []interface{}{1, 2, 3}},
	}}

	sql, args := compile(t, schema, root)
	assert.Equal(t,
		"((u.age >= ? OR u.id IS NULL) AND NOT (u.name = ?) AND u.age IN (?,?,?))",
		sql)
	assert.Equal(t, []interface{}{int64(18), "bob", int64(1), int64(2), int64(3)}, args)
}

func TestCompileFilterInEmptyAndNullSemantics(t *testing.T) {
	schema := testSchema(t)

	sql, args := compile(t, schema, &Filter{Op: OpIn, Field: "age", Values: []interface{}{}})
	assert.Equal(t, "(1=0)", sql)
	assert.Equal(t, []interface{}{}, args)

	sql, args = compile(t, schema, &Filter{Op: OpNotIn, Field: "age", Values: []interface{}{}})
	assert.Equal(t, "(1=1)", sql)
	assert.Equal(t, []interface{}{}, args)

	// NULL inside IN is bound as a real NULL arg, not stringified.
	sql, args = compile(t, schema, &Filter{Op: OpIn, Field: "id", Values: []interface{}{1, nil, 3}})
	assert.Equal(t, "u.id IN (?,?,?)", sql)
	assert.Equal(t, []interface{}{int64(1), nil, int64(3)}, args)

	// NULL elements require a nullable field.
	fce := compileErr(t, schema, &Filter{Op: OpIn, Field: "age", Values: []interface{}{1, nil}})
	assert.Equal(t, FilterErrorNotNullable, fce.Code)
	assert.Equal(t, "$.values[1]", fce.Path)
}

func TestCompileFilterInLengthBudget(t *testing.T) {
	schema := testSchema(t)
	fce := compileErr(t, schema, &Filter{Op: OpIn, Field: "age", Values: []interface{}{1, 2, 3, 4, 5}})
	assert.Equal(t, FilterErrorBudget, fce.Code)
	assert.True(t, errors.Is(fce, ErrFilterBudget))
}

func TestCompileFilterNullability(t *testing.T) {
	schema := testSchema(t)

	// age is not nullable and its allow-list does not contain is_null.
	fce := compileErr(t, schema, &Filter{Op: OpIsNull, Field: "age"})
	assert.Equal(t, FilterErrorOpNotAllowed, fce.Code)

	// scalar nil is rejected with an explicit contract error.
	fce = compileErr(t, schema, &Filter{Op: OpEq, Field: "age", Value: nil})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
	assert.Contains(t, fce.Error(), "is_null")

	// typed nil pointers are rejected rather than treated as NULL.
	var nilInt *int
	fce = compileErr(t, schema, &Filter{Op: OpEq, Field: "age", Value: nilInt})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
	assert.Contains(t, fce.Error(), "typed-nil")

	nonNil := 7
	fce = compileErr(t, schema, &Filter{Op: OpEq, Field: "age", Value: &nonNil})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)

	// typed nil inside an IN list is rejected too.
	fce = compileErr(t, schema, &Filter{Op: OpIn, Field: "id", Values: []interface{}{nilInt}})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
	assert.Equal(t, "$.values[0]", fce.Path)
}

func TestCompileFilterIntegerPrecision(t *testing.T) {
	schema := testSchema(t)

	cases := []struct {
		name  string
		value interface{}
		want  interface{}
	}{
		{"int", int(7), int64(7)},
		{"int64 max via number", json.Number("9223372036854775807"), int64(9223372036854775807)},
		{"int64 min via number", json.Number("-9223372036854775808"), int64(-9223372036854775808)},
		{"integral float64", float64(42), int64(42)},
		{"integral float32", float32(42), int64(42)},
		{"uint32", uint32(42), int64(42)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, args := compile(t, schema, &Filter{Op: OpEq, Field: "age", Value: tc.value})
			assert.Equal(t, []interface{}{tc.want}, args)
		})
	}

	reject := []struct {
		name  string
		value interface{}
		code  FilterErrorCode
	}{
		{"json.Number overflow", json.Number("9223372036854775808"), FilterErrorOverflow},
		{"json.Number huge", json.Number("99999999999999999999999999"), FilterErrorOverflow},
		{"float64 max int64 rounds to 2^63", float64(9223372036854775807), FilterErrorOverflow},
		{"float64 huge", 1e20, FilterErrorOverflow},
		{"float64 below min", -1e20, FilterErrorOverflow},
		{"fractional float", 3.5, FilterErrorInvalidValue},
		{"string pseudo number", "42", FilterErrorInvalidValue},
		{"bool", true, FilterErrorInvalidValue},
		{"uint64 max", uint64(18446744073709551615), FilterErrorOverflow},
		{"json.Number pseudo text", json.Number("12abc"), FilterErrorInvalidValue},
	}
	for _, tc := range reject {
		t.Run(tc.name, func(t *testing.T) {
			fce := compileErr(t, schema, &Filter{Op: OpEq, Field: "age", Value: tc.value})
			assert.Equal(t, tc.code, fce.Code, "unexpected reason: %s", fce.Error())
			assert.Equal(t, "age", fce.Field)
		})
	}
}

func TestCompileFilterFloatContract(t *testing.T) {
	schema := testSchema(t)

	_, args := compile(t, schema, &Filter{Op: OpEq, Field: "score", Value: json.Number("1.25")})
	assert.Equal(t, []interface{}{1.25}, args)

	fce := compileErr(t, schema, &Filter{Op: OpEq, Field: "score", Value: "1.25"})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
}

func TestCompileFilterBytesContract(t *testing.T) {
	schema := testSchema(t)

	// Default: []byte is one scalar blob arg.
	blob := []byte{1, 2, 3}
	sql, args := compile(t, schema, &Filter{Op: OpEq, Field: "blob", Value: blob})
	assert.Equal(t, "u.blob = ?", sql)
	assert.Equal(t, []interface{}{[]byte{1, 2, 3}}, args)

	// []byte inside a regular IN list stays a scalar blob, no reflect expansion.
	_, args = compile(t, schema, &Filter{Op: OpIn, Field: "blob", Values: []interface{}{[]byte{9}, []byte{8}}})
	assert.Equal(t, []interface{}{[]byte{9}, []byte{8}}, args)

	// BytesAsList: []byte expands per byte into integer args for IN.
	sql, args = compile(t, schema, &Filter{Op: OpIn, Field: "bloblist", Value: []byte{10, 20, 30}})
	assert.Equal(t, "u.bytecode IN (?,?,?)", sql)
	assert.Equal(t, []interface{}{int64(10), int64(20), int64(30)}, args)

	// Scalar ops are refused on list-configured byte fields.
	fce := compileErr(t, schema, &Filter{Op: OpIn, Field: "bloblist", Value: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9}})
	assert.Equal(t, FilterErrorBudget, fce.Code)

	// Plain []interface{} must not be accepted as a scalar anywhere.
	fce = compileErr(t, schema, &Filter{Op: OpEq, Field: "age", Value: []interface{}{1}})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
}

func TestCompileFilterRejectsUnknownNodesAndFields(t *testing.T) {
	schema := testSchema(t)

	fce := compileErr(t, schema, &Filter{Op: "eq; DROP TABLE u--", Field: "age", Value: 1})
	assert.Equal(t, FilterErrorUnknownOp, fce.Code)

	fce = compileErr(t, schema, &Filter{Op: OpEq, Field: "id; DROP TABLE u--", Value: 1})
	assert.Equal(t, FilterErrorUnknownField, fce.Code)
	assert.True(t, errors.Is(fce, ErrFilterUnknownField))

	// Operator not in the field allow-list.
	fce = compileErr(t, schema, &Filter{Op: OpLt, Field: "name", Value: "a"})
	assert.Equal(t, FilterErrorOpNotAllowed, fce.Code)

	// Malformed shapes.
	fce = compileErr(t, schema, &Filter{Op: OpEq, Value: 1})
	assert.Equal(t, FilterErrorMalformedNode, fce.Code)

	fce = compileErr(t, schema, &Filter{Op: OpAnd, Children: nil})
	assert.Equal(t, FilterErrorMalformedNode, fce.Code)

	fce = compileErr(t, schema, &Filter{Op: OpNot, Children: nil})
	assert.Equal(t, FilterErrorMalformedNode, fce.Code)

	fce = compileErr(t, schema, &Filter{Op: OpNot, Children: []*Filter{
		{Op: OpEq, Field: "age", Value: 1},
		{Op: OpEq, Field: "age", Value: 2},
	}})
	assert.Equal(t, FilterErrorMalformedNode, fce.Code)

	fce = compileErr(t, schema, &Filter{Op: OpEq, Field: "age", Value: 1, Children: []*Filter{{Op: OpEq, Field: "age", Value: 1}}})
	assert.Equal(t, FilterErrorMalformedNode, fce.Code)

	fce = compileErr(t, schema, &Filter{Op: OpAnd, Field: "age", Children: []*Filter{{Op: OpEq, Field: "age", Value: 1}}})
	assert.Equal(t, FilterErrorMalformedNode, fce.Code)

	// Nil root.
	cond, err := CompileFilter(schema, nil)
	assert.Error(t, err)
	assert.Nil(t, cond)

	// Nil schema.
	cond, err = CompileFilter(nil, &Filter{Op: OpEq, Field: "age", Value: 1})
	assert.Error(t, err)
	assert.Nil(t, cond)
}

func TestCompileFilterCycleAndReusedNodes(t *testing.T) {
	schema := testSchema(t)

	cycle := &Filter{Op: OpNot}
	cycle.Children = []*Filter{cycle}
	fce := compileErr(t, schema, cycle)
	assert.Equal(t, FilterErrorCycle, fce.Code)
	assert.True(t, errors.Is(fce, ErrFilterCycle))

	shared := &Filter{Op: OpEq, Field: "age", Value: 1}
	dag := &Filter{Op: OpAnd, Children: []*Filter{shared, shared}}
	fce = compileErr(t, schema, dag)
	assert.Equal(t, FilterErrorCycle, fce.Code)
}

func TestCompileFilterBudgets(t *testing.T) {
	small, err := NewFilterSchema(map[string]FieldSchema{
		"age": {Column: "age", Type: FieldTypeInt, Ops: []FilterOp{OpEq, OpIn, OpNotIn}, MaxIN: 10},
	}, FilterBudget{MaxDepth: 2, MaxNodes: 3, MaxArgs: 4, MaxSQLBytes: 200})
	require.NoError(t, err)

	// Depth: root(1) -> child(2) -> grandchild(3) exceeds MaxDepth 2.
	deep := &Filter{Op: OpNot, Children: []*Filter{{
		Op: OpNot, Children: []*Filter{{Op: OpEq, Field: "age", Value: 1}},
	}}}
	fce := compileErr(t, small, deep)
	assert.Equal(t, FilterErrorBudget, fce.Code)
	assert.Contains(t, fce.Error(), "depth")

	// Nodes: root + 3 children = 4 > MaxNodes 3.
	wide := &Filter{Op: OpAnd, Children: []*Filter{
		{Op: OpEq, Field: "age", Value: 1},
		{Op: OpEq, Field: "age", Value: 2},
		{Op: OpEq, Field: "age", Value: 3},
	}}
	fce = compileErr(t, small, wide)
	assert.Equal(t, FilterErrorBudget, fce.Code)
	assert.Contains(t, fce.Error(), "node count")

	// Args: IN of 5 elements exceeds MaxArgs 4.
	fce = compileErr(t, small, &Filter{Op: OpIn, Field: "age", Values: []interface{}{1, 2, 3, 4, 5}})
	assert.Equal(t, FilterErrorBudget, fce.Code)

	// SQL bytes: tiny budget rejects even a leaf.
	tiny, err := NewFilterSchema(map[string]FieldSchema{
		"age": {Column: "age", Type: FieldTypeInt, Ops: []FilterOp{OpEq}},
	}, FilterBudget{MaxSQLBytes: 5})
	require.NoError(t, err)
	fce = compileErr(t, tiny, &Filter{Op: OpEq, Field: "age", Value: 1})
	assert.Equal(t, FilterErrorBudget, fce.Code)
}

func TestCompileFilterFrozenOutput(t *testing.T) {
	schema := testSchema(t)

	tree := &Filter{Op: OpIn, Field: "age", Values: []interface{}{1, 2}}
	cond, err := CompileFilter(schema, tree)
	require.NoError(t, err)

	// Mutating the input tree after compilation must not change the output.
	tree.Values[0] = 999
	tree.Values = append(tree.Values, 3)
	sql, args, err := cond.ToSql()
	require.NoError(t, err)
	assert.Equal(t, "u.age IN (?,?)", sql)
	assert.Equal(t, []interface{}{int64(1), int64(2)}, args)

	// The returned arg slice is a copy: caller mutation cannot poison reuse.
	args[0] = "poison"
	_, args2, err := cond.ToSql()
	require.NoError(t, err)
	assert.Equal(t, []interface{}{int64(1), int64(2)}, args2)

	// Blob args are copied too.
	blob := []byte{1, 2}
	blobCond, err := CompileFilter(schema, &Filter{Op: OpEq, Field: "blob", Value: blob})
	require.NoError(t, err)
	blob[0] = 99
	_, blobArgs, err := blobCond.ToSql()
	require.NoError(t, err)
	assert.Equal(t, []interface{}{[]byte{1, 2}}, blobArgs)

	// The schema freezes its own construction input.
	decls := map[string]FieldSchema{"x": {Column: "x", Type: FieldTypeInt, Ops: []FilterOp{OpEq}}}
	s2, err := NewFilterSchema(decls, FilterBudget{})
	require.NoError(t, err)
	delete(decls, "x")
	_, err = s2.CompileFilter(&Filter{Op: OpEq, Field: "x", Value: 1})
	assert.NoError(t, err)
}

func TestCompileFilterSensitiveValuesRedacted(t *testing.T) {
	schema := testSchema(t)
	secret := "super-secret-token"

	fce := compileErr(t, schema, &Filter{Op: OpEq, Field: "name", Value: 123})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
	assert.NotContains(t, fce.Error(), secret)

	// Even when the bad value itself is the sensitive string, it is not echoed.
	fce = compileErr(t, schema, &Filter{Op: OpIn, Field: "name", Values: []interface{}{secret, 456}})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
	assert.NotContains(t, fce.Error(), secret)
	assert.Equal(t, "$.values[1]", fce.Path)
}

func TestCompileFilterBlocksOpaqueExpr(t *testing.T) {
	schema := testSchema(t)

	fce := compileErr(t, schema, &Filter{Op: OpEq, Field: "age", Value: Expr("1=1")})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
	assert.Contains(t, fce.Error(), "SQL expressions")

	fce = compileErr(t, schema, &Filter{Op: OpIn, Field: "age", Values: []interface{}{Expr("1=1")}})
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
}

func TestCompileFilterMaliciousTextNeverReachesSQL(t *testing.T) {
	schema := testSchema(t)
	payloads := []string{
		"id; DROP TABLE users--",
		"age OR 1=1",
		"name` --",
		"(*)",
		"age DESC; DELETE FROM users",
	}
	for _, payload := range payloads {
		cond, err := CompileFilter(schema, &Filter{Op: OpEq, Field: payload, Value: 1})
		require.Error(t, err, "payload %q must be rejected", payload)
		assert.Nil(t, cond)

		_, err = schema.CompileOrderBy([]OrderClause{{Field: payload, Desc: true}})
		require.Error(t, err, "payload %q must be rejected by order by", payload)
	}

	// A malicious value stays a bound placeholder arg, never SQL structure.
	evil := "x' OR '1'='1"
	sql, args := compile(t, schema, &Filter{Op: OpEq, Field: "name", Value: evil})
	assert.Equal(t, "u.name = ?", sql)
	assert.Equal(t, []interface{}{evil}, args)
	assert.NotContains(t, sql, evil)
}

func TestCompileOrderBy(t *testing.T) {
	schema := testSchema(t)

	cond, err := schema.CompileOrderBy([]OrderClause{
		{Field: "age", Desc: true},
		{Field: "name"},
	})
	require.NoError(t, err)
	sql, args, err := cond.ToSql()
	require.NoError(t, err)
	assert.Equal(t, "u.age DESC, u.name ASC", sql)
	assert.Equal(t, []interface{}{}, args)

	_, err = schema.CompileOrderBy(nil)
	assert.Error(t, err)
}

func TestNewFilterSchemaValidation(t *testing.T) {
	cases := []struct {
		name   string
		decls  map[string]FieldSchema
		budget FilterBudget
	}{
		{"empty", map[string]FieldSchema{}, FilterBudget{}},
		{"bad column", map[string]FieldSchema{"a": {Column: "a; DROP TABLE x", Type: FieldTypeInt}}, FilterBudget{}},
		{"bad type", map[string]FieldSchema{"a": {Column: "a", Type: FieldType(99)}}, FilterBudget{}},
		{"null op on non-nullable", map[string]FieldSchema{"a": {Column: "a", Type: FieldTypeInt, Ops: []FilterOp{OpIsNull}}}, FilterBudget{}},
		{"bytes flag on int", map[string]FieldSchema{"a": {Column: "a", Type: FieldTypeInt, BytesAsList: true}}, FilterBudget{}},
		{"negative maxin", map[string]FieldSchema{"a": {Column: "a", Type: FieldTypeInt, Ops: []FilterOp{OpIn}, MaxIN: -1}}, FilterBudget{}},
		{"logical op declared", map[string]FieldSchema{"a": {Column: "a", Type: FieldTypeInt, Ops: []FilterOp{OpAnd}}}, FilterBudget{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, err := NewFilterSchema(tc.decls, tc.budget)
			assert.Error(t, err)
			assert.Nil(t, s)
			var fce *FilterCompileError
			assert.True(t, errors.As(err, &fce))
			assert.Equal(t, FilterErrorSchema, fce.Code)
		})
	}

	// MaxIN may exceed the global arg budget; the smaller runtime bound wins.
	s, err := NewFilterSchema(map[string]FieldSchema{
		"a": {Column: "a", Type: FieldTypeInt, Ops: []FilterOp{OpIn}, MaxIN: 99},
	}, FilterBudget{MaxArgs: 10})
	require.NoError(t, err)
	require.NotNil(t, s)

	_, err = NewFilterSchema(map[string]FieldSchema{
		"a": {Column: "a", Type: FieldTypeInt},
	}, FilterBudget{MaxDepth: -1})
	assert.Error(t, err)
}

func TestCompileFilterWithSelectBuilder(t *testing.T) {
	schema := testSchema(t)
	root := &Filter{Op: OpAnd, Children: []*Filter{
		{Op: OpGte, Field: "age", Value: 18},
		{Op: OpIn, Field: "name", Values: []interface{}{"a", "b"}},
	}}
	cond, err := CompileFilter(schema, root)
	require.NoError(t, err)

	t.Run("question placeholders", func(t *testing.T) {
		q := Select("id").From("users").Where(cond).Where(Expr("tenant = ?", "t1"))
		sql, args, err := q.ToSql()
		require.NoError(t, err)
		assert.Equal(t,
			"SELECT id FROM users WHERE (u.age >= ? AND u.name IN (?,?)) AND tenant = ?",
			sql)
		// Original tree order is preserved, followed by builder args.
		assert.Equal(t, []interface{}{int64(18), "a", "b", "t1"}, args)
	})

	t.Run("dollar placeholders", func(t *testing.T) {
		q := StatementBuilder.PlaceholderFormat(Dollar).Select("id").From("users").Where(cond)
		sql, args, err := q.ToSql()
		require.NoError(t, err)
		assert.Equal(t,
			"SELECT id FROM users WHERE (u.age >= $1 AND u.name IN ($2,$3))",
			sql)
		assert.Equal(t, []interface{}{int64(18), "a", "b"}, args)
	})

	// Same compiled Sqlizer can be reused independently by multiple builders.
	q1 := Select("id").From("users").Where(cond)
	q2 := Select("id").From("admins").Where(cond)
	s1, a1, err := q1.ToSql()
	require.NoError(t, err)
	s2, a2, err := q2.ToSql()
	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(s1, "SELECT id FROM users WHERE "))
	assert.True(t, strings.HasPrefix(s2, "SELECT id FROM admins WHERE "))
	assert.Equal(t, a1, a2)
}

func TestCompileFilterConcurrentSchemaReuse(t *testing.T) {
	schema := testSchema(t)

	const workers = 32
	const iterations = 200
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		w := w
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				want := int64(w*iterations + i)
				cond, err := CompileFilter(schema, &Filter{Op: OpEq, Field: "age", Value: want})
				if err != nil {
					errCh <- err
					return
				}
				sql, args, err := cond.ToSql()
				if err != nil {
					errCh <- err
					return
				}
				if sql != "u.age = ?" || len(args) != 1 || args[0] != want {
					errCh <- fmt.Errorf("cross-contamination: got sql=%q args=%v want %d", sql, args, want)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Fatal(err)
	}
}

func TestCompileFilterJSONDecoding(t *testing.T) {
	schema := testSchema(t)
	raw := `{
		"op": "and",
		"children": [
			{"op": "eq", "field": "age", "value": 21},
			{"op": "in", "field": "id", "values": [9223372036854775807, null]}
		]
	}`
	var tree Filter
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	require.NoError(t, dec.Decode(&tree))

	sql, args := compile(t, schema, &tree)
	assert.Equal(t, "(u.age = ? AND u.id IN (?,?))", sql)
	assert.Equal(t, []interface{}{int64(21), int64(9223372036854775807), nil}, args)
}

func TestCompileFilterErrorPathAndSentinels(t *testing.T) {
	schema := testSchema(t)
	root := &Filter{Op: OpAnd, Children: []*Filter{
		{Op: OpOr, Children: []*Filter{
			{Op: OpEq, Field: "age", Value: "not-a-number"},
		}},
	}}
	fce := compileErr(t, schema, root)
	assert.Equal(t, FilterErrorInvalidValue, fce.Code)
	assert.Equal(t, "$.children[0].children[0]", fce.Path)
	assert.Equal(t, "age", fce.Field)
	assert.True(t, errors.Is(fce, ErrFilterInvalidValue))
	assert.Contains(t, fce.Error(), "$.children[0].children[0]")
}
