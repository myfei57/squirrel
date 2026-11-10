package filterverify

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"

	sqrl "github.com/Masterminds/squirrel"
	_ "modernc.org/sqlite"
)

const maxI64 = int64(9223372036854775807)
const minI64 = int64(-9223372036854775808)

// finiteDataset is the single source of truth shared by the interpreter and
// the SQLite database. nil means SQL NULL.
func finiteDataset() map[int]Row {
	return map[int]Row{
		0: {"id": int64(1), "age": int64(10), "score": 1.5, "name": "a", "active": true, "blob": []byte{1}, "code": int64(0)},
		1: {"id": int64(2), "age": nil, "score": 2.5, "name": nil, "active": false, "blob": nil, "code": nil},
		2: {"id": int64(3), "age": int64(30), "score": nil, "name": "b", "active": true, "blob": []byte{1, 2}, "code": int64(3)},
		3: {"id": int64(4), "age": int64(30), "score": 4.5, "name": "a", "active": nil, "blob": []byte{3}, "code": int64(3)},
		4: {"id": nil, "age": int64(50), "score": 5.5, "name": "c", "active": true, "blob": []byte{}, "code": int64(5)},
		5: {"id": int64(6), "age": nil, "score": nil, "name": nil, "active": false, "blob": nil, "code": nil},
		6: {"id": maxI64, "age": int64(70), "score": 7.25, "name": "max", "active": true, "blob": []byte{7}, "code": int64(8)},
		7: {"id": minI64, "age": int64(80), "score": -0.5, "name": "min", "active": false, "blob": []byte{8}, "code": int64(9)},
	}
}

const ddl = `
CREATE TABLE widgets (
	rid    INTEGER PRIMARY KEY,
	id     INTEGER,
	age    INTEGER,
	score  REAL,
	name   TEXT,
	active INTEGER,
	blob   BLOB,
	code   INTEGER
)`

func openFixture(t *testing.T) (*sql.DB, *sqrl.FilterSchema, map[int]Row) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	// A :memory: database is private to one connection; pin the pool so the
	// fixture table is visible to every query. Compiler concurrency itself is
	// covered by the package-level tests.
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })

	if _, err := db.Exec(ddl); err != nil {
		t.Fatalf("ddl: %v", err)
	}

	rows := finiteDataset()
	schema, err := sqrl.NewFilterSchema(map[string]sqrl.FieldSchema{
		"id":     {Column: "id", Type: sqrl.FieldTypeInt, Nullable: true, MaxIN: 10},
		"age":    {Column: "age", Type: sqrl.FieldTypeInt, Nullable: true, MaxIN: 10},
		"score":  {Column: "score", Type: sqrl.FieldTypeFloat, Nullable: true, MaxIN: 10},
		"name":   {Column: "name", Type: sqrl.FieldTypeString, Nullable: true, MaxIN: 10, Sensitive: true},
		"active": {Column: "active", Type: sqrl.FieldTypeBool, Nullable: true},
		"blob":   {Column: "blob", Type: sqrl.FieldTypeBytes, Nullable: true, MaxIN: 10},
		"code":   {Column: "code", Type: sqrl.FieldTypeBytes, Nullable: true, MaxIN: 10, BytesAsList: true},
	}, sqrl.FilterBudget{})
	if err != nil {
		t.Fatalf("schema: %v", err)
	}

	insert, args, err := sqrl.Insert("widgets").
		Columns("rid", "id", "age", "score", "name", "active", "blob", "code").
		Values(0, rows[0]["id"], rows[0]["age"], rows[0]["score"], rows[0]["name"], rows[0]["active"], rows[0]["blob"], rows[0]["code"]).
		Values(1, rows[1]["id"], rows[1]["age"], rows[1]["score"], rows[1]["name"], rows[1]["active"], rows[1]["blob"], rows[1]["code"]).
		Values(2, rows[2]["id"], rows[2]["age"], rows[2]["score"], rows[2]["name"], rows[2]["active"], rows[2]["blob"], rows[2]["code"]).
		Values(3, rows[3]["id"], rows[3]["age"], rows[3]["score"], rows[3]["name"], rows[3]["active"], rows[3]["blob"], rows[3]["code"]).
		Values(4, rows[4]["id"], rows[4]["age"], rows[4]["score"], rows[4]["name"], rows[4]["active"], rows[4]["blob"], rows[4]["code"]).
		Values(5, rows[5]["id"], rows[5]["age"], rows[5]["score"], rows[5]["name"], rows[5]["active"], rows[5]["blob"], rows[5]["code"]).
		Values(6, rows[6]["id"], rows[6]["age"], rows[6]["score"], rows[6]["name"], rows[6]["active"], rows[6]["blob"], rows[6]["code"]).
		Values(7, rows[7]["id"], rows[7]["age"], rows[7]["score"], rows[7]["name"], rows[7]["active"], rows[7]["blob"], rows[7]["code"]).
		ToSql()
	if err != nil {
		t.Fatalf("build insert: %v", err)
	}
	if _, err := db.Exec(insert, args...); err != nil {
		t.Fatalf("insert fixture: %v", err)
	}
	return db, schema, rows
}

func sqliteRids(t *testing.T, db *sql.DB, cond sqrl.Sqlizer) []int {
	t.Helper()
	query, qargs, err := cond.ToSql()
	if err != nil {
		t.Fatalf("cond ToSql: %v", err)
	}
	full := "SELECT rid FROM widgets WHERE " + query + " ORDER BY rid"
	rs, err := db.Query(full, qargs...)
	if err != nil {
		t.Fatalf("sqlite query %q args %v: %v", full, qargs, err)
	}
	defer rs.Close()
	var out []int
	for rs.Next() {
		var rid int
		if err := rs.Scan(&rid); err != nil {
			t.Fatalf("scan: %v", err)
		}
		out = append(out, rid)
	}
	return out
}

func mustJSONNumber(t *testing.T, s string) json.Number {
	t.Helper()
	var n json.Number
	if err := json.Unmarshal([]byte(s), &n); err != nil {
		t.Fatalf("json.Number %q: %v", s, err)
	}
	return n
}

type crossCase struct {
	name string
	tree *sqrl.Filter
	want []int // hard-coded specification, independent of both engines
}

func crossCases(t *testing.T) []crossCase {
	n := func(s string) json.Number { return mustJSONNumber(t, s) }
	return []crossCase{
		{"eq scalar", &sqrl.Filter{Op: sqrl.OpEq, Field: "age", Value: int64(30)}, []int{2, 3}},
		{"neq excludes NULL", &sqrl.Filter{Op: sqrl.OpNeq, Field: "age", Value: int64(30)}, []int{0, 4, 6, 7}},
		{"lt excludes NULL", &sqrl.Filter{Op: sqrl.OpLt, Field: "age", Value: int64(30)}, []int{0}},
		{"lte boundary", &sqrl.Filter{Op: sqrl.OpLte, Field: "age", Value: int64(30)}, []int{0, 2, 3}},
		{"gte", &sqrl.Filter{Op: sqrl.OpGte, Field: "age", Value: int64(50)}, []int{4, 6, 7}},
		{"gt float NULL excluded", &sqrl.Filter{Op: sqrl.OpGt, Field: "score", Value: 3.0}, []int{3, 4, 6}},
		{"float negative", &sqrl.Filter{Op: sqrl.OpLt, Field: "score", Value: 0.0}, []int{7}},
		{"string eq", &sqrl.Filter{Op: sqrl.OpEq, Field: "name", Value: "a"}, []int{0, 3}},
		{"string neq NULL excluded", &sqrl.Filter{Op: sqrl.OpNeq, Field: "name", Value: "a"}, []int{2, 4, 6, 7}},
		{"bool true NULL excluded", &sqrl.Filter{Op: sqrl.OpEq, Field: "active", Value: true}, []int{0, 2, 4, 6}},
		{"bool false", &sqrl.Filter{Op: sqrl.OpEq, Field: "active", Value: false}, []int{1, 5, 7}},
		{"is null", &sqrl.Filter{Op: sqrl.OpIsNull, Field: "age"}, []int{1, 5}},
		{"is not null", &sqrl.Filter{Op: sqrl.OpIsNotNull, Field: "age"}, []int{0, 2, 3, 4, 6, 7}},
		{"not is null keeps 3VL", &sqrl.Filter{Op: sqrl.OpNot, Children: []*sqrl.Filter{{Op: sqrl.OpIsNull, Field: "age"}}}, []int{0, 2, 3, 4, 6, 7}},
		{"not eq equals neq including NULL exclusion", &sqrl.Filter{Op: sqrl.OpNot, Children: []*sqrl.Filter{{Op: sqrl.OpEq, Field: "age", Value: int64(30)}}}, []int{0, 4, 6, 7}},
		{"in plain", &sqrl.Filter{Op: sqrl.OpIn, Field: "id", Values: []interface{}{int64(1), int64(3)}}, []int{0, 2}},
		{"in with NULL only matches equality", &sqrl.Filter{Op: sqrl.OpIn, Field: "id", Values: []interface{}{int64(3), nil}}, []int{2}},
		{"not in with NULL never true", &sqrl.Filter{Op: sqrl.OpNotIn, Field: "id", Values: []interface{}{int64(3), nil}}, []int{}},
		{"in empty is false", &sqrl.Filter{Op: sqrl.OpIn, Field: "age", Values: []interface{}{}}, []int{}},
		{"not in empty is true for all incl NULL rows", &sqrl.Filter{Op: sqrl.OpNotIn, Field: "age", Values: []interface{}{}}, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"and", &sqrl.Filter{Op: sqrl.OpAnd, Children: []*sqrl.Filter{
			{Op: sqrl.OpGte, Field: "age", Value: int64(30)},
			{Op: sqrl.OpEq, Field: "name", Value: "a"},
		}}, []int{3}},
		{"or with null check", &sqrl.Filter{Op: sqrl.OpOr, Children: []*sqrl.Filter{
			{Op: sqrl.OpIsNull, Field: "age"},
			{Op: sqrl.OpEq, Field: "id", Value: int64(6)},
		}}, []int{1, 5}},
		// rid3 (active NULL) and NULL-age rows stay UNKNOWN under NOT and are
		// excluded, never promoted to binary TRUE.
		{"not of and keeps UNKNOWN excluded", &sqrl.Filter{Op: sqrl.OpNot, Children: []*sqrl.Filter{
			{Op: sqrl.OpAnd, Children: []*sqrl.Filter{
				{Op: sqrl.OpGte, Field: "age", Value: int64(30)},
				{Op: sqrl.OpEq, Field: "active", Value: true},
			}},
		}}, []int{0, 1, 5, 7}},
		// NULL-age rows make the NOT(age > 50) branch UNKNOWN, so they drop out.
		{"nested and/or/not", &sqrl.Filter{Op: sqrl.OpAnd, Children: []*sqrl.Filter{
			{Op: sqrl.OpOr, Children: []*sqrl.Filter{
				{Op: sqrl.OpEq, Field: "name", Value: "a"},
				{Op: sqrl.OpIsNull, Field: "name"},
			}},
			{Op: sqrl.OpNot, Children: []*sqrl.Filter{
				{Op: sqrl.OpGt, Field: "age", Value: int64(50)},
			}},
		}}, []int{0, 3}},
		{"blob eq", &sqrl.Filter{Op: sqrl.OpEq, Field: "blob", Value: []byte{1, 2}}, []int{2}},
		{"blob in scalar blobs", &sqrl.Filter{Op: sqrl.OpIn, Field: "blob", Values: []interface{}{[]byte{1}, []byte{3}}}, []int{0, 3}},
		{"bytes-as-list in single byte", &sqrl.Filter{Op: sqrl.OpIn, Field: "code", Value: []byte{3}}, []int{2, 3}},
		{"bytes-as-list in multi bytes", &sqrl.Filter{Op: sqrl.OpIn, Field: "code", Value: []byte{7, 8}}, []int{6}},
		{"int64 max precision via json.Number", &sqrl.Filter{Op: sqrl.OpEq, Field: "id", Value: n("9223372036854775807")}, []int{6}},
		{"int64 min precision via json.Number", &sqrl.Filter{Op: sqrl.OpEq, Field: "id", Value: n("-9223372036854775808")}, []int{7}},
		{"float eq", &sqrl.Filter{Op: sqrl.OpEq, Field: "score", Value: 7.25}, []int{6}},
		{"is null code", &sqrl.Filter{Op: sqrl.OpIsNull, Field: "code"}, []int{1, 5}},
	}
}

func TestSQLiteMatchesThreeValuedInterpreter(t *testing.T) {
	db, schema, rows := openFixture(t)

	for _, tc := range crossCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			cond, err := sqrl.CompileFilter(schema, tc.tree)
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			gotDB := sqliteRids(t, db, cond)
			gotInterp, err := MatchingRows(tc.tree, rows)
			if err != nil {
				t.Fatalf("interpreter: %v", err)
			}
			want := append([]int(nil), tc.want...)
			sort.Ints(want)
			if fmt.Sprint(gotDB) != fmt.Sprint(want) {
				t.Errorf("sqlite mismatch: got %v want %v", gotDB, want)
			}
			if fmt.Sprint(gotInterp) != fmt.Sprint(want) {
				t.Errorf("interpreter mismatch: got %v want %v", gotInterp, want)
			}
		})
	}
}

func TestInterpreterUnknownTriExcluded(t *testing.T) {
	// Direct tri-level check: NULL comparison must be UNKNOWN, not FALSE/TRUE.
	row := Row{"age": nil}
	v, err := Eval(&sqrl.Filter{Op: sqrl.OpEq, Field: "age", Value: int64(1)}, row)
	if err != nil {
		t.Fatal(err)
	}
	if v != triUnknown {
		t.Fatalf("NULL eq = %v, want UNKNOWN", v)
	}
	v, _ = Eval(&sqrl.Filter{Op: sqrl.OpNot, Children: []*sqrl.Filter{
		{Op: sqrl.OpEq, Field: "age", Value: int64(1)},
	}}, row)
	if v != triUnknown {
		t.Fatalf("NOT(NULL eq) = %v, want UNKNOWN (must not become TRUE)", v)
	}
	v, _ = Eval(&sqrl.Filter{Op: sqrl.OpNotIn, Field: "age", Values: []interface{}{int64(1), nil}}, row)
	if v != triUnknown {
		t.Fatalf("NULL NOT IN (1,NULL) = %v, want UNKNOWN", v)
	}
}

func TestCompileRejections(t *testing.T) {
	_, schema, _ := openFixture(t)

	malicious := []string{
		"id; DROP TABLE widgets--",
		"age OR 1=1",
		"name` --",
	}
	for _, m := range malicious {
		cond, err := sqrl.CompileFilter(schema, &sqrl.Filter{Op: sqrl.OpEq, Field: m, Value: int64(1)})
		if err == nil || cond != nil {
			t.Fatalf("malicious field %q must be rejected, got cond=%v err=%v", m, cond, err)
		}
		var fce *sqrl.FilterCompileError
		if !errors.As(err, &fce) {
			t.Fatalf("want FilterCompileError, got %T", err)
		}
		if fce.Code != sqrl.FilterErrorUnknownField {
			t.Fatalf("code = %s, want unknown_field", fce.Code)
		}
	}

	// Unknown operator carrying a payload never reaches SQL structure.
	cond, err := sqrl.CompileFilter(schema, &sqrl.Filter{Op: "eq; DROP TABLE widgets--", Field: "age", Value: int64(1)})
	if err == nil || cond != nil {
		t.Fatalf("malicious op must be rejected, got cond=%v", cond)
	}

	// Unregistered column identity cannot be smuggled through schema build.
	s, err := sqrl.NewFilterSchema(map[string]sqrl.FieldSchema{
		"evil": {Column: "id; DROP TABLE widgets--", Type: sqrl.FieldTypeInt},
	}, sqrl.FilterBudget{})
	if err == nil || s != nil {
		t.Fatalf("unsafe column registration must fail")
	}

	// Malicious value stays a bound parameter: absent from SQL text.
	cond, err = sqrl.CompileFilter(schema, &sqrl.Filter{Op: sqrl.OpEq, Field: "name", Value: "x' OR '1'='1"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	sqlText, args, err := cond.ToSql()
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(sqlText, "OR") || strings.Contains(sqlText, "1'='1") {
		t.Fatalf("value leaked into SQL: %s", sqlText)
	}
	if len(args) != 1 {
		t.Fatalf("args = %v", args)
	}

	// Sensitive value must not appear in error text.
	_, err = sqrl.CompileFilter(schema, &sqrl.Filter{Op: sqrl.OpEq, Field: "name", Value: int64(42)})
	if err == nil || strings.Contains(err.Error(), "42") {
		t.Fatalf("sensitive error must not embed values: %v", err)
	}
}

func TestBudgetRejections(t *testing.T) {
	_, base, _ := openFixture(t)

	// Tree depth budget.
	small, err := sqrl.NewFilterSchema(map[string]sqrl.FieldSchema{
		"age": {Column: "age", Type: sqrl.FieldTypeInt, Ops: []sqrl.FilterOp{sqrl.OpEq}},
	}, sqrl.FilterBudget{MaxDepth: 2, MaxNodes: 50, MaxArgs: 50, MaxSQLBytes: 4096})
	if err != nil {
		t.Fatal(err)
	}
	deep := &sqrl.Filter{Op: sqrl.OpNot, Children: []*sqrl.Filter{{
		Op: sqrl.OpNot, Children: []*sqrl.Filter{{Op: sqrl.OpEq, Field: "age", Value: int64(1)}},
	}}}
	if cond, err := sqrl.CompileFilter(small, deep); err == nil || cond != nil {
		t.Fatalf("depth budget must reject, got %v", cond)
	}

	// Node budget.
	wide := &sqrl.Filter{Op: sqrl.OpAnd, Children: []*sqrl.Filter{
		{Op: sqrl.OpEq, Field: "age", Value: int64(1)},
		{Op: sqrl.OpEq, Field: "age", Value: int64(2)},
	}}
	tiny, err := sqrl.NewFilterSchema(map[string]sqrl.FieldSchema{
		"age": {Column: "age", Type: sqrl.FieldTypeInt, Ops: []sqrl.FilterOp{sqrl.OpEq}},
	}, sqrl.FilterBudget{MaxDepth: 10, MaxNodes: 2, MaxArgs: 50, MaxSQLBytes: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sqrl.CompileFilter(tiny, wide); err == nil {
		t.Fatalf("node budget must reject")
	}

	// Cycle.
	cycle := &sqrl.Filter{Op: sqrl.OpNot}
	cycle.Children = []*sqrl.Filter{cycle}
	if cond, err := sqrl.CompileFilter(base, cycle); err == nil || cond != nil {
		t.Fatalf("cycle must reject, got %v", cond)
	}

	// Unknown node kind.
	if cond, err := sqrl.CompileFilter(base, &sqrl.Filter{Op: "wat", Field: "age"}); err == nil || cond != nil {
		t.Fatalf("unknown node must reject, got %v", cond)
	}

	// Failed compiles leave nothing executable: no SQL/args surface at all.
	cond, err := sqrl.CompileFilter(base, &sqrl.Filter{Op: sqrl.OpEq, Field: "ghost", Value: int64(1)})
	if err == nil {
		t.Fatal("unknown field expected to fail")
	}
	if cond != nil {
		t.Fatal("failed compile returned a non-nil Sqlizer")
	}
}

func TestEndToEndSelectBuilder(t *testing.T) {
	db, schema, rows := openFixture(t)

	tree := &sqrl.Filter{Op: sqrl.OpIn, Field: "age", Values: []interface{}{int64(10), int64(30)}}
	cond, err := sqrl.CompileFilter(schema, tree)
	if err != nil {
		t.Fatal(err)
	}
	order, err := schema.CompileOrderBy([]sqrl.OrderClause{{Field: "id", Desc: true}})
	if err != nil {
		t.Fatal(err)
	}

	q := sqrl.StatementBuilder.Select("rid").From("widgets").
		Where(cond).
		Where("rid > ?", 0).
		OrderByClause(order)
	query, args, err := q.ToSql()
	if err != nil {
		t.Fatal(err)
	}
	wantSQL := "SELECT rid FROM widgets WHERE age IN (?,?) AND rid > ? ORDER BY id DESC"
	if query != wantSQL {
		t.Fatalf("sql = %q, want %q", query, wantSQL)
	}
	// Tree args keep traversal order; builder-supplied args follow.
	wantArgs := []interface{}{int64(10), int64(30), int64(0)}
	if fmt.Sprint(args) != fmt.Sprint(wantArgs) {
		t.Fatalf("args = %v, want %v", args, wantArgs)
	}

	rs, err := db.Query(query, args...)
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	defer rs.Close()
	var got []int
	for rs.Next() {
		var rid int
		if err := rs.Scan(&rid); err != nil {
			t.Fatal(err)
		}
		got = append(got, rid)
	}
	// age in (10,30) matches rids 0,2,3; rid>0 drops rid0; id DESC => 4 then 3.
	if fmt.Sprint(got) != fmt.Sprint([]int{3, 2}) {
		t.Fatalf("rids = %v, want [3 2]", got)
	}

	// Same result set as the interpreter for the filter alone.
	interp, err := MatchingRows(tree, rows)
	if err != nil {
		t.Fatal(err)
	}
	sort.Ints(interp)
	if fmt.Sprint(interp) != fmt.Sprint([]int{0, 2, 3}) {
		t.Fatalf("interpreter rids = %v", interp)
	}
}

func TestConcurrentCompileAndQuery(t *testing.T) {
	db, schema, rows := openFixture(t)

	trees := crossCases(t)
	expected := make(map[string][]int, len(trees))
	compiled := make(map[string]sqrl.Sqlizer, len(trees))
	for _, tc := range trees {
		want, err := MatchingRows(tc.tree, rows)
		if err != nil {
			t.Fatal(err)
		}
		sort.Ints(want)
		expected[tc.name] = want
		cond, err := sqrl.CompileFilter(schema, tc.tree)
		if err != nil {
			t.Fatalf("compile %s: %v", tc.name, err)
		}
		compiled[tc.name] = cond
	}

	const workers = 24
	const iterations = 50
	var wg sync.WaitGroup
	errCh := make(chan error, workers)
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				for _, tc := range trees {
					got := sqliteRids(t, db, compiled[tc.name])
					if fmt.Sprint(got) != fmt.Sprint(expected[tc.name]) {
						errCh <- fmt.Errorf("%s: got %v want %v", tc.name, got, expected[tc.name])
						return
					}
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
