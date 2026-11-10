// Package rangesetsql_test executes RangeSet predicates against a real,
// embedded SQLite database. It lives in a separate module so the root
// squirrel module stays free of a CGO SQL-driver dependency; run it with a
// C toolchain available (CGO_ENABLED=1).
package rangesetsql_test

import (
	"database/sql"
	"fmt"
	"math"
	"sort"
	"testing"
	"time"

	sqrl "github.com/Masterminds/squirrel"
	_ "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func openSQLite(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, db.Close()) })
	_, err = db.Exec("CREATE TABLE nums (id INTEGER)")
	require.NoError(t, err)
	_, err = db.Exec("CREATE TABLE events (ts TEXT)")
	require.NoError(t, err)
	return db
}

// int64Result partitions query output into non-NULL values and a NULL count.
type int64Result struct {
	values   []int64
	nullRows int
}

func runInt64Query(t *testing.T, db *sql.DB, query string, args []interface{}) int64Result {
	t.Helper()
	rows, err := db.Query(query, args...)
	require.NoError(t, err)
	defer rows.Close()

	var out int64Result
	for rows.Next() {
		var v sql.NullInt64
		require.NoError(t, rows.Scan(&v))
		if v.Valid {
			out.values = append(out.values, v.Int64)
		} else {
			out.nullRows++
		}
	}
	require.NoError(t, rows.Err())
	sort.Slice(out.values, func(i, j int) bool { return out.values[i] < out.values[j] })
	return out
}

func insertInt64Rows(t *testing.T, db *sql.DB, values []sql.NullInt64) {
	t.Helper()
	tx, err := db.Begin()
	require.NoError(t, err)
	stmt, err := tx.Prepare("INSERT INTO nums (id) VALUES (?)")
	require.NoError(t, err)
	defer stmt.Close()
	for _, v := range values {
		if v.Valid {
			_, err = stmt.Exec(v.Int64)
		} else {
			_, err = stmt.Exec(nil)
		}
		require.NoError(t, err)
	}
	require.NoError(t, tx.Commit())
}

func sampleNullInt64Rows() []sql.NullInt64 {
	rows := []sql.NullInt64{
		{Valid: false}, // NULL
		{Int64: math.MinInt64, Valid: true},
		{Int64: math.MaxInt64, Valid: true},
		{Int64: 100, Valid: true},
		{Int64: 101, Valid: true},
	}
	for v := int64(-10); v <= 10; v++ {
		rows = append(rows, sql.NullInt64{Int64: v, Valid: true})
	}
	return rows
}

func mustRange(t *testing.T, low, high sqrl.RangeEndpoint) sqrl.Range {
	t.Helper()
	r, err := sqrl.RangeBetween(low, high)
	require.NoError(t, err)
	return r
}

// TestSQLiteInt64Ranges exercises NULL handling, int64 extremes, overlapping
// and nested ranges, holes, open/closed endpoints and bound-argument order on
// a real database.
func TestSQLiteInt64Ranges(t *testing.T) {
	db := openSQLite(t)
	insertInt64Rows(t, db, sampleNullInt64Rows())
	col := sqrl.MustRegisterRangeColumn("id")

	ranges := []sqrl.Range{
		// closed block [0,5]
		mustRange(t, sqrl.Int64Endpoint(0, true), sqrl.Int64Endpoint(5, true)),
		// overlapping open block (3,8): merges with the block above into [0,8)
		mustRange(t, sqrl.Int64Endpoint(3, false), sqrl.Int64Endpoint(8, false)),
		// isolated point
		sqrl.Int64Point(100),
		// minimum extreme, open upper bound; only MinInt64 is present
		mustRange(t, sqrl.Int64Endpoint(math.MinInt64, true),
			sqrl.Int64Endpoint(math.MinInt64+1, false)),
		// maximum extreme, open lower bound; only MaxInt64 is present
		mustRange(t, sqrl.Int64Endpoint(math.MaxInt64-1, false),
			sqrl.Int64Endpoint(math.MaxInt64, true)),
	}

	// Construction normalizes the two overlapping ranges away.
	set, err := sqrl.NewInt64RangeSet(col, sqrl.ExcludeNull, ranges...)
	require.NoError(t, err)
	assert.Equal(t, 4, set.Len(), "overlapping ranges must be merged")

	query, args, err := sqrl.Select("id").From("nums").
		Where(set.Contains()).ToSql()
	require.NoError(t, err)

	// Every finite endpoint is a bound arg in exact SQL order, and the extreme
	// values survive verbatim (no +/-1 endpoint rewriting).
	expectedArgs := []interface{}{
		int64(math.MinInt64), int64(math.MinInt64 + 1),
		int64(0), int64(8),
		int64(100),
		int64(math.MaxInt64 - 1), int64(math.MaxInt64),
	}
	assert.Equal(t, expectedArgs, args)
	assert.Equal(t, len(args), stringsCount(query, "?"))

	got := runInt64Query(t, db, query, args)
	wantValues := []int64{
		math.MinInt64,
		0, 1, 2, 3, 4, 5, 6, 7,
		100,
		math.MaxInt64,
	}
	assert.Equal(t, wantValues, got.values)
	assert.Equal(t, 0, got.nullRows, "ExcludeNull must not return NULL rows")

	// The same predicate with IncludeNull additionally returns the NULL row.
	incl, err := sqrl.NewInt64RangeSet(col, sqrl.IncludeNull, ranges...)
	require.NoError(t, err)
	query, args, err = sqrl.Select("id").From("nums").Where(incl.Contains()).ToSql()
	require.NoError(t, err)
	got = runInt64Query(t, db, query, args)
	assert.Equal(t, wantValues, got.values)
	assert.Equal(t, 1, got.nullRows, "IncludeNull must return the NULL row")
}

// TestSQLiteDifferenceHoles verifies that Difference splits a range around
// excluded points while preserving open endpoints, end to end.
func TestSQLiteDifferenceHoles(t *testing.T) {
	db := openSQLite(t)
	insertInt64Rows(t, db, sampleNullInt64Rows())
	col := sqrl.MustRegisterRangeColumn("id")

	a, err := sqrl.NewInt64RangeSet(col, sqrl.ExcludeNull,
		mustRange(t, sqrl.Int64Endpoint(-5, true), sqrl.Int64Endpoint(5, true)))
	require.NoError(t, err)
	b, err := sqrl.NewInt64RangeSet(col, sqrl.ExcludeNull,
		sqrl.Int64Point(-2), sqrl.Int64Point(0), sqrl.Int64Point(3))
	require.NoError(t, err)

	diff, err := a.Difference(b)
	require.NoError(t, err)
	assert.Equal(t, 4, diff.Len(), "three removed points split the range into four")

	query, args, err := sqrl.Select("id").From("nums").Where(diff.Contains()).ToSql()
	require.NoError(t, err)
	got := runInt64Query(t, db, query, args)
	assert.Equal(t, []int64{-5, -4, -3, -1, 1, 2, 4, 5}, got.values)
	assert.Equal(t, 0, got.nullRows)
}

// TestSQLiteEmptyAndUniversal covers the empty and fully unbounded sets under
// both NULL policies.
func TestSQLiteEmptyAndUniversal(t *testing.T) {
	db := openSQLite(t)
	rows := sampleNullInt64Rows()
	insertInt64Rows(t, db, rows)
	nonNullCount := 0
	for _, r := range rows {
		if r.Valid {
			nonNullCount++
		}
	}
	col := sqrl.MustRegisterRangeColumn("id")

	cases := []struct {
		name      string
		policy    sqrl.NullPolicy
		ranges    []sqrl.Range
		wantVals  int
		wantNulls int
	}{
		{"empty exclude", sqrl.ExcludeNull, nil, 0, 0},
		{"empty include", sqrl.IncludeNull, nil, 0, 1},
		{"universal exclude", sqrl.ExcludeNull, []sqrl.Range{sqrl.AllValuesRange()}, nonNullCount, 0},
		{"universal include", sqrl.IncludeNull, []sqrl.Range{sqrl.AllValuesRange()}, nonNullCount, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			set, err := sqrl.NewInt64RangeSet(col, tc.policy, tc.ranges...)
			require.NoError(t, err)
			query, args, err := sqrl.Select("id").From("nums").Where(set.Contains()).ToSql()
			require.NoError(t, err)
			got := runInt64Query(t, db, query, args)
			assert.Len(t, got.values, tc.wantVals)
			assert.Equal(t, tc.wantNulls, got.nullRows)
		})
	}
}

// TestSQLiteNormalizationReducesRanges proves that normalization removes
// redundant overlapping OR ranges without changing which rows match, by
// comparing against the OR of the same raw ranges built independently.
func TestSQLiteNormalizationReducesRanges(t *testing.T) {
	db := openSQLite(t)
	insertInt64Rows(t, db, sampleNullInt64Rows())
	col := sqrl.MustRegisterRangeColumn("id")

	raw := []sqrl.Range{
		mustRange(t, sqrl.Int64Endpoint(1, true), sqrl.Int64Endpoint(4, true)),
		mustRange(t, sqrl.Int64Endpoint(2, true), sqrl.Int64Endpoint(5, true)),
		mustRange(t, sqrl.Int64Endpoint(3, false), sqrl.Int64Endpoint(6, false)),
		mustRange(t, sqrl.Int64Endpoint(0, false), sqrl.Int64Endpoint(7, false)),
		mustRange(t, sqrl.Int64Endpoint(1, true), sqrl.Int64Endpoint(3, true)),
	}

	normalized, err := sqrl.NewInt64RangeSet(col, sqrl.ExcludeNull, raw...)
	require.NoError(t, err)
	assert.Equal(t, 1, normalized.Len(), "five overlapping ranges normalize to one")
	if normalized.Len() >= len(raw) {
		t.Fatalf("normalization did not reduce ranges: %d -> %d", len(raw), normalized.Len())
	}

	normQuery, normArgs, err := sqrl.Select("id").From("nums").
		Where(normalized.Contains()).ToSql()
	require.NoError(t, err)
	gotNormalized := runInt64Query(t, db, normQuery, normArgs)

	// Independent naive predicate: OR each raw range on its own.
	preds := make([]sqrl.Sqlizer, 0, len(raw))
	for _, r := range raw {
		single, err := sqrl.NewInt64RangeSet(col, sqrl.ExcludeNull, r)
		require.NoError(t, err)
		preds = append(preds, single.Contains())
	}
	naiveQuery, naiveArgs, err := sqrl.Select("id").From("nums").
		Where(sqrl.Or(preds)).ToSql()
	require.NoError(t, err)
	gotNaive := runInt64Query(t, db, naiveQuery, naiveArgs)

	assert.Equal(t, gotNaive.values, gotNormalized.values)
	assert.Equal(t, gotNaive.nullRows, gotNormalized.nullRows)
	// The normalized SQL binds strictly fewer arguments than the naive OR.
	if !(len(normArgs) < len(naiveArgs)) {
		t.Fatalf("normalized arg count %d is not below naive arg count %d",
			len(normArgs), len(naiveArgs))
	}
}

// TestSQLiteWhereComposition checks AND/OR parentheses against outer clauses.
func TestSQLiteWhereComposition(t *testing.T) {
	db := openSQLite(t)
	insertInt64Rows(t, db, sampleNullInt64Rows())
	col := sqrl.MustRegisterRangeColumn("id")

	set, err := sqrl.NewInt64RangeSet(col, sqrl.ExcludeNull,
		mustRange(t, sqrl.Int64Endpoint(0, true), sqrl.Int64Endpoint(3, true)))
	require.NoError(t, err)

	// (id in [0,3]) AND (id > 1) -> {2,3}
	andQuery, andArgs, err := sqrl.Select("id").From("nums").
		Where(set.Contains()).
		Where(sqrl.Gt{"id": 1}).
		ToSql()
	require.NoError(t, err)
	got := runInt64Query(t, db, andQuery, andArgs)
	assert.Equal(t, []int64{2, 3}, got.values)

	// (id in [0,3]) OR (id = 100) -> {0,1,2,3,100}
	orQuery, orArgs, err := sqrl.Select("id").From("nums").
		Where(sqrl.Or{set.Contains(), sqrl.Eq{"id": 100}}).
		ToSql()
	require.NoError(t, err)
	got = runInt64Query(t, db, orQuery, orArgs)
	assert.Equal(t, []int64{0, 1, 2, 3, 100}, got.values)
}

// TestSQLiteTimeBoundaries stores time.Time values (encoded as TEXT by the
// driver) and verifies second-precision normalization plus open/closed time
// boundaries on real queries.
func TestSQLiteTimeBoundaries(t *testing.T) {
	db := openSQLite(t)
	col := sqrl.MustRegisterRangeColumn("ts")
	base := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

	inserted := []time.Time{}
	for i := 0; i <= 5; i++ {
		inserted = append(inserted, base.Add(time.Duration(i)*time.Second))
	}
	for _, ts := range inserted {
		_, err := db.Exec("INSERT INTO events (ts) VALUES (?)", ts)
		require.NoError(t, err)
	}
	_, err := db.Exec("INSERT INTO events (ts) VALUES (?)", base.Add(10*time.Second))
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO events (ts) VALUES (NULL)")
	require.NoError(t, err)

	ranges := []sqrl.Range{
		// Sub-second endpoints truncate to [base+1s, base+3s] closed.
		mustRange(t,
			sqrl.TimeEndpoint(base.Add(time.Second+500*time.Millisecond), true),
			sqrl.TimeEndpoint(base.Add(3*time.Second+200*time.Millisecond), true)),
		// Open block (base+3s, base+5s): the shared boundary second stays
		// included only through the closed range above.
		mustRange(t,
			sqrl.TimeEndpoint(base.Add(3*time.Second), false),
			sqrl.TimeEndpoint(base.Add(5*time.Second+900*time.Millisecond), false)),
	}

	set, err := sqrl.NewTimeRangeSet(col, time.Second, sqrl.ExcludeNull, ranges...)
	require.NoError(t, err)
	// The closed high of the first range covers the shared boundary second, so
	// the two ranges normalize to one [base+1s, base+5s) interval.
	require.Equal(t, 1, set.Len())

	query, args, err := sqrl.Select("ts").From("events").
		Where(set.Contains()).ToSql()
	require.NoError(t, err)

	// Bound args are time.Time values already truncated to whole seconds.
	require.Equal(t, 2, len(args))
	firstArg, ok := args[0].(time.Time)
	require.True(t, ok)
	assert.True(t, firstArg.Equal(base.Add(time.Second)))
	assert.Equal(t, 0, firstArg.Nanosecond())
	lastArg, ok := args[1].(time.Time)
	require.True(t, ok)
	assert.True(t, lastArg.Equal(base.Add(5*time.Second)),
		"sub-second upper endpoint must truncate to the whole second")
	assert.Equal(t, 0, lastArg.Nanosecond())

	rows, err := db.Query(query, args...)
	require.NoError(t, err)
	defer rows.Close()
	var got []string
	for rows.Next() {
		var s string
		require.NoError(t, rows.Scan(&s))
		got = append(got, s)
	}
	require.NoError(t, rows.Err())
	sort.Strings(got)

	format := "2006-01-02 15:04:05-07:00"
	want := []string{
		base.Add(1 * time.Second).Format(format),
		base.Add(2 * time.Second).Format(format),
		base.Add(3 * time.Second).Format(format),
		base.Add(4 * time.Second).Format(format),
	}
	assert.Equal(t, want, got)

	// IncludeNull surfaces the NULL row.
	incl, err := sqrl.NewTimeRangeSet(col, time.Second, sqrl.IncludeNull, ranges...)
	require.NoError(t, err)
	query, args, err = sqrl.Select("ts").From("events").Where(incl.Contains()).ToSql()
	require.NoError(t, err)
	rows2, err := db.Query(query, args...)
	require.NoError(t, err)
	defer rows2.Close()
	nulls, nonNulls := 0, 0
	for rows2.Next() {
		var s sql.NullString
		require.NoError(t, rows2.Scan(&s))
		if s.Valid {
			nonNulls++
		} else {
			nulls++
		}
	}
	require.NoError(t, rows2.Err())
	assert.Equal(t, 1, nulls)
	assert.Equal(t, 4, nonNulls)
}

func stringsCount(s, sub string) int {
	count := 0
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			count++
		}
	}
	return count
}

// TestSQLiteRejectsUnregisteredColumn documents the safe identifier protocol:
// an unvalidated string cannot be turned into a column handle.
func TestSQLiteRejectsUnregisteredColumn(t *testing.T) {
	for _, name := range []string{"id; DROP TABLE nums", "id--", "a.b.c;"} {
		_, err := sqrl.RegisterRangeColumn(name)
		assert.Error(t, err, fmt.Sprintf("identifier %q should be rejected", name))
	}
}
