package squirrel

import (
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func seekTestKeys() []SeekKey {
	return []SeekKey{
		{Expr: "a", Order: SeekAsc, Nulls: SeekNullsLast},
		{Expr: "id", Order: SeekAsc, Nulls: SeekNullsLast, Unique: true},
	}
}

func TestSeekWindowValidation(t *testing.T) {
	base := func() SelectBuilder { return Select("id", "a").From("t") }

	cases := []struct {
		name     string
		builder  SelectBuilder
		keys     []SeekKey
		pageSize uint64
	}{
		{"zero page size", base(), seekTestKeys(), 0},
		{"page size over cap", base(), seekTestKeys(), MaxSeekPageSize + 1},
		{"no keys", base(), nil, 10},
		{
			"empty expr",
			base(),
			[]SeekKey{{Expr: "  "}, {Expr: "id", Unique: true}},
			10,
		},
		{
			"statement in expr",
			base(),
			[]SeekKey{{Expr: "a; DROP TABLE t"}, {Expr: "id", Unique: true}},
			10,
		},
		{
			"invalid order",
			base(),
			[]SeekKey{{Expr: "a", Order: SeekOrder(42), Nulls: SeekNullsLast}, {Expr: "id", Unique: true}},
			10,
		},
		{
			"invalid nulls",
			base(),
			[]SeekKey{{Expr: "a", Order: SeekAsc, Nulls: SeekNulls(7)}, {Expr: "id", Unique: true}},
			10,
		},
		{
			"duplicate expr",
			base(),
			[]SeekKey{
				{Expr: "a", Order: SeekAsc, Nulls: SeekNullsLast},
				{Expr: "a", Order: SeekDesc, Nulls: SeekNullsFirst},
				{Expr: "id", Unique: true},
			},
			10,
		},
		{
			"unique not last",
			base(),
			[]SeekKey{
				{Expr: "id", Order: SeekAsc, Nulls: SeekNullsLast, Unique: true},
				{Expr: "a", Order: SeekAsc, Nulls: SeekNullsLast},
			},
			10,
		},
		{
			"missing unique",
			base(),
			[]SeekKey{{Expr: "a", Order: SeekAsc, Nulls: SeekNullsLast}},
			10,
		},
		{"distinct", base().Distinct(), seekTestKeys(), 10},
		{"group by", base().GroupBy("a"), seekTestKeys(), 10},
		{"having", base().Having("count(a) > 1"), seekTestKeys(), 10},
		{"raw order by", base().OrderBy("a ASC"), seekTestKeys(), 10},
		{"existing offset", base().Offset(20), seekTestKeys(), 10},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.builder.SeekWindow(tc.keys, tc.pageSize)
			assert.Error(t, err)
		})
	}
}

func TestSeekWindowAccessors(t *testing.T) {
	b := Select("id", "a").From("t")
	w, err := b.SeekWindow(seekTestKeys(), 10)
	assert.NoError(t, err)
	assert.Equal(t, uint64(10), w.PageSize())
	assert.Equal(t, seekTestKeys(), w.Keys())

	// Mutating the returned key slice must not affect the window.
	keys := w.Keys()
	keys[0].Expr = "tampered"
	assert.Equal(t, "a", w.Keys()[0].Expr)

	// Mutating the input slice must not affect the window either.
	input := seekTestKeys()
	w2, err := b.SeekWindow(input, 5)
	assert.NoError(t, err)
	input[0].Expr = "tampered"
	assert.Equal(t, "a", w2.Keys()[0].Expr)
}

func TestSeekForwardPredicateSqlAndArgs(t *testing.T) {
	b := Select("id", "a").From("t").Where("g = ?", "x")
	w, err := b.SeekWindow(seekTestKeys(), 10)
	assert.NoError(t, err)

	q, err := w.Page([]interface{}{5, 9}, SeekAfter)
	assert.NoError(t, err)
	assert.Equal(t, SeekAfter, q.Direction())

	sql, args, err := q.ToSql()
	assert.NoError(t, err)

	expectedSql := "SELECT id, a FROM t WHERE g = ? AND " +
		"((((a > ?) OR (a IS NULL))) OR ((a = ?) AND ((id > ?) OR (id IS NULL)))) " +
		"ORDER BY a ASC NULLS LAST, id ASC NULLS LAST LIMIT 10"
	assert.Equal(t, expectedSql, sql)
	assert.Equal(t, []interface{}{"x", 5, 5, 9}, args)
	assert.NotContains(t, sql, "OFFSET")
}

func TestSeekNullCursorAndMixedDirections(t *testing.T) {
	keys := []SeekKey{
		{Expr: "b", Order: SeekDesc, Nulls: SeekNullsFirst},
		{Expr: "id", Order: SeekAsc, Nulls: SeekNullsLast, Unique: true},
	}
	w, err := Select("id", "b").From("t").SeekWindow(keys, 10)
	assert.NoError(t, err)

	// NULL leading cursor value: IS NOT NULL branch for NULLS FIRST.
	q, err := w.Page([]interface{}{nil, 7}, SeekAfter)
	assert.NoError(t, err)
	sql, args, err := q.ToSql()
	assert.NoError(t, err)
	expectedSql := "SELECT id, b FROM t WHERE " +
		"(((b IS NOT NULL)) OR ((b IS NULL) AND ((id > ?) OR (id IS NULL)))) " +
		"ORDER BY b DESC NULLS FIRST, id ASC NULLS LAST LIMIT 10"
	assert.Equal(t, expectedSql, sql)
	assert.Equal(t, []interface{}{7}, args)

	// NULL cursor value with NULLS LAST produces the impossible (1=0) branch.
	nullLastKeys := []SeekKey{
		{Expr: "a", Order: SeekAsc, Nulls: SeekNullsLast},
		{Expr: "id", Order: SeekAsc, Nulls: SeekNullsLast, Unique: true},
	}
	w2, err := Select("id", "a").From("t").SeekWindow(nullLastKeys, 10)
	assert.NoError(t, err)
	q2, err := w2.Page([]interface{}{nil, 5}, SeekAfter)
	assert.NoError(t, err)
	sql2, args2, err := q2.ToSql()
	assert.NoError(t, err)
	expectedSql2 := "SELECT id, a FROM t WHERE " +
		"(((1=0)) OR ((a IS NULL) AND ((id > ?) OR (id IS NULL)))) " +
		"ORDER BY a ASC NULLS LAST, id ASC NULLS LAST LIMIT 10"
	assert.Equal(t, expectedSql2, sql2)
	assert.Equal(t, []interface{}{5}, args2)
}

func TestSeekBackwardFlipsSortAndOperator(t *testing.T) {
	keys := []SeekKey{
		{Expr: "b", Order: SeekDesc, Nulls: SeekNullsFirst},
		{Expr: "id", Order: SeekAsc, Nulls: SeekNullsLast, Unique: true},
	}
	w, err := Select("id", "b").From("t").SeekWindow(keys, 10)
	assert.NoError(t, err)

	q, err := w.Page([]interface{}{3, 7}, SeekBefore)
	assert.NoError(t, err)
	assert.Equal(t, SeekBefore, q.Direction())

	sql, args, err := q.ToSql()
	assert.NoError(t, err)
	expectedSql := "SELECT id, b FROM t WHERE " +
		"((((b > ?) OR (b IS NULL))) OR ((b = ?) AND (id < ?))) " +
		"ORDER BY b ASC NULLS LAST, id DESC NULLS FIRST LIMIT 10"
	assert.Equal(t, expectedSql, sql)
	assert.Equal(t, []interface{}{3, 3, 7}, args)
}

func TestSeekNilCursorIsBoundaryPage(t *testing.T) {
	keys := []SeekKey{
		{Expr: "b", Order: SeekDesc, Nulls: SeekNullsFirst},
		{Expr: "id", Order: SeekAsc, Nulls: SeekNullsLast, Unique: true},
	}
	w, err := Select("id", "b").From("t").SeekWindow(keys, 10)
	assert.NoError(t, err)

	first, err := w.Page(nil, SeekAfter)
	assert.NoError(t, err)
	sql, _, err := first.ToSql()
	assert.NoError(t, err)
	assert.Equal(t,
		"SELECT id, b FROM t ORDER BY b DESC NULLS FIRST, id ASC NULLS LAST LIMIT 10", sql)

	last, err := w.Page(nil, SeekBefore)
	assert.NoError(t, err)
	sql, _, err = last.ToSql()
	assert.NoError(t, err)
	assert.Equal(t,
		"SELECT id, b FROM t ORDER BY b ASC NULLS LAST, id DESC NULLS FIRST LIMIT 10", sql)
}

func TestSeekCursorValidation(t *testing.T) {
	w, err := Select("id", "a").From("t").SeekWindow(seekTestKeys(), 10)
	assert.NoError(t, err)

	_, err = w.Page([]interface{}{1}, SeekAfter)
	assert.Error(t, err)

	// All-NULL cursor is rejected because the unique tie-breaker is NULL.
	_, err = w.Page([]interface{}{nil, nil}, SeekAfter)
	assert.Error(t, err)

	// A NULL tie-breaker alongside a non-NULL leading key is rejected too.
	_, err = w.Page([]interface{}{1, nil}, SeekAfter)
	assert.Error(t, err)

	_, err = w.Page([]interface{}{1, 2}, SeekDirection(99))
	assert.Error(t, err)
}

func TestSeekOriginalBuilderImmutable(t *testing.T) {
	b := Select("id", "a").From("t").Where("g = ?", "x")
	originalSql, originalArgs, err := b.ToSql()
	assert.NoError(t, err)

	w, err := b.SeekWindow(seekTestKeys(), 10)
	assert.NoError(t, err)

	_, err = w.Page([]interface{}{1, 2}, SeekAfter)
	assert.NoError(t, err)
	_, err = w.Page([]interface{}{3, 4}, SeekBefore)
	assert.NoError(t, err)
	_, err = w.Page(nil, SeekAfter)
	assert.NoError(t, err)

	sql, args, err := b.ToSql()
	assert.NoError(t, err)
	assert.Equal(t, originalSql, sql)
	assert.Equal(t, originalArgs, args)
}

func TestSeekReplacesExistingLimitAndNeverOffsets(t *testing.T) {
	b := Select("id", "a").From("t").Limit(99)
	w, err := b.SeekWindow(seekTestKeys(), 10)
	assert.NoError(t, err)

	q, err := w.Page(nil, SeekAfter)
	assert.NoError(t, err)
	sql, _, err := q.ToSql()
	assert.NoError(t, err)
	assert.Contains(t, sql, "LIMIT 10")
	assert.NotContains(t, sql, "LIMIT 99")
	assert.NotContains(t, sql, "OFFSET")
}

func TestSeekPrefixSuffixArgsFollowSqlOrder(t *testing.T) {
	keys := []SeekKey{
		{Expr: "id", Order: SeekAsc, Nulls: SeekNullsFirst, Unique: true},
	}
	b := Select("a").From("t").
		Prefix("WITH p AS (SELECT ?)", 0).
		Where("g = ?", "x").
		Suffix("FETCH FIRST ? ROWS ONLY", 99)
	w, err := b.SeekWindow(keys, 10)
	assert.NoError(t, err)

	q, err := w.Page([]interface{}{5}, SeekAfter)
	assert.NoError(t, err)
	sql, args, err := q.ToSql()
	assert.NoError(t, err)

	assert.True(t, strings.HasPrefix(sql, "WITH p AS (SELECT ?)"))
	assert.Contains(t, sql, "WHERE g = ? AND (((id > ?)))")
	assert.True(t, strings.HasSuffix(sql, "FETCH FIRST ? ROWS ONLY"))
	// Prefix args first, original WHERE next, seek predicate next, suffix last.
	assert.Equal(t, []interface{}{0, "x", 5, 99}, args)
}

func TestSeekDollarPlaceholdersRenumbered(t *testing.T) {
	b := Select("id", "a").From("t").Where("g = ?", "x").PlaceholderFormat(Dollar)
	w, err := b.SeekWindow(seekTestKeys(), 10)
	assert.NoError(t, err)

	q, err := w.Page([]interface{}{5, 9}, SeekAfter)
	assert.NoError(t, err)
	sql, args, err := q.ToSql()
	assert.NoError(t, err)

	assert.Contains(t, sql, "g = $1")
	assert.Contains(t, sql, "a > $2")
	assert.Contains(t, sql, "a IS NULL")
	assert.Contains(t, sql, "a = $3")
	assert.Contains(t, sql, "id > $4")
	assert.NotContains(t, sql, "?")
	assert.Equal(t, []interface{}{"x", 5, 5, 9}, args)
}

type failingSqlizer struct{}

func (failingSqlizer) ToSql() (string, []interface{}, error) {
	return "", nil, errors.New("custom sqlizer boom")
}

func TestSeekPropagatesCustomSqlizerErrors(t *testing.T) {
	keys := []SeekKey{
		{Expr: "id", Order: SeekAsc, Nulls: SeekNullsFirst, Unique: true},
	}

	// Error in an existing WHERE part.
	w, err := Select("a").From("t").Where(failingSqlizer{}).SeekWindow(keys, 10)
	assert.NoError(t, err)
	q, err := w.Page([]interface{}{1}, SeekAfter)
	assert.NoError(t, err)
	_, _, err = q.ToSql()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "custom sqlizer boom")

	// Error in a PREFIX part surfaces through the same path.
	w2, err := Select("a").From("t").PrefixExpr(failingSqlizer{}).SeekWindow(keys, 10)
	assert.NoError(t, err)
	q2, err := w2.Page(nil, SeekAfter)
	assert.NoError(t, err)
	_, _, err = q2.ToSql()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "custom sqlizer boom")
}

func TestSeekQueryWithoutRunner(t *testing.T) {
	w, err := Select("a").From("t").
		SeekWindow([]SeekKey{{Expr: "id", Order: SeekAsc, Nulls: SeekNullsFirst, Unique: true}}, 10)
	assert.NoError(t, err)

	q, err := w.Page(nil, SeekAfter)
	assert.NoError(t, err)
	_, err = q.Query()
	assert.Equal(t, RunnerNotSet, err)
}
