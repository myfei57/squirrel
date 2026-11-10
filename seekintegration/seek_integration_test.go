package seekintegration

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"

	sqrl "github.com/Masterminds/squirrel"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

// Column layout used by every test query:
//
//	0: id, 1: grp, 2: score, 3: label, 4: cscore (only when selected)
const schema = `
CREATE TABLE items (
	id    INTEGER PRIMARY KEY,
	grp   TEXT NOT NULL,
	score INTEGER,
	label TEXT
);`

// Deliberately messy data: repeated scores, repeated labels, NULLs in both
// nullable columns and two groups, so every NULL/direction/tie-break branch
// is exercised against a real engine.
const data = `
INSERT INTO items (id, grp, score, label) VALUES
	(1,  'a', 10,   'alpha'),
	(2,  'a', 10,   'beta'),
	(3,  'a', 10,   NULL),
	(4,  'b', 20,   'alpha'),
	(5,  'b', 20,   NULL),
	(6,  'a', NULL, 'gamma'),
	(7,  'b', NULL, 'gamma'),
	(8,  'a', NULL, NULL),
	(9,  'b', 5,    'beta'),
	(10, 'a', 5,    'alpha'),
	(11, 'b', 30,   NULL),
	(12, 'a', 10,   'beta'),
	(13, 'b', NULL, 'alpha'),
	(14, 'a', 20,   'delta'),
	(15, 'b', 5,    NULL),
	(16, 'a', NULL, 'beta');
`

var db *sql.DB

func TestMain(m *testing.M) {
	var err error
	db, err = sql.Open("sqlite", ":memory:")
	if err != nil {
		fmt.Println("open:", err)
		os.Exit(-1)
	}
	// A single connection keeps the in-memory schema visible to every query.
	db.SetMaxOpenConns(1)
	if _, err = db.Exec(schema); err != nil {
		fmt.Println("schema:", err)
		os.Exit(-2)
	}
	if _, err = db.Exec(data); err != nil {
		fmt.Println("data:", err)
		os.Exit(-3)
	}
	os.Exit(m.Run())
}

// referenceIDs runs the full, un-paged ORDER BY query and returns the
// canonical id sequence every seek walk must reproduce.
func referenceIDs(t *testing.T, orderBy, where string, args ...interface{}) []int64 {
	t.Helper()
	query := "SELECT id FROM items"
	if where != "" {
		query += " WHERE " + where
	}
	query += " ORDER BY " + orderBy
	rows, err := db.Query(query, args...)
	require.NoError(t, err)
	defer rows.Close()

	var out []int64
	for rows.Next() {
		var id int64
		require.NoError(t, rows.Scan(&id))
		out = append(out, id)
	}
	require.NoError(t, rows.Err())
	return out
}

func tuple(row []interface{}, colIdx []int) []interface{} {
	out := make([]interface{}, len(colIdx))
	for i, idx := range colIdx {
		out[i] = row[idx]
	}
	return out
}

func flatten(pages [][]int64) []int64 {
	var out []int64
	for _, page := range pages {
		out = append(out, page...)
	}
	return out
}

func reverse(ids []int64) []int64 {
	out := make([]int64, len(ids))
	for i, id := range ids {
		out[len(ids)-1-i] = id
	}
	return out
}

// expectedBackwardPages independently computes what a SeekBefore walk must
// return: the reference sequence carved into page-size chunks from the tail,
// every chunk kept in forward display order.
func expectedBackwardPages(ref []int64, size int) [][]int64 {
	var pages [][]int64
	for end := len(ref); end > 0; {
		start := end - size
		if start < 0 {
			start = 0
		}
		pages = append(pages, append([]int64(nil), ref[start:end]...))
		end = start
	}
	return pages
}

// walk pages through the whole result set. Forward walks seed each next page
// with the last row, backward walks with the first row; both terminate on an
// empty page.
func walk(t *testing.T, w sqrl.SeekWindow, keyIdx []int, dir sqrl.SeekDirection) [][]int64 {
	t.Helper()
	var pages [][]int64
	var cursor []interface{}
	for {
		q, err := w.Page(cursor, dir)
		require.NoError(t, err)
		page, err := q.Query()
		require.NoError(t, err)
		if page.Len() == 0 {
			return pages
		}
		ids := make([]int64, page.Len())
		for i, row := range page.Rows {
			ids[i] = row[0].(int64)
		}
		pages = append(pages, ids)

		if dir == sqrl.SeekAfter {
			cursor = tuple(page.Rows[page.Len()-1], keyIdx)
		} else {
			cursor = tuple(page.Rows[0], keyIdx)
		}
	}
}

func assertNoDuplicates(t *testing.T, ids []int64) {
	t.Helper()
	seen := make(map[int64]struct{}, len(ids))
	for _, id := range ids {
		_, dup := seen[id]
		assert.False(t, dup, "duplicate id %d across pages", id)
		seen[id] = struct{}{}
	}
}

func assertPagesInReferenceOrder(t *testing.T, pages [][]int64, ref []int64) {
	t.Helper()
	pos := make(map[int64]int, len(ref))
	for i, id := range ref {
		pos[id] = i
	}
	// Every individual page must read in forward display order, including
	// pages fetched SeekBefore (which arrive reversed at the SQL level).
	for _, page := range pages {
		for i := 1; i < len(page); i++ {
			assert.Less(t, pos[page[i-1]], pos[page[i]],
				"page %v is not in reference order", page)
		}
	}
}

func TestSeekForwardBackwardDuplicatesAndNulls(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast},
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	b := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label").From("items")
	w, err := b.SeekWindow(keys, 3)
	require.NoError(t, err)

	ref := referenceIDs(t, "score ASC NULLS LAST, id ASC NULLS LAST", "")

	// Forward: no repeats, no misses, exact reference order.
	forwardPages := walk(t, w, []int{2, 0}, sqrl.SeekAfter)
	forward := flatten(forwardPages)
	assert.Equal(t, ref, forward)
	assertNoDuplicates(t, forward)
	assert.Len(t, forwardPages, 6) // 16 rows at page size 3: five full + one partial

	// Backward: tail-to-head pages, each restored to forward display order.
	backwardPages := walk(t, w, []int{2, 0}, sqrl.SeekBefore)
	assert.Equal(t, expectedBackwardPages(ref, 3), backwardPages)
	assertNoDuplicates(t, flatten(backwardPages))
	assertPagesInReferenceOrder(t, backwardPages, ref)

	// Boundary pages match the head and tail of the reference ordering.
	first, err := w.Page(nil, sqrl.SeekAfter)
	require.NoError(t, err)
	firstPage, err := first.Query()
	require.NoError(t, err)
	require.Len(t, firstPage.Rows, 3)
	assert.Equal(t, ref[:3], []int64{
		firstPage.Rows[0][0].(int64),
		firstPage.Rows[1][0].(int64),
		firstPage.Rows[2][0].(int64),
	})

	last, err := w.Page(nil, sqrl.SeekBefore)
	require.NoError(t, err)
	lastPage, err := last.Query()
	require.NoError(t, err)
	require.Len(t, lastPage.Rows, 3)
	assert.Equal(t, ref[len(ref)-3:], []int64{
		lastPage.Rows[0][0].(int64),
		lastPage.Rows[1][0].(int64),
		lastPage.Rows[2][0].(int64),
	})
}

func TestSeekMixedDirectionsAndNullsFirst(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "label", Order: sqrl.SeekDesc, Nulls: sqrl.SeekNullsFirst},
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsFirst},
		{Expr: "id", Order: sqrl.SeekDesc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	b := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label").From("items")
	w, err := b.SeekWindow(keys, 4)
	require.NoError(t, err)

	ref := referenceIDs(t, "label DESC NULLS FIRST, score ASC NULLS FIRST, id DESC NULLS LAST", "")

	forward := flatten(walk(t, w, []int{3, 2, 0}, sqrl.SeekAfter))
	assert.Equal(t, ref, forward)
	assertNoDuplicates(t, forward)

	backwardPages := walk(t, w, []int{3, 2, 0}, sqrl.SeekBefore)
	assert.Equal(t, expectedBackwardPages(ref, 4), backwardPages)
	assertNoDuplicates(t, flatten(backwardPages))
}

func TestSeekPureExpressionKey(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "COALESCE(score, -1)", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast},
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	b := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label", "COALESCE(score, -1) AS cscore").From("items")
	w, err := b.SeekWindow(keys, 5)
	require.NoError(t, err)

	ref := referenceIDs(t, "COALESCE(score, -1) ASC NULLS LAST, id ASC NULLS LAST", "")

	forward := flatten(walk(t, w, []int{4, 0}, sqrl.SeekAfter))
	assert.Equal(t, ref, forward)
	assertNoDuplicates(t, forward)

	backwardPages := walk(t, w, []int{4, 0}, sqrl.SeekBefore)
	assert.Equal(t, expectedBackwardPages(ref, 5), backwardPages)
	assertNoDuplicates(t, flatten(backwardPages))
}

func TestSeekWithOriginalWhereFilter(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsFirst},
		{Expr: "id", Order: sqrl.SeekDesc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	b := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label").From("items").
		Where("grp = ?", "a")
	w, err := b.SeekWindow(keys, 2)
	require.NoError(t, err)

	ref := referenceIDs(t, "score ASC NULLS FIRST, id DESC NULLS LAST", "grp = ?", "a")

	forward := flatten(walk(t, w, []int{2, 0}, sqrl.SeekAfter))
	assert.Equal(t, ref, forward)
	assertNoDuplicates(t, forward)

	backwardPages := walk(t, w, []int{2, 0}, sqrl.SeekBefore)
	assert.Equal(t, expectedBackwardPages(ref, 2), backwardPages)
	assertNoDuplicates(t, flatten(backwardPages))
}

func TestSeekBoundarySymmetry(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast},
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	w, err := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label").From("items").
		SeekWindow(keys, 3)
	require.NoError(t, err)

	p1Query, err := w.Page(nil, sqrl.SeekAfter)
	require.NoError(t, err)
	p1, err := p1Query.Query()
	require.NoError(t, err)
	require.Len(t, p1.Rows, 3)

	p2Query, err := w.Page(tuple(p1.Rows[2], []int{2, 0}), sqrl.SeekAfter)
	require.NoError(t, err)
	p2, err := p2Query.Query()
	require.NoError(t, err)
	require.Len(t, p2.Rows, 3)

	// Going back from the first row of page 2 reproduces page 1 exactly.
	prevQuery, err := w.Page(tuple(p2.Rows[0], []int{2, 0}), sqrl.SeekBefore)
	require.NoError(t, err)
	prev, err := prevQuery.Query()
	require.NoError(t, err)
	assert.Equal(t, pageIDs(p1), pageIDs(prev))

	// And going forward from the last row of page 1 reproduces page 2.
	nextQuery, err := w.Page(tuple(p1.Rows[2], []int{2, 0}), sqrl.SeekAfter)
	require.NoError(t, err)
	next, err := nextQuery.Query()
	require.NoError(t, err)
	assert.Equal(t, pageIDs(p2), pageIDs(next))
}

func TestSeekEmptyPages(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast},
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	w, err := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label").From("items").
		SeekWindow(keys, 3)
	require.NoError(t, err)

	// With NULLS LAST, NULL scores sort AFTER every non-NULL cursor value:
	// a cursor beyond the largest non-NULL score still returns the NULL rows,
	// which is exactly why the NULL-aware disjunction is required.
	nullsAfter, err := w.Page([]interface{}{int64(9999), int64(999999)}, sqrl.SeekAfter)
	require.NoError(t, err)
	page, err := nullsAfter.Query()
	require.NoError(t, err)
	// Five NULL-score rows exist; the page size caps this first page at three.
	require.Equal(t, 3, page.Len())
	for _, row := range page.Rows {
		assert.Nil(t, row[2])
	}

	// The true end is the last NULL row itself: nothing sorts after it.
	afterEnd, err := w.Page([]interface{}{nil, int64(16)}, sqrl.SeekAfter)
	require.NoError(t, err)
	page, err = afterEnd.Query()
	require.NoError(t, err)
	assert.Equal(t, 0, page.Len())

	// Before the smallest value there is nothing; NULLS FIRST on the flipped
	// key keeps NULL rows out of the backward branch.
	beforeStart, err := w.Page([]interface{}{int64(-9999), int64(-1)}, sqrl.SeekBefore)
	require.NoError(t, err)
	page, err = beforeStart.Query()
	require.NoError(t, err)
	assert.Equal(t, 0, page.Len())
}

func TestSeekPageSizeOneStress(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	w, err := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label").From("items").
		SeekWindow(keys, 1)
	require.NoError(t, err)

	ref := referenceIDs(t, "id ASC NULLS LAST", "")
	forward := flatten(walk(t, w, []int{0}, sqrl.SeekAfter))
	assert.Equal(t, ref, forward)
	assert.Len(t, forward, 16)

	backwardPages := walk(t, w, []int{0}, sqrl.SeekBefore)
	assert.Equal(t, expectedBackwardPages(ref, 1), backwardPages)
	assert.Equal(t, reverse(ref), flatten(backwardPages))
}

func TestSeekRejections(t *testing.T) {
	base := func() sqrl.SelectBuilder {
		return sqrl.StatementBuilder.RunWith(db).
			Select("id", "grp", "score", "label").From("items")
	}
	keys := []sqrl.SeekKey{
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast},
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}

	// Non-unique sort: no tie-breaker declared.
	_, err := base().SeekWindow([]sqrl.SeekKey{
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast},
	}, 3)
	assert.Error(t, err)

	// DISTINCT, GROUP BY, OFFSET are refused rather than silently adapted.
	_, err = base().Distinct().SeekWindow(keys, 3)
	assert.Error(t, err)
	_, err = base().GroupBy("grp").SeekWindow(keys, 3)
	assert.Error(t, err)
	_, err = base().Offset(5).SeekWindow(keys, 3)
	assert.Error(t, err)

	// Page size bounds.
	_, err = base().SeekWindow(keys, 0)
	assert.Error(t, err)
	_, err = base().SeekWindow(keys, sqrl.MaxSeekPageSize+1)
	assert.Error(t, err)

	w, err := base().SeekWindow(keys, 3)
	require.NoError(t, err)

	// Cursor arity and enum checks.
	_, err = w.Page([]interface{}{int64(1)}, sqrl.SeekAfter)
	assert.Error(t, err)
	_, err = w.Page([]interface{}{int64(1), int64(2)}, sqrl.SeekDirection(42))
	assert.Error(t, err)

	// All-NULL cursor: the NULL unique tie-breaker cannot define a total order.
	_, err = w.Page([]interface{}{nil, nil}, sqrl.SeekAfter)
	assert.Error(t, err)
}

type failingSqlizer struct{}

func (failingSqlizer) ToSql() (string, []interface{}, error) {
	return "", nil, errors.New("integration boom")
}

func TestSeekCustomSqlizerErrorSurfacesAtQuery(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsFirst, Unique: true},
	}
	w, err := sqrl.StatementBuilder.RunWith(db).
		Select("id").From("items").Where(failingSqlizer{}).
		SeekWindow(keys, 3)
	require.NoError(t, err)

	q, err := w.Page([]interface{}{int64(1)}, sqrl.SeekAfter)
	require.NoError(t, err)
	_, err = q.Query()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "integration boom")
}

// recordingRunner captures the exact SQL and bound args sent to the driver.
type recordingRunner struct {
	db        *sql.DB
	lastQuery string
	lastArgs  []interface{}
}

func (r *recordingRunner) Exec(query string, args ...interface{}) (sql.Result, error) {
	return r.db.Exec(query, args...)
}

func (r *recordingRunner) Query(query string, args ...interface{}) (*sql.Rows, error) {
	r.lastQuery = query
	r.lastArgs = args
	return r.db.Query(query, args...)
}

func TestSeekParameterBindingAndNoOffset(t *testing.T) {
	rec := &recordingRunner{db: db}
	keys := []sqrl.SeekKey{
		{Expr: "score", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast},
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	w, err := sqrl.StatementBuilder.RunWith(rec).
		Select("id", "grp", "score", "label").From("items").
		Where("grp = ?", "a").
		SeekWindow(keys, 3)
	require.NoError(t, err)

	q, err := w.Page([]interface{}{int64(10), int64(3)}, sqrl.SeekAfter)
	require.NoError(t, err)
	page, err := q.Query()
	require.NoError(t, err)
	require.Greater(t, page.Len(), 0)

	assert.NotContains(t, rec.lastQuery, "OFFSET")
	assert.Contains(t, rec.lastQuery, "LIMIT 3")
	assert.Contains(t, rec.lastQuery, "grp = ?")
	// Every placeholder has a bound arg in final SQL order.
	assert.Equal(t, strings.Count(rec.lastQuery, "?"), len(rec.lastArgs))
	assert.Equal(t, []interface{}{"a", int64(10), int64(10), int64(3)}, rec.lastArgs)
}

func TestSeekQueryContextReversesBackwardPage(t *testing.T) {
	keys := []sqrl.SeekKey{
		{Expr: "id", Order: sqrl.SeekAsc, Nulls: sqrl.SeekNullsLast, Unique: true},
	}
	w, err := sqrl.StatementBuilder.RunWith(db).
		Select("id", "grp", "score", "label").From("items").
		SeekWindow(keys, 5)
	require.NoError(t, err)

	// Last page via the context entry point.
	q, err := w.Page(nil, sqrl.SeekBefore)
	require.NoError(t, err)
	page, err := q.QueryContext(context.Background())
	require.NoError(t, err)
	require.Len(t, page.Rows, 5)

	// Rows must be restored to ascending display order.
	ids := pageIDs(page)
	for i := 1; i < len(ids); i++ {
		assert.Less(t, ids[i-1], ids[i])
	}
	assert.Equal(t, []int64{12, 13, 14, 15, 16}, ids)
}

func pageIDs(page *sqrl.SeekPage) []int64 {
	ids := make([]int64, page.Len())
	for i, row := range page.Rows {
		ids[i] = row[0].(int64)
	}
	return ids
}
