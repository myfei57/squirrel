package squirrel

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/lann/builder"
)

// MaxSeekPageSize is the largest page size a SeekWindow will accept. It
// guards against an accidental full-table scan caused by a mistaken limit.
const MaxSeekPageSize = 1000

// SeekOrder is the sort direction of a SeekKey.
type SeekOrder int

const (
	// SeekAsc sorts the key ascending: smaller values come first.
	SeekAsc SeekOrder = iota
	// SeekDesc sorts the key descending: larger values come first.
	SeekDesc
)

// SeekNulls positions NULL values relative to non-NULL values for a key.
type SeekNulls int

const (
	// SeekNullsFirst places NULL values before every non-NULL value.
	SeekNullsFirst SeekNulls = iota
	// SeekNullsLast places NULL values after every non-NULL value.
	SeekNullsLast
)

// SeekDirection selects which side of the cursor a page is taken from.
type SeekDirection int

const (
	// SeekAfter returns the rows that sort strictly after the cursor (next
	// page, forward traversal).
	SeekAfter SeekDirection = iota
	// SeekBefore returns the rows that sort strictly before the cursor
	// (previous page, backward traversal). The query sort is reversed and the
	// rows are restored to forward display order by
	// SeekPageQuery.Query/QueryContext.
	SeekBefore
)

// SeekKey declares one structured sort dimension of a SeekWindow.
//
// Expr must be a plain column name or a pure SQL expression that is valid
// wherever the key is used (both ORDER BY and the seek predicate), for
// example "created_at" or "COALESCE(priority, 0)". Expr is treated as
// trusted SQL and is never parsed: direction and NULL positioning are
// declared explicitly here instead of being inferred from raw ORDER BY
// strings.
//
// Exactly one key — the final one — must be marked Unique. It acts as the
// total-order tie-breaker (a PRIMARY KEY or equivalent): its values must be
// neither duplicated nor NULL. Without such a key, rows sharing equal
// leading keys cannot be paged without skipping or repeating them.
type SeekKey struct {
	Expr   string
	Order  SeekOrder
	Nulls  SeekNulls
	Unique bool
}

// SeekWindow binds an immutable SelectBuilder to an explicit total-order
// keyset (seek) specification and produces offset-free page queries.
//
// Consistency contract: seek pagination assumes the underlying data does
// not change between page requests, or that every request runs against the
// same transaction snapshot. Concurrent inserts, deletes or updates can
// still cause rows to be repeated or skipped — no offset mechanism is
// involved or claimed to paper over that.
type SeekWindow struct {
	builder  SelectBuilder
	keys     []SeekKey
	pageSize uint64
}

// Builder returns the SelectBuilder the window was declared on. It is never
// modified by SeekWindow or Page.
func (w SeekWindow) Builder() SelectBuilder {
	return w.builder
}

// Keys returns a defensive copy of the declared sort keys.
func (w SeekWindow) Keys() []SeekKey {
	keys := make([]SeekKey, len(w.keys))
	copy(keys, w.keys)
	return keys
}

// PageSize returns the maximum number of rows fetched per page.
func (w SeekWindow) PageSize() uint64 {
	return w.pageSize
}

// SeekWindow declares keyset (seek) pagination on b using explicit,
// structured sort keys. It returns a new SeekWindow without modifying b;
// the original SelectBuilder stays fully usable and immutable.
//
// Construction rejects queries whose row-level ordering is not well defined
// for seek predicates (DISTINCT, GROUP BY, HAVING), queries that already
// carry an ORDER BY (ordering must be declared through SeekKey rather than
// guessed from raw strings), and any pre-existing OFFSET. Such queries must
// be adapted explicitly, typically by wrapping them in a derived table and
// seeking over the wrapper.
func (b SelectBuilder) SeekWindow(keys []SeekKey, pageSize uint64) (SeekWindow, error) {
	if pageSize == 0 {
		return SeekWindow{}, fmt.Errorf("seek: page size must be greater than zero")
	}
	if pageSize > MaxSeekPageSize {
		return SeekWindow{}, fmt.Errorf("seek: page size %d exceeds MaxSeekPageSize (%d)", pageSize, MaxSeekPageSize)
	}
	if len(keys) == 0 {
		return SeekWindow{}, fmt.Errorf("seek: at least one SeekKey is required")
	}

	seen := make(map[string]struct{}, len(keys))
	for i, key := range keys {
		expr := strings.TrimSpace(key.Expr)
		if expr == "" {
			return SeekWindow{}, fmt.Errorf("seek: key at index %d has an empty Expr", i)
		}
		if strings.ContainsAny(expr, ";") {
			return SeekWindow{}, fmt.Errorf("seek: key Expr %q must be a column or pure expression, not a statement", expr)
		}
		if key.Order != SeekAsc && key.Order != SeekDesc {
			return SeekWindow{}, fmt.Errorf("seek: key %q has invalid Order %d", expr, key.Order)
		}
		if key.Nulls != SeekNullsFirst && key.Nulls != SeekNullsLast {
			return SeekWindow{}, fmt.Errorf("seek: key %q has invalid Nulls %d", expr, key.Nulls)
		}
		if _, dup := seen[expr]; dup {
			return SeekWindow{}, fmt.Errorf("seek: duplicate key Expr %q", expr)
		}
		seen[expr] = struct{}{}
		if key.Unique && i != len(keys)-1 {
			return SeekWindow{}, fmt.Errorf("seek: the unique tie-breaker must be the last key; %q is at index %d", expr, i)
		}
	}
	if !keys[len(keys)-1].Unique {
		return SeekWindow{}, fmt.Errorf(
			"seek: the last key %q must be marked Unique to act as the non-NULL total-order tie-breaker",
			keys[len(keys)-1].Expr)
	}

	data := builder.GetStruct(b).(selectData)
	if len(data.Options) > 0 {
		return SeekWindow{}, fmt.Errorf("seek: DISTINCT/OPTIONS queries are not supported; wrap the query in a derived table first")
	}
	if len(data.GroupBys) > 0 {
		return SeekWindow{}, fmt.Errorf("seek: GROUP BY queries are not supported; wrap the aggregation in a derived table first")
	}
	if len(data.HavingParts) > 0 {
		return SeekWindow{}, fmt.Errorf("seek: HAVING queries are not supported; wrap the aggregation in a derived table first")
	}
	if len(data.OrderByParts) > 0 {
		return SeekWindow{}, fmt.Errorf("seek: declare ordering through SeekKey; existing ORDER BY clauses are never interpreted")
	}
	if len(data.Offset) > 0 {
		return SeekWindow{}, fmt.Errorf("seek: keyset pagination never uses OFFSET; call RemoveOffset first")
	}

	copied := make([]SeekKey, len(keys))
	copy(copied, keys)
	return SeekWindow{builder: b, keys: copied, pageSize: pageSize}, nil
}

// Page builds one page query relative to cursor.
//
// A nil cursor requests a boundary page: the first page for SeekAfter or the
// last page for SeekBefore. A non-nil cursor must hold exactly one typed
// value per SeekKey, in declaration order; nil elements encode SQL NULL and
// the unique tie-breaker value must never be nil, because a NULL tie-breaker
// cannot define a total order.
//
// SeekBefore flips every key (ASC/DESC and NULLS FIRST/LAST) and generates
// the mirror predicate; the returned SQL therefore sorts in reverse. Row
// order is only restored to forward display order when the page is executed
// through SeekPageQuery.Query/QueryContext — ToSql alone renders the
// reversed statement and cannot reorder rows.
//
// The SeekWindow and its original SelectBuilder are never modified.
func (w SeekWindow) Page(cursor []interface{}, direction SeekDirection) (SeekPageQuery, error) {
	if direction != SeekAfter && direction != SeekBefore {
		return SeekPageQuery{}, fmt.Errorf("seek: invalid direction %d", direction)
	}
	if cursor != nil {
		if len(cursor) != len(w.keys) {
			return SeekPageQuery{}, fmt.Errorf("seek: cursor has %d values but %d keys were declared", len(cursor), len(w.keys))
		}
		if cursor[len(cursor)-1] == nil {
			return SeekPageQuery{}, fmt.Errorf(
				"seek: unique tie-breaker %q cannot be NULL; a NULL tie-breaker cannot define a total order",
				w.keys[len(w.keys)-1].Expr)
		}
	}

	effective := w.keys
	if direction == SeekBefore {
		effective = make([]SeekKey, len(w.keys))
		for i, key := range w.keys {
			key.Order = flipSeekOrder(key.Order)
			key.Nulls = flipSeekNulls(key.Nulls)
			effective[i] = key
		}
	}

	// A seek query owns its LIMIT; any limit on the original builder is
	// replaced rather than silently combined. OFFSET was rejected up front.
	pb := w.builder.RemoveLimit()
	for _, key := range effective {
		pb = pb.OrderByClause(renderSeekOrder(key))
	}
	if cursor != nil {
		predicate := seekPredicate(effective, cursor)
		pb = pb.Where(predicate)
	}
	pb = pb.Limit(w.pageSize)

	return SeekPageQuery{SelectBuilder: pb, direction: direction}, nil
}

// SeekPageQuery is a rendered single-page query. It embeds the underlying
// SelectBuilder, so ToSql, PlaceholderFormat, RunWith and the other
// SelectBuilder entry points keep working as usual.
//
// A SeekBefore query is emitted with reversed ORDER BY and seek direction.
// Reversing the SQL alone does not reverse the returned rows: use
// Query/QueryContext, which buffer the rows and restore forward display
// order in memory.
type SeekPageQuery struct {
	SelectBuilder
	direction SeekDirection
}

// Direction reports whether the page was built SeekAfter or SeekBefore.
func (q SeekPageQuery) Direction() SeekDirection {
	return q.direction
}

// Query executes the page query against the runner set by RunWith and
// buffers every row. SeekBefore pages are reversed back to forward display
// order before returning. Column values use the default database/sql scan
// types (int64, float64, bool, []byte, string or nil).
func (q SeekPageQuery) Query() (*SeekPage, error) {
	rows, err := q.SelectBuilder.Query()
	if err != nil {
		return nil, err
	}
	return drainSeekRows(rows, q.direction)
}

// QueryContext is the context-aware variant of Query.
func (q SeekPageQuery) QueryContext(ctx context.Context) (*SeekPage, error) {
	rows, err := q.SelectBuilder.QueryContext(ctx)
	if err != nil {
		return nil, err
	}
	return drainSeekRows(rows, q.direction)
}

// SeekPage holds the rows of one executed seek query in forward display
// order, regardless of the SeekDirection used to fetch them.
type SeekPage struct {
	// Columns are the result column names in SELECT order.
	Columns []string
	// Rows holds one slice of column values per row, in SELECT column order.
	Rows [][]interface{}
}

// Len returns the number of buffered rows.
func (p *SeekPage) Len() int {
	return len(p.Rows)
}

func drainSeekRows(rows *sql.Rows, direction SeekDirection) (*SeekPage, error) {
	if rows == nil {
		return nil, fmt.Errorf("seek: runner returned nil rows without an error")
	}
	defer rows.Close()

	columns, err := rows.Columns()
	if err != nil {
		return nil, err
	}
	page := &SeekPage{Columns: columns}

	for rows.Next() {
		values := make([]interface{}, len(columns))
		pointers := make([]interface{}, len(columns))
		for i := range values {
			pointers[i] = &values[i]
		}
		if err := rows.Scan(pointers...); err != nil {
			return nil, err
		}
		page.Rows = append(page.Rows, values)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if direction == SeekBefore {
		for i, j := 0, len(page.Rows)-1; i < j; i, j = i+1, j-1 {
			page.Rows[i], page.Rows[j] = page.Rows[j], page.Rows[i]
		}
	}
	return page, nil
}

// seekPredicate builds the lexicographic "strictly after the cursor"
// predicate for the effective keys. Each dimension contributes one OR
// branch: every earlier dimension must be equal, and this dimension must
// sort strictly after the cursor. NULL handling is explicit per key because
// SQL NULL comparisons evaluate to UNKNOWN and a bare tuple comparison
// could express neither mixed directions nor NULLS FIRST/LAST semantics.
func seekPredicate(keys []SeekKey, cursor []interface{}) Sqlizer {
	branches := make([]Sqlizer, 0, len(keys))
	for i := range keys {
		terms := make([]Sqlizer, 0, i+1)
		for j := 0; j < i; j++ {
			terms = append(terms, seekEquality(keys[j], cursor[j]))
		}
		terms = append(terms, seekStrictlyAfter(keys[i], cursor[i]))
		branches = append(branches, And(terms))
	}
	return Or(branches)
}

// seekEquality matches rows equal to the cursor value on one key, using
// IS NULL for a NULL cursor since "col = NULL" is never true.
func seekEquality(key SeekKey, value interface{}) Sqlizer {
	if value == nil {
		return Expr(fmt.Sprintf("(%s IS NULL)", key.Expr))
	}
	return Expr(fmt.Sprintf("(%s = ?)", key.Expr), value)
}

// seekStrictlyAfter matches rows that sort strictly after the cursor value
// on a single key, according to the key's effective direction and NULL
// positioning.
func seekStrictlyAfter(key SeekKey, value interface{}) Sqlizer {
	if value == nil {
		if key.Nulls == SeekNullsFirst {
			// NULL sorts first, so every non-NULL value comes after it.
			return Expr(fmt.Sprintf("(%s IS NOT NULL)", key.Expr))
		}
		// NULL sorts last, so nothing sorts strictly after a NULL value.
		return Expr(sqlFalse)
	}

	operator := ">"
	if key.Order == SeekDesc {
		operator = "<"
	}
	if key.Nulls == SeekNullsLast {
		// NULLs are positioned after every non-NULL value, so they come
		// after the cursor on this dimension too.
		return Expr(fmt.Sprintf("((%s %s ?) OR (%s IS NULL))", key.Expr, operator, key.Expr), value)
	}
	return Expr(fmt.Sprintf("(%s %s ?)", key.Expr, operator), value)
}

func renderSeekOrder(key SeekKey) string {
	order := "ASC"
	if key.Order == SeekDesc {
		order = "DESC"
	}
	nulls := "NULLS FIRST"
	if key.Nulls == SeekNullsLast {
		nulls = "NULLS LAST"
	}
	return fmt.Sprintf("%s %s %s", key.Expr, order, nulls)
}

func flipSeekOrder(order SeekOrder) SeekOrder {
	if order == SeekAsc {
		return SeekDesc
	}
	return SeekAsc
}

func flipSeekNulls(nulls SeekNulls) SeekNulls {
	if nulls == SeekNullsFirst {
		return SeekNullsLast
	}
	return SeekNullsFirst
}
