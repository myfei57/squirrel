package squirrel

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Registered column / safe identifier protocol
// ---------------------------------------------------------------------------

func TestRegisterRangeColumn(t *testing.T) {
	valid := []string{"id", "_", "_x1", "t.col", "schema.table.column", "a1_b2"}
	for _, name := range valid {
		col, err := RegisterRangeColumn(name)
		require.NoError(t, err, name)
		assert.Equal(t, name, col.Name())
	}

	invalid := []string{
		"",
		"a b",
		"a;DROP TABLE x",
		"a--",
		"1a",
		".a",
		"a.",
		"a..b",
		"a-b",
		"a`b",
		`a"b`,
		"a(b)",
		"*",
	}
	for _, name := range invalid {
		_, err := RegisterRangeColumn(name)
		assert.Error(t, err, name)
	}

	long := make([]byte, maxRangeIdentifierLength+1)
	for i := range long {
		long[i] = 'a'
	}
	_, err := RegisterRangeColumn(string(long))
	assert.Error(t, err)

	assert.Panics(t, func() { MustRegisterRangeColumn("bad identifier") })
	assert.Equal(t, "ok", MustRegisterRangeColumn("ok").Name())

	// The zero value handle is rejected rather than rendered.
	_, err = NewInt64RangeSet(RangeColumn{}, ExcludeNull, Int64Point(1))
	assert.Error(t, err)
}

// ---------------------------------------------------------------------------
// Endpoint and range construction validation
// ---------------------------------------------------------------------------

func TestRangeBetweenValidation(t *testing.T) {
	// Mixed kinds are rejected.
	_, err := RangeBetween(Int64Endpoint(1, true), TimeEndpoint(time.Now(), true))
	assert.Error(t, err)
	_, err = RangeBetween(TimeEndpoint(time.Now(), false), Int64Endpoint(2, false))
	assert.Error(t, err)

	// Invalid infinity directions.
	_, err = RangeBetween(PosInfinityEndpoint(), Int64Endpoint(2, true))
	assert.Error(t, err)
	_, err = RangeBetween(Int64Endpoint(1, true), NegInfinityEndpoint())
	assert.Error(t, err)

	// Invalid finite direction.
	_, err = RangeBetween(Int64Endpoint(3, true), Int64Endpoint(2, true))
	assert.Error(t, err)
	_, err = RangeBetween(
		TimeEndpoint(time.Unix(10, 0), true),
		TimeEndpoint(time.Unix(9, 0), true))
	assert.Error(t, err)

	// Equal endpoints with any open side are recognized as empty, not errors.
	emptyVariants := []Range{
		mustRange(t, Int64Endpoint(1, false), Int64Endpoint(1, true)),  // (1, 1]
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(1, false)),  // [1, 1)
		mustRange(t, Int64Endpoint(1, false), Int64Endpoint(1, false)), // (1, 1)
	}
	for _, r := range emptyVariants {
		assert.True(t, r.empty)
	}

	// Closed equal endpoints form a point.
	p := mustRange(t, Int64Endpoint(1, true), Int64Endpoint(1, true))
	assert.False(t, p.empty)
	assert.True(t, isPointRange(p))

	// Fully unbounded range is untyped.
	all := AllValuesRange()
	assert.Equal(t, kindAny, all.kind)
	assert.True(t, all.low.infinite && !all.low.positive)
	assert.True(t, all.high.infinite && all.high.positive)
}

func mustRange(t *testing.T, low, high RangeEndpoint) Range {
	r, err := RangeBetween(low, high)
	require.NoError(t, err)
	return r
}

func TestNewRangeSetKindRejection(t *testing.T) {
	col := MustRegisterRangeColumn("id")

	// int64 set refuses time ranges and vice versa.
	_, err := NewInt64RangeSet(col, ExcludeNull, TimePoint(time.Unix(1, 0)))
	assert.Error(t, err)

	_, err = NewTimeRangeSet(col, time.Second, ExcludeNull, Int64Point(1))
	assert.Error(t, err)

	// Time sets require a positive explicit precision.
	_, err = NewTimeRangeSet(col, 0, ExcludeNull, TimePoint(time.Unix(1, 0)))
	assert.Error(t, err)
	_, err = NewTimeRangeSet(col, -time.Second, ExcludeNull, TimePoint(time.Unix(1, 0)))
	assert.Error(t, err)

	// Invalid NULL policy.
	_, err = NewInt64RangeSet(col, NullPolicy(99), Int64Point(1))
	assert.Error(t, err)
}

// ---------------------------------------------------------------------------
// Normalization: hand-pinned boundary semantics
// ---------------------------------------------------------------------------

func int64Set(t *testing.T, policy NullPolicy, ranges ...Range) *RangeSet {
	s, err := NewInt64RangeSet(MustRegisterRangeColumn("id"), policy, ranges...)
	require.NoError(t, err)
	return s
}

func TestNormalizationOverlapNestedHoles(t *testing.T) {
	// Overlapping and nested ranges collapse to one.
	s := int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(10, true)),
		mustRange(t, Int64Endpoint(5, true), Int64Endpoint(15, false)),
		mustRange(t, Int64Endpoint(7, false), Int64Endpoint(8, true)),
	)
	require.Equal(t, 1, s.Len())
	r := s.Ranges()[0]
	assert.Equal(t, int64(1), r.low.intVal)
	assert.True(t, r.low.closed)
	assert.Equal(t, int64(15), r.high.intVal)
	assert.False(t, r.high.closed) // open high of the second range is preserved

	// Adjacent ranges that share the boundary point merge.
	s = int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(2, false)),
		mustRange(t, Int64Endpoint(2, true), Int64Endpoint(3, true)),
	)
	require.Equal(t, 1, s.Len())
	r = s.Ranges()[0]
	assert.True(t, r.low.closed && r.high.closed)
	assert.Equal(t, int64(1), r.low.intVal)
	assert.Equal(t, int64(3), r.high.intVal)

	// Adjacent ranges whose shared point is excluded by both stay split.
	s = int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(1, false), Int64Endpoint(2, false)),
		mustRange(t, Int64Endpoint(2, false), Int64Endpoint(3, true)),
	)
	require.Equal(t, 2, s.Len())
	assert.Equal(t, int64(2), s.Ranges()[0].high.intVal)
	assert.False(t, s.Ranges()[0].high.closed)
	assert.Equal(t, int64(2), s.Ranges()[1].low.intVal)
	assert.False(t, s.Ranges()[1].low.closed)

	// Open lower bound merged into a covering closed range keeps openness.
	s = int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(1, false), Int64Endpoint(2, true)),
		mustRange(t, Int64Endpoint(2, true), Int64Endpoint(3, true)),
	)
	require.Equal(t, 1, s.Len())
	r = s.Ranges()[0]
	assert.False(t, r.low.closed)
	assert.Equal(t, int64(1), r.low.intVal)
	assert.True(t, r.high.closed)
	assert.Equal(t, int64(3), r.high.intVal)

	// Empty ranges are dropped; an all-empty set normalizes to zero ranges.
	s = int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(1, false), Int64Endpoint(1, false)),
		mustRange(t, Int64Endpoint(2, true), Int64Endpoint(2, false)),
	)
	assert.Equal(t, 0, s.Len())
}

func TestNormalizationPointAndUnbounded(t *testing.T) {
	// Point absorbed by covering range.
	s := int64Set(t, ExcludeNull,
		Int64Point(3),
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(5, true)),
	)
	require.Equal(t, 1, s.Len())
	assert.False(t, isPointRange(s.Ranges()[0]))

	// Isolated point stays a point.
	s = int64Set(t, ExcludeNull, Int64Point(42))
	require.Equal(t, 1, s.Len())
	assert.True(t, isPointRange(s.Ranges()[0]))
	assert.Equal(t, int64(42), s.Ranges()[0].low.intVal)

	// Fully unbounded swallows everything.
	s = int64Set(t, ExcludeNull,
		AllValuesRange(),
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(2, false)),
		Int64Point(42),
	)
	require.Equal(t, 1, s.Len())
	assert.True(t, s.Ranges()[0].low.infinite)
	assert.True(t, s.Ranges()[0].high.infinite)

	// Half-unbounded ranges merge into one unbounded tail.
	s = int64Set(t, ExcludeNull,
		mustRange(t, NegInfinityEndpoint(), Int64Endpoint(0, false)),
		mustRange(t, Int64Endpoint(0, true), PosInfinityEndpoint()),
	)
	require.Equal(t, 1, s.Len())
	assert.True(t, s.Ranges()[0].low.infinite)
	assert.True(t, s.Ranges()[0].high.infinite)
}

func TestDifferenceSplitsAndKeepsEndpoints(t *testing.T) {
	a := int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(10, true)))

	// [1,10] - (3,5) = [1,3] U [5,10]
	b := int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(3, false), Int64Endpoint(5, false)))
	d, err := a.Difference(b)
	require.NoError(t, err)
	require.Equal(t, 2, d.Len())
	assert.Equal(t, int64(1), d.Ranges()[0].low.intVal)
	assert.True(t, d.Ranges()[0].low.closed)
	assert.Equal(t, int64(3), d.Ranges()[0].high.intVal)
	assert.True(t, d.Ranges()[0].high.closed)
	assert.Equal(t, int64(5), d.Ranges()[1].low.intVal)
	assert.True(t, d.Ranges()[1].low.closed)
	assert.Equal(t, int64(10), d.Ranges()[1].high.intVal)
	assert.True(t, d.Ranges()[1].high.closed)

	// [1,10] - [3,5] = [1,3) U (5,10]: original open exclusions preserved.
	b = int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(3, true), Int64Endpoint(5, true)))
	d, err = a.Difference(b)
	require.NoError(t, err)
	require.Equal(t, 2, d.Len())
	assert.Equal(t, int64(3), d.Ranges()[0].high.intVal)
	assert.False(t, d.Ranges()[0].high.closed)
	assert.Equal(t, int64(5), d.Ranges()[1].low.intVal)
	assert.False(t, d.Ranges()[1].low.closed)

	// Removing a point splits into two ranges, both excluding that point.
	b = int64Set(t, ExcludeNull, Int64Point(5))
	d, err = a.Difference(b)
	require.NoError(t, err)
	require.Equal(t, 2, d.Len())
	assert.False(t, d.Ranges()[0].high.closed)
	assert.False(t, d.Ranges()[1].low.closed)

	// Intersection with empty set is empty; union keeps A.
	empty := int64Set(t, ExcludeNull)
	i, err := a.Intersect(empty)
	require.NoError(t, err)
	assert.Equal(t, 0, i.Len())
	u, err := a.Union(empty)
	require.NoError(t, err)
	require.Equal(t, 1, u.Len())
	assert.Equal(t, a.Ranges(), u.Ranges())

	// Set identity checks.
	otherCol := MustRegisterRangeColumn("other")
	other, err := NewInt64RangeSet(otherCol, ExcludeNull, Int64Point(1))
	require.NoError(t, err)
	_, err = a.Union(other)
	assert.Error(t, err)
	_, err = a.Union(nil)
	assert.Error(t, err)

	timeSet, err := NewTimeRangeSet(MustRegisterRangeColumn("id"), time.Second, ExcludeNull,
		TimePoint(time.Unix(1, 0)))
	require.NoError(t, err)
	_, err = a.Union(timeSet)
	assert.Error(t, err)
}

func TestInt64ExtremesKeepOpenEndpoints(t *testing.T) {
	// Open interval over the whole int64 domain must keep the exact extreme
	// values as bound args and use strict comparisons - no +/-1 arithmetic.
	r := mustRange(t,
		Int64Endpoint(math.MinInt64, false),
		Int64Endpoint(math.MaxInt64, false))
	s := int64Set(t, ExcludeNull, r)
	sql, args, err := s.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "(id > ? AND id < ?)", sql)
	require.Equal(t, 2, len(args))
	assert.Equal(t, int64(math.MinInt64), args[0])
	assert.Equal(t, int64(math.MaxInt64), args[1])

	// Closed extreme endpoints use inclusive comparisons.
	r = mustRange(t,
		Int64Endpoint(math.MinInt64, true),
		Int64Endpoint(math.MaxInt64, true))
	s = int64Set(t, ExcludeNull, r)
	sql, args, err = s.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "(id >= ? AND id <= ?)", sql)
	assert.Equal(t, []interface{}{int64(math.MinInt64), int64(math.MaxInt64)}, args)

	// Half-unbounded at each extreme.
	s = int64Set(t, ExcludeNull,
		mustRange(t, NegInfinityEndpoint(), Int64Endpoint(math.MinInt64, true)))
	sql, args, err = s.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "id <= ?", sql)
	assert.Equal(t, []interface{}{int64(math.MinInt64)}, args)

	s = int64Set(t, ExcludeNull,
		mustRange(t, Int64Endpoint(math.MaxInt64, false), PosInfinityEndpoint()))
	sql, args, err = s.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "id > ?", sql)
	assert.Equal(t, []interface{}{int64(math.MaxInt64)}, args)
}

// ---------------------------------------------------------------------------
// Exhaustive small-domain reference comparison
// ---------------------------------------------------------------------------

func refRangeContains(r Range, v int64) bool {
	if r.empty {
		return false
	}
	if !r.low.infinite {
		if r.low.closed && v < r.low.intVal {
			return false
		}
		if !r.low.closed && v <= r.low.intVal {
			return false
		}
	}
	if !r.high.infinite {
		if r.high.closed && v > r.high.intVal {
			return false
		}
		if !r.high.closed && v >= r.high.intVal {
			return false
		}
	}
	return true
}

func refAny(ranges []Range, v int64) bool {
	for _, r := range ranges {
		if refRangeContains(r, v) {
			return true
		}
	}
	return false
}

func setContains(s *RangeSet, v int64) bool {
	for _, r := range s.Ranges() {
		if refRangeContains(r, v) {
			return true
		}
	}
	return false
}

func assertDisjointSorted(t *testing.T, ranges []Range) {
	for i := 1; i < len(ranges); i++ {
		prev, cur := ranges[i-1], ranges[i]
		cmp := compareEndpointValue(prev.high, cur.low)
		if cmp > 0 {
			t.Fatalf("ranges not sorted at %d: %v", i, ranges)
		}
		if cmp == 0 {
			// Touching at one value is only legal when the point is excluded
			// by at least one side; otherwise normalization would have merged.
			assert.False(t, prev.high.closed && cur.low.closed,
				"ranges %d share an included point but were not merged", i)
		}
	}
}

func TestRangeSetExhaustiveReference(t *testing.T) {
	rng := rand.New(rand.NewSource(20261110))
	domain := []int64{-30, -8, -5, -3, -1, 0, 1, 2, 3, 4, 7, 8, 30}
	bounds := []int64{-8, -5, -3, -1, 0, 2, 4, 7, 8}
	col := MustRegisterRangeColumn("id")

	mkEndpoint := func(lowSide bool) RangeEndpoint {
		if rng.Intn(8) == 0 {
			if lowSide {
				return NegInfinityEndpoint()
			}
			return PosInfinityEndpoint()
		}
		v := bounds[rng.Intn(len(bounds))]
		return Int64Endpoint(v, rng.Intn(2) == 0)
	}
	mkRanges := func() []Range {
		n := rng.Intn(7)
		out := make([]Range, 0, n)
		for i := 0; i < n; i++ {
			rg, err := RangeBetween(mkEndpoint(true), mkEndpoint(false))
			if err != nil {
				continue
			}
			out = append(out, rg)
		}
		return out
	}

	for iter := 0; iter < 400; iter++ {
		rawA, rawB := mkRanges(), mkRanges()
		sa, err := NewInt64RangeSet(col, ExcludeNull, rawA...)
		require.NoError(t, err, "iter %d", iter)
		sb, err := NewInt64RangeSet(col, ExcludeNull, rawB...)
		require.NoError(t, err, "iter %d", iter)

		union, err := sa.Union(sb)
		require.NoError(t, err)
		inter, err := sa.Intersect(sb)
		require.NoError(t, err)
		diff, err := sa.Difference(sb)
		require.NoError(t, err)

		assertDisjointSorted(t, union.Ranges())
		assertDisjointSorted(t, inter.Ranges())
		assertDisjointSorted(t, diff.Ranges())
		if union.Len() > sa.Len()+sb.Len() {
			t.Fatalf("iter %d: union grew beyond input sizes", iter)
		}

		for _, v := range domain {
			wantA, wantB := refAny(rawA, v), refAny(rawB, v)
			assert.Equal(t, wantA, setContains(sa, v), "A membership at %d (iter %d)", v, iter)
			assert.Equal(t, wantB, setContains(sb, v), "B membership at %d (iter %d)", v, iter)
			assert.Equal(t, wantA || wantB, setContains(union, v),
				"union at %d (iter %d)", v, iter)
			assert.Equal(t, wantA && wantB, setContains(inter, v),
				"intersect at %d (iter %d)", v, iter)
			assert.Equal(t, wantA && !wantB, setContains(diff, v),
				"difference at %d (iter %d)", v, iter)
		}
	}
}

// ---------------------------------------------------------------------------
// Time precision normalization
// ---------------------------------------------------------------------------

func TestTimeRangeSetPrecision(t *testing.T) {
	col := MustRegisterRangeColumn("ts")
	base := time.Date(2026, 11, 10, 10, 0, 0, 0, time.UTC)

	// Sub-second endpoints are truncated to the explicit second precision.
	r, err := RangeBetween(
		TimeEndpoint(base.Add(500*time.Millisecond), true),
		TimeEndpoint(base.Add(2*time.Second+900*time.Millisecond), false))
	require.NoError(t, err)

	s, err := NewTimeRangeSet(col, time.Second, ExcludeNull, r)
	require.NoError(t, err)
	require.Equal(t, 1, s.Len())
	got := s.Ranges()[0]
	assert.True(t, got.low.timeVal.Equal(base))
	assert.True(t, got.high.timeVal.Equal(base.Add(2*time.Second)))
	assert.False(t, got.high.closed)

	// After truncation the endpoints collapse: open side makes the range empty.
	r, err = RangeBetween(
		TimeEndpoint(base.Add(100*time.Millisecond), true),
		TimeEndpoint(base.Add(200*time.Millisecond), false))
	require.NoError(t, err)
	s, err = NewTimeRangeSet(col, time.Second, ExcludeNull, r)
	require.NoError(t, err)
	assert.Equal(t, 0, s.Len())

	// Closed collapse becomes a point and binds a single time argument.
	r, err = RangeBetween(
		TimeEndpoint(base.Add(100*time.Millisecond), true),
		TimeEndpoint(base.Add(200*time.Millisecond), true))
	require.NoError(t, err)
	s, err = NewTimeRangeSet(col, time.Second, ExcludeNull, r)
	require.NoError(t, err)
	require.Equal(t, 1, s.Len())
	sql, args, err := s.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "ts = ?", sql)
	require.Equal(t, 1, len(args))
	assert.IsType(t, time.Time{}, args[0])
	assert.True(t, args[0].(time.Time).Equal(base))

	// Monotonic clock readings do not affect comparison after truncation.
	wall := base
	withMono := base.Add(0) // Add returns a time carrying a monotonic reading
	r1, err := RangeBetween(TimeEndpoint(wall, true), TimeEndpoint(wall.Add(time.Second), true))
	require.NoError(t, err)
	r2, err := RangeBetween(TimeEndpoint(withMono, true),
		TimeEndpoint(withMono.Add(time.Second), true))
	require.NoError(t, err)
	s1, err := NewTimeRangeSet(col, time.Second, ExcludeNull, r1)
	require.NoError(t, err)
	s2, err := NewTimeRangeSet(col, time.Second, ExcludeNull, r2)
	require.NoError(t, err)
	assert.Equal(t, s1.Ranges(), s2.Ranges())

	// Precision mismatch between operands is rejected.
	minuteSet, err := NewTimeRangeSet(col, time.Minute, ExcludeNull,
		TimePoint(base))
	require.NoError(t, err)
	_, err = s1.Union(minuteSet)
	assert.Error(t, err)
}

// ---------------------------------------------------------------------------
// NULL policy SQL generation
// ---------------------------------------------------------------------------

func TestContainsNullPolicyAndShape(t *testing.T) {
	col := MustRegisterRangeColumn("id")
	r1 := mustRange(t, Int64Endpoint(1, true), Int64Endpoint(5, false))
	r2 := Int64Point(9)

	excl, err := NewInt64RangeSet(col, ExcludeNull, r1, r2)
	require.NoError(t, err)
	sql, args, err := excl.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "((id >= ? AND id < ?) OR id = ?)", sql)
	assert.Equal(t, []interface{}{int64(1), int64(5), int64(9)}, args)

	incl, err := NewInt64RangeSet(col, IncludeNull, r1, r2)
	require.NoError(t, err)
	sql, args, err = incl.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "(((id >= ? AND id < ?) OR id = ?) OR id IS NULL)", sql)
	assert.Equal(t, []interface{}{int64(1), int64(5), int64(9)}, args)

	// Empty set.
	emptyExcl := int64Set(t, ExcludeNull)
	sql, args, err = emptyExcl.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "(1=0)", sql)
	assert.Equal(t, []interface{}{}, args)

	emptyIncl, err := NewInt64RangeSet(col, IncludeNull)
	require.NoError(t, err)
	sql, args, err = emptyIncl.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "(id IS NULL)", sql)
	assert.Equal(t, []interface{}{}, args)

	// Universal set: policy drives the NULL rule explicitly.
	univExcl := int64Set(t, ExcludeNull, AllValuesRange())
	sql, args, err = univExcl.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "(id IS NOT NULL)", sql)
	assert.Equal(t, []interface{}{}, args)

	univIncl, err := NewInt64RangeSet(col, IncludeNull, AllValuesRange())
	require.NoError(t, err)
	sql, args, err = univIncl.Contains().ToSql()
	require.NoError(t, err)
	assert.Equal(t, "(1=1)", sql)
	assert.Equal(t, []interface{}{}, args)
}

func TestContainsWhereCompositionAndPlaceholders(t *testing.T) {
	col := MustRegisterRangeColumn("id")
	set, err := NewInt64RangeSet(col, IncludeNull,
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(5, true)),
		Int64Point(9))
	require.NoError(t, err)

	// Compound predicate stays grouped inside parentheses when ANDed with
	// further Where parts, and argument order follows the generated SQL.
	query, args, err := Select("id").From("t").
		Where(set.Contains()).
		Where(Eq{"tenant": "acme"}).
		ToSql()
	require.NoError(t, err)
	assert.Equal(t,
		"SELECT id FROM t WHERE (((id >= ? AND id <= ?) OR id = ?) OR id IS NULL) AND tenant = ?",
		query)
	assert.Equal(t, []interface{}{int64(1), int64(5), int64(9), "acme"}, args)

	// Works inside Or/And conjunctions too.
	query, args, err = Select("id").From("t").
		Where(Or{set.Contains(), Eq{"other": 7}}).
		ToSql()
	require.NoError(t, err)
	assert.Equal(t,
		"SELECT id FROM t WHERE ((((id >= ? AND id <= ?) OR id = ?) OR id IS NULL) OR other = ?)",
		query)
	assert.Equal(t, []interface{}{int64(1), int64(5), int64(9), 7}, args)

	// Positional placeholder formats renumber the same bound args.
	dollar, args, err := Select("id").From("t").
		PlaceholderFormat(Dollar).
		Where(set.Contains()).
		ToSql()
	require.NoError(t, err)
	assert.Equal(t,
		"SELECT id FROM t WHERE (((id >= $1 AND id <= $2) OR id = $3) OR id IS NULL)",
		dollar)
	assert.Equal(t, []interface{}{int64(1), int64(5), int64(9)}, args)
}

// ---------------------------------------------------------------------------
// Budgets and failure atomicity
// ---------------------------------------------------------------------------

func TestRangeSetBudgets(t *testing.T) {
	col := MustRegisterRangeColumn("id")

	tooMany := make([]Range, MaxRangeSetInputRanges+1)
	for i := range tooMany {
		tooMany[i] = Int64Point(int64(i))
	}
	_, err := NewInt64RangeSet(col, ExcludeNull, tooMany...)
	assert.Error(t, err)

	// Difference that would produce more output ranges than allowed.
	big := mustRange(t, Int64Endpoint(0, true),
		Int64Endpoint(int64(MaxRangeSetOutputRanges+10), true))
	holes := make([]Range, MaxRangeSetInputRanges)
	for i := range holes {
		holes[i] = Int64Point(int64(i + 1))
	}
	holeSet, err := NewInt64RangeSet(col, ExcludeNull, holes...)
	require.NoError(t, err)
	bigSet, err := NewInt64RangeSet(col, ExcludeNull, big)
	require.NoError(t, err)
	_, err = bigSet.Difference(holeSet)
	assert.Error(t, err)

	// ToSql arg budget: build a predicate directly beyond the bound and
	// assert no partial predicate leaks out.
	oversized := make([]Range, MaxRangeSetOutputRanges+1)
	for i := range oversized {
		v := int64(i * 2)
		oversized[i] = mustRange(t, Int64Endpoint(v, true), Int64Endpoint(v, false))
	}
	// Normalize path is blocked by the output budget, so exercise the SQL
	// budget check on a hand-assembled predicate of the same shape.
	pred := rangeSetPredicate{
		column: "id",
		kind:   kindInt64,
		policy: ExcludeNull,
		ranges: oversized,
	}
	sql, args, err := pred.ToSql()
	assert.Error(t, err)
	assert.Equal(t, "", sql)
	assert.Nil(t, args)
}

// ---------------------------------------------------------------------------
// Immutability
// ---------------------------------------------------------------------------

func TestRangeSetImmutability(t *testing.T) {
	col := MustRegisterRangeColumn("id")
	input := []Range{
		mustRange(t, Int64Endpoint(1, true), Int64Endpoint(3, true)),
		mustRange(t, Int64Endpoint(2, true), Int64Endpoint(5, true)),
	}
	snapshot := append([]Range(nil), input...)

	s, err := NewInt64RangeSet(col, ExcludeNull, input...)
	require.NoError(t, err)
	require.Equal(t, 1, s.Len())

	// Mutating the caller's slice after construction has no effect.
	input[0] = Int64Point(999)
	s2, err := NewInt64RangeSet(col, ExcludeNull, snapshot...)
	require.NoError(t, err)
	assert.Equal(t, s2.Ranges(), s.Ranges())

	// Ranges() returns an independent copy.
	ranges := s.Ranges()
	ranges[0] = Int64Point(-123)
	assert.NotEqual(t, ranges, s.Ranges())

	// Set operations leave operands untouched and return independent slices.
	other, err := NewInt64RangeSet(col, ExcludeNull, Int64Point(4))
	require.NoError(t, err)
	before := s.Ranges()
	u, err := s.Union(other)
	require.NoError(t, err)
	_, err = s.Difference(other)
	require.NoError(t, err)
	assert.Equal(t, before, s.Ranges())
	uCopy := u.Ranges()
	uCopy[0] = Int64Point(777)
	// Mutating the returned copy must not affect the immutable set.
	assert.NotEqual(t, uCopy, u.Ranges())
	assert.Equal(t, before, s.Ranges())
}

// ---------------------------------------------------------------------------
// Legacy predicates keep their existing behavior
// ---------------------------------------------------------------------------

func TestLegacyPredicatesUnchanged(t *testing.T) {
	sql, args, err := Eq{"id": 1}.ToSql()
	require.NoError(t, err)
	assert.Equal(t, "id = ?", sql)
	assert.Equal(t, []interface{}{1}, args)

	sql, args, err = Lt{"id": 10}.ToSql()
	require.NoError(t, err)
	assert.Equal(t, "id < ?", sql)
	assert.Equal(t, []interface{}{10}, args)

	sql, args, err = GtOrEq{"id": 10}.ToSql()
	require.NoError(t, err)
	assert.Equal(t, "id >= ?", sql)
	assert.Equal(t, []interface{}{10}, args)

	sql, args, err = NotEq{"id": nil}.ToSql()
	require.NoError(t, err)
	assert.Equal(t, "id IS NOT NULL", sql)
	assert.Equal(t, []interface{}(nil), args)
}
