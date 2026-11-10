package squirrel

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
)

// Budget limits applied to RangeSet construction, set operations and SQL
// generation. They bound the amount of work and the size of a generated
// predicate so a caller cannot force unbounded SQL or argument slices.
const (
	// MaxRangeSetInputRanges bounds the number of Range values accepted by a
	// single constructor or set operation call.
	MaxRangeSetInputRanges = 256
	// MaxRangeSetOutputRanges bounds the number of normalized, disjoint ranges
	// held by a RangeSet.
	MaxRangeSetOutputRanges = 256
	// MaxRangeSetBoundArgs bounds the number of bound arguments produced by a
	// Contains predicate.
	MaxRangeSetBoundArgs = 512

	maxRangeIdentifierLength = 128
)

// rangeIdentifierPattern is the safe identifier protocol. A column name is
// only ever emitted verbatim after matching this pattern, so an external
// string can never smuggle SQL syntax into a generated predicate. Qualified
// identifiers of the form "table.column" (or deeper) are supported, with each
// dot-separated part validated independently.
var rangeIdentifierPattern = regexp.MustCompile(
	`^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$`)

// NullPolicy decides how a RangeSet predicate treats SQL NULL values.
//
// SQL comparisons evaluate to UNKNOWN (not TRUE) for NULL input, so the
// policy is applied explicitly instead of being inferred from interval
// shape: in particular, a fully unbounded interval is not silently treated
// as an IS NOT NULL rule.
type NullPolicy int

const (
	// ExcludeNull (the zero value) makes the predicate never match NULL rows.
	ExcludeNull NullPolicy = iota
	// IncludeNull makes the predicate additionally match NULL rows by emitting
	// an explicit "OR col IS NULL" clause.
	IncludeNull
)

func (p NullPolicy) valid() bool {
	return p == ExcludeNull || p == IncludeNull
}

// RangeColumn is a trusted column identifier for use with RangeSet.
//
// Instances can only be obtained through RegisterRangeColumn (or
// MustRegisterRangeColumn), which validates the identifier. The zero value is
// not usable. RangeSet renders the validated name verbatim; unvalidated
// strings are never interpolated into SQL.
type RangeColumn struct {
	name string
}

// RegisterRangeColumn validates name against the safe identifier protocol and
// returns a trusted RangeColumn handle.
func RegisterRangeColumn(name string) (RangeColumn, error) {
	if name == "" {
		return RangeColumn{}, fmt.Errorf("range column identifier must not be empty")
	}
	if len(name) > maxRangeIdentifierLength {
		return RangeColumn{}, fmt.Errorf(
			"range column identifier %q exceeds maximum length of %d characters",
			name, maxRangeIdentifierLength)
	}
	for _, part := range strings.Split(name, ".") {
		if len(part) == 0 || len(part) > maxRangeIdentifierLength {
			return RangeColumn{}, fmt.Errorf(
				"range column identifier %q contains an empty identifier part", name)
		}
	}
	if !rangeIdentifierPattern.MatchString(name) {
		return RangeColumn{}, fmt.Errorf(
			"range column identifier %q is not a safe SQL identifier: "+
				"expected letter or underscore followed by letters, digits or underscores",
			name)
	}
	return RangeColumn{name: name}, nil
}

// MustRegisterRangeColumn is like RegisterRangeColumn but panics on error.
func MustRegisterRangeColumn(name string) RangeColumn {
	col, err := RegisterRangeColumn(name)
	if err != nil {
		panic(err)
	}
	return col
}

// Name returns the validated identifier carried by the column handle.
func (c RangeColumn) Name() string {
	return c.name
}

// rangeKind identifies the comparable value type of a range set. All ranges
// and endpoints inside one set share a single kind.
type rangeKind uint8

const (
	kindAny rangeKind = iota
	kindInt64
	kindTime
)

func (k rangeKind) String() string {
	switch k {
	case kindInt64:
		return "int64"
	case kindTime:
		return "time.Time"
	default:
		return "untyped"
	}
}

// RangeEndpoint is one end of a Range. A finite endpoint carries either an
// int64 or a time.Time value and an open/closed flag. Infinite endpoints are
// created with NegInfinityEndpoint and PosInfinityEndpoint and carry no value.
type RangeEndpoint struct {
	kind     rangeKind
	infinite bool
	positive bool // direction of an infinite endpoint
	closed   bool // meaningful only for finite endpoints
	intVal   int64
	timeVal  time.Time
}

// Int64Endpoint builds a finite int64 endpoint. closed=true means the endpoint
// value belongs to the range ("[" / "]"), closed=false means it is excluded
// ("(" / ")").
func Int64Endpoint(v int64, closed bool) RangeEndpoint {
	return RangeEndpoint{kind: kindInt64, closed: closed, intVal: v}
}

// TimeEndpoint builds a finite time.Time endpoint. The supplied time is
// normalized to the owning RangeSet's explicit precision when the set is
// constructed.
func TimeEndpoint(t time.Time, closed bool) RangeEndpoint {
	return RangeEndpoint{kind: kindTime, closed: closed, timeVal: t}
}

// NegInfinityEndpoint builds a negative-infinity lower endpoint.
func NegInfinityEndpoint() RangeEndpoint {
	return RangeEndpoint{kind: kindAny, infinite: true, positive: false}
}

// PosInfinityEndpoint builds a positive-infinity upper endpoint.
func PosInfinityEndpoint() RangeEndpoint {
	return RangeEndpoint{kind: kindAny, infinite: true, positive: true}
}

func (e RangeEndpoint) finiteWith(closed bool) RangeEndpoint {
	e.infinite = false
	e.positive = false
	e.closed = closed
	return e
}

// Range is a single interval. Use RangeBetween and the point constructors
// rather than building values directly.
type Range struct {
	kind  rangeKind
	empty bool
	low   RangeEndpoint
	high  RangeEndpoint
}

// RangeBetween builds a Range from two endpoints. It rejects mixed int64 and
// time.Time endpoints, invalid infinity directions and ranges whose lower
// endpoint is greater than the upper endpoint. An interval whose endpoints
// are equal but which excludes that point on either side is recognized as an
// empty range and is dropped during normalization.
//
// Open endpoints are kept as open comparisons: endpoints are never adjusted by
// adding or subtracting 1, which would overflow int64 at its extremes or
// change time precision.
func RangeBetween(low, high RangeEndpoint) (Range, error) {
	if low.infinite && low.positive {
		return Range{}, fmt.Errorf("lower endpoint cannot be positive infinity")
	}
	if high.infinite && !high.positive {
		return Range{}, fmt.Errorf("upper endpoint cannot be negative infinity")
	}

	kind := kindAny
	for _, e := range []RangeEndpoint{low, high} {
		if e.kind == kindAny {
			continue
		}
		if kind == kindAny {
			kind = e.kind
		} else if kind != e.kind {
			return Range{}, fmt.Errorf(
				"cannot mix int64 and time.Time endpoints in the same range")
		}
	}

	if !low.infinite && !high.infinite {
		switch compareEndpointValue(low, high) {
		case 1:
			return Range{}, fmt.Errorf(
				"invalid range direction: lower endpoint is greater than upper endpoint")
		case 0:
			if !low.closed || !high.closed {
				return Range{kind: kind, empty: true, low: low, high: high}, nil
			}
		}
	}

	return Range{kind: kind, low: low, high: high}, nil
}

// Int64Point builds the closed single-point range [v, v].
func Int64Point(v int64) Range {
	r, err := RangeBetween(Int64Endpoint(v, true), Int64Endpoint(v, true))
	if err != nil {
		panic(err)
	}
	return r
}

// TimePoint builds the closed single-point range [t, t]. The time is
// normalized to the owning RangeSet's precision when the set is constructed.
func TimePoint(t time.Time) Range {
	r, err := RangeBetween(TimeEndpoint(t, true), TimeEndpoint(t, true))
	if err != nil {
		panic(err)
	}
	return r
}

// AllValuesRange builds the fully unbounded range (-infinity, +infinity). It
// is untyped and is accepted by both int64 and time sets.
func AllValuesRange() Range {
	return Range{kind: kindAny, low: NegInfinityEndpoint(), high: PosInfinityEndpoint()}
}

// compareEndpointValue orders two finite endpoints of the same kind.
func compareEndpointValue(a, b RangeEndpoint) int {
	switch a.kind {
	case kindInt64:
		switch {
		case a.intVal < b.intVal:
			return -1
		case a.intVal > b.intVal:
			return 1
		default:
			return 0
		}
	case kindTime:
		switch {
		case a.timeVal.Before(b.timeVal):
			return -1
		case a.timeVal.After(b.timeVal):
			return 1
		default:
			return 0
		}
	default:
		panic(fmt.Sprintf("cannot compare endpoints of kind %s", a.kind))
	}
}

// RangeSet is an immutable, normalized collection of disjoint intervals over
// one registered column. Instances are created with NewInt64RangeSet or
// NewTimeRangeSet and combined with Union, Intersect and Difference, each of
// which returns a new RangeSet without modifying its operands.
type RangeSet struct {
	column    RangeColumn
	kind      rangeKind
	precision time.Duration // explicit precision for kindTime; zero otherwise
	policy    NullPolicy
	ranges    []Range // sorted, disjoint, normalized; owned exclusively
}

// NewInt64RangeSet builds an int64 RangeSet for a registered column. Empty
// input ranges are dropped and the remaining ranges are union-normalized.
func NewInt64RangeSet(column RangeColumn, policy NullPolicy, ranges ...Range) (*RangeSet, error) {
	return newRangeSet(column, kindInt64, 0, policy, ranges)
}

// NewTimeRangeSet builds a time.Time RangeSet for a registered column. The
// positive precision determines the explicit normalization applied to every
// endpoint: each time is truncated towards the past to a multiple of
// precision measured as absolute time, which also strips monotonic clock
// readings so endpoint comparison is stable.
func NewTimeRangeSet(column RangeColumn, precision time.Duration, policy NullPolicy,
	ranges ...Range) (*RangeSet, error) {

	if precision <= 0 {
		return nil, fmt.Errorf("time range set precision must be a positive duration")
	}
	return newRangeSet(column, kindTime, precision, policy, ranges)
}

func newRangeSet(column RangeColumn, kind rangeKind, precision time.Duration,
	policy NullPolicy, input []Range) (*RangeSet, error) {

	if column.name == "" {
		return nil, fmt.Errorf("RangeSet requires a registered RangeColumn")
	}
	if !policy.valid() {
		return nil, fmt.Errorf("invalid NullPolicy %d", policy)
	}
	if len(input) > MaxRangeSetInputRanges {
		return nil, fmt.Errorf(
			"range set input has %d ranges, exceeding budget of %d",
			len(input), MaxRangeSetInputRanges)
	}

	prepared := make([]Range, 0, len(input))
	for i, r := range input {
		pr, err := prepareRange(r, kind, precision)
		if err != nil {
			return nil, fmt.Errorf("range at index %d: %v", i, err)
		}
		if pr.empty {
			continue
		}
		prepared = append(prepared, pr)
	}

	normalized := combineRanges(prepared, nil, kind, func(inA, _ bool) bool {
		return inA
	})
	if len(normalized) > MaxRangeSetOutputRanges {
		return nil, fmt.Errorf(
			"normalized range set has %d ranges, exceeding budget of %d",
			len(normalized), MaxRangeSetOutputRanges)
	}

	return &RangeSet{
		column:    column,
		kind:      kind,
		precision: precision,
		policy:    policy,
		ranges:    normalized,
	}, nil
}

// prepareRange coerces a caller-supplied range to the set's kind, applies
// time precision normalization and re-validates direction afterwards.
func prepareRange(r Range, kind rangeKind, precision time.Duration) (Range, error) {
	if r.empty {
		return Range{kind: kind, empty: true}, nil
	}
	low, err := prepareEndpoint(r.low, kind, precision)
	if err != nil {
		return Range{}, err
	}
	high, err := prepareEndpoint(r.high, kind, precision)
	if err != nil {
		return Range{}, err
	}

	if !low.infinite && !high.infinite {
		switch compareEndpointValue(low, high) {
		case 1:
			return Range{}, fmt.Errorf(
				"invalid range direction after precision normalization: lower endpoint is greater than upper endpoint")
		case 0:
			if !low.closed || !high.closed {
				return Range{kind: kind, empty: true}, nil
			}
		}
	}
	return Range{kind: kind, low: low, high: high}, nil
}

func prepareEndpoint(e RangeEndpoint, kind rangeKind, precision time.Duration) (RangeEndpoint, error) {
	if e.infinite {
		e.kind = kindAny
		return e, nil
	}
	if e.kind != kind {
		return RangeEndpoint{}, fmt.Errorf(
			"range set accepts %s endpoints only, got %s endpoint", kind, e.kind)
	}
	if kind == kindTime {
		e.timeVal = e.timeVal.Truncate(precision)
	}
	return e, nil
}

// Len returns the number of normalized ranges held by the set.
func (s *RangeSet) Len() int {
	return len(s.ranges)
}

// NullPolicy returns the NULL policy of the set.
func (s *RangeSet) NullPolicy() NullPolicy {
	return s.policy
}

// Column returns the registered column handle of the set.
func (s *RangeSet) Column() RangeColumn {
	return s.column
}

// Ranges returns a defensive copy of the normalized ranges. Callers cannot
// mutate the set through the returned slice.
func (s *RangeSet) Ranges() []Range {
	out := make([]Range, len(s.ranges))
	copy(out, s.ranges)
	return out
}

func (s *RangeSet) checkCompatible(other *RangeSet, operation string) error {
	if other == nil {
		return fmt.Errorf("%s requires a non-nil RangeSet argument", operation)
	}
	if s.column != other.column {
		return fmt.Errorf(
			"%s requires both range sets to target the same registered column (%q vs %q)",
			operation, s.column.name, other.column.name)
	}
	if s.kind != other.kind {
		return fmt.Errorf(
			"%s requires both range sets to use the same endpoint kind (%s vs %s)",
			operation, s.kind, other.kind)
	}
	if s.kind == kindTime && s.precision != other.precision {
		return fmt.Errorf(
			"%s requires both time range sets to use the same precision (%s vs %s)",
			operation, s.precision, other.precision)
	}
	return nil
}

func (s *RangeSet) combined(other *RangeSet, operation string,
	keep func(inA, inB bool) bool) (*RangeSet, error) {

	if err := s.checkCompatible(other, operation); err != nil {
		return nil, err
	}
	if len(s.ranges) > MaxRangeSetInputRanges || len(other.ranges) > MaxRangeSetInputRanges {
		return nil, fmt.Errorf("%s input exceeds range budget of %d",
			operation, MaxRangeSetInputRanges)
	}
	out := combineRanges(s.ranges, other.ranges, s.kind, keep)
	if len(out) > MaxRangeSetOutputRanges {
		return nil, fmt.Errorf(
			"%s result has %d ranges, exceeding budget of %d",
			operation, len(out), MaxRangeSetOutputRanges)
	}
	return &RangeSet{
		column:    s.column,
		kind:      s.kind,
		precision: s.precision,
		policy:    s.policy,
		ranges:    out,
	}, nil
}

// Union returns a new RangeSet containing every value contained in either set.
// The receiver and argument are not modified.
func (s *RangeSet) Union(other *RangeSet) (*RangeSet, error) {
	return s.combined(other, "Union", func(inA, inB bool) bool {
		return inA || inB
	})
}

// Intersect returns a new RangeSet containing only values contained in both
// sets. The receiver and argument are not modified.
func (s *RangeSet) Intersect(other *RangeSet) (*RangeSet, error) {
	return s.combined(other, "Intersect", func(inA, inB bool) bool {
		return inA && inB
	})
}

// Difference returns a new RangeSet containing values contained in the
// receiver but not in other. The result may split into multiple ranges that
// retain the original open/closed endpoint flags. The receiver and argument
// are not modified.
func (s *RangeSet) Difference(other *RangeSet) (*RangeSet, error) {
	return s.combined(other, "Difference", func(inA, inB bool) bool {
		return inA && !inB
	})
}

// eventGroup counts interval endpoints of each shape that land on one
// boundary value.
type eventGroup struct {
	openHigh   int // open upper endpoint: coverage stops *at* the value
	closedLow  int // closed lower endpoint: coverage starts *at* the value
	closedHigh int // closed upper endpoint: coverage stops *after* the value
	openLow    int // open lower endpoint: coverage starts *after* the value
}

// valueIndex locates a boundary value in the deduplicated sorted slice.
type valueIndex struct {
	intVals  map[int64]int
	timeVals map[time.Time]int
}

func (vi valueIndex) of(e RangeEndpoint) int {
	if e.kind == kindInt64 {
		return vi.intVals[e.intVal]
	}
	return vi.timeVals[e.timeVal]
}

// combineRanges runs a boundary sweep over two normalized range sets. The
// keep predicate decides whether a point covered by (inA, inB) belongs in the
// result, which expresses union, intersection and difference uniformly. The
// algorithm only inspects interval boundaries - no domain values are
// enumerated - so int64 extremes and arbitrary timestamps cost the same.
func combineRanges(a, b []Range, kind rangeKind, keep func(inA, inB bool) bool) []Range {
	values := make([]RangeEndpoint, 0, 2*(len(a)+len(b)))
	for _, rs := range [][]Range{a, b} {
		for _, r := range rs {
			if !r.low.infinite {
				values = append(values, r.low)
			}
			if !r.high.infinite {
				values = append(values, r.high)
			}
		}
	}
	sort.SliceStable(values, func(i, j int) bool {
		return compareEndpointValue(values[i], values[j]) < 0
	})
	unique := make([]RangeEndpoint, 0, len(values))
	for _, v := range values {
		if len(unique) == 0 || compareEndpointValue(unique[len(unique)-1], v) != 0 {
			unique = append(unique, v)
		}
	}

	index := valueIndex{}
	if kind == kindInt64 {
		index.intVals = make(map[int64]int, len(unique))
		for i, e := range unique {
			index.intVals[e.intVal] = i
		}
	} else {
		index.timeVals = make(map[time.Time]int, len(unique))
		for i, e := range unique {
			index.timeVals[e.timeVal] = i
		}
	}

	eventsA := make([]eventGroup, len(unique))
	eventsB := make([]eventGroup, len(unique))
	addEvents(a, index, eventsA)
	addEvents(b, index, eventsB)

	depthA := initialDepth(a)
	depthB := initialDepth(b)

	var out []Range
	open := false
	if keep(depthA > 0, depthB > 0) {
		out = append(out, Range{kind: kind, low: NegInfinityEndpoint()})
		open = true
	}

	for i, boundary := range unique {
		beforeA, beforeB := depthA > 0, depthB > 0

		atDepthA := depthA - eventsA[i].openHigh + eventsA[i].closedLow
		atDepthB := depthB - eventsB[i].openHigh + eventsB[i].closedLow
		afterDepthA := atDepthA - eventsA[i].closedHigh + eventsA[i].openLow
		afterDepthB := atDepthB - eventsB[i].closedHigh + eventsB[i].openLow

		beforeKeep := keep(beforeA, beforeB)
		atKeep := keep(atDepthA > 0, atDepthB > 0)
		afterKeep := keep(afterDepthA > 0, afterDepthB > 0)

		// Transitions evaluated at the boundary point itself.
		if beforeKeep != atKeep {
			if atKeep {
				out = append(out, Range{
					kind: kind,
					low:  boundary.finiteWith(true),
				})
				open = true
			} else {
				out[len(out)-1].high = boundary.finiteWith(false)
				open = false
			}
		}
		// Transitions evaluated immediately after the boundary point.
		if atKeep != afterKeep {
			if afterKeep {
				out = append(out, Range{
					kind: kind,
					low:  boundary.finiteWith(false),
				})
				open = true
			} else {
				out[len(out)-1].high = boundary.finiteWith(true)
				open = false
			}
		}

		depthA, depthB = afterDepthA, afterDepthB
	}

	if open {
		out[len(out)-1].high = PosInfinityEndpoint()
	}
	return out
}

func addEvents(ranges []Range, index valueIndex, groups []eventGroup) {
	for _, r := range ranges {
		if !r.low.infinite {
			i := index.of(r.low)
			if r.low.closed {
				groups[i].closedLow++
			} else {
				groups[i].openLow++
			}
		}
		if !r.high.infinite {
			i := index.of(r.high)
			if r.high.closed {
				groups[i].closedHigh++
			} else {
				groups[i].openHigh++
			}
		}
	}
}

func initialDepth(ranges []Range) int {
	depth := 0
	for _, r := range ranges {
		if r.low.infinite {
			depth++
		}
	}
	return depth
}

// rangeSetPredicate is the immutable Sqlizer returned by RangeSet.Contains.
type rangeSetPredicate struct {
	column string
	kind   rangeKind
	policy NullPolicy
	ranges []Range
}

// Contains returns a Sqlizer predicate matching rows whose registered column
// value belongs to the set. Every finite endpoint is emitted as a bound
// argument; the column identifier is the one captured through
// RegisterRangeColumn and is never built from external strings.
//
// The generated predicate is always wrapped in parentheses when it is
// compound, so combining it with further Where expressions using AND keeps
// the intended OR/AND grouping.
func (s *RangeSet) Contains() Sqlizer {
	ranges := make([]Range, len(s.ranges))
	copy(ranges, s.ranges)
	return rangeSetPredicate{
		column: s.column.name,
		kind:   s.kind,
		policy: s.policy,
		ranges: ranges,
	}
}

func endpointArg(e RangeEndpoint) interface{} {
	if e.kind == kindInt64 {
		return e.intVal
	}
	return e.timeVal
}

func isPointRange(r Range) bool {
	return !r.low.infinite && !r.high.infinite &&
		r.low.closed && r.high.closed &&
		compareEndpointValue(r.low, r.high) == 0
}

func (p rangeSetPredicate) countArgs() int {
	count := 0
	for _, r := range p.ranges {
		if isPointRange(r) {
			count++
			continue
		}
		if !r.low.infinite {
			count++
		}
		if !r.high.infinite {
			count++
		}
	}
	return count
}

func (p rangeSetPredicate) rangeClause(r Range) (string, []interface{}) {
	if isPointRange(r) {
		return p.column + " = ?", []interface{}{endpointArg(r.low)}
	}

	var clauses []string
	var args []interface{}
	if !r.low.infinite {
		op := ">"
		if r.low.closed {
			op = ">="
		}
		clauses = append(clauses, p.column+" "+op+" ?")
		args = append(args, endpointArg(r.low))
	}
	if !r.high.infinite {
		op := "<"
		if r.high.closed {
			op = "<="
		}
		clauses = append(clauses, p.column+" "+op+" ?")
		args = append(args, endpointArg(r.high))
	}
	clause := strings.Join(clauses, " AND ")
	if len(clauses) > 1 {
		clause = "(" + clause + ")"
	}
	return clause, args
}

// ToSql implements Sqlizer. It either returns a complete, executable
// predicate with all of its arguments or an error; a failed call never
// returns a partially built predicate.
func (p rangeSetPredicate) ToSql() (sql string, args []interface{}, err error) {
	if n := p.countArgs(); n > MaxRangeSetBoundArgs {
		return "", nil, fmt.Errorf(
			"range set predicate needs %d bound arguments, exceeding budget of %d",
			n, MaxRangeSetBoundArgs)
	}

	universal := len(p.ranges) == 1 &&
		p.ranges[0].low.infinite && p.ranges[0].high.infinite

	switch {
	case universal:
		if p.policy == IncludeNull {
			return sqlTrue, []interface{}{}, nil
		}
		return "(" + p.column + " IS NOT NULL)", []interface{}{}, nil
	case len(p.ranges) == 0:
		if p.policy == IncludeNull {
			return "(" + p.column + " IS NULL)", []interface{}{}, nil
		}
		return sqlFalse, []interface{}{}, nil
	}

	parts := make([]string, 0, len(p.ranges))
	for _, r := range p.ranges {
		clause, clauseArgs := p.rangeClause(r)
		parts = append(parts, clause)
		args = append(args, clauseArgs...)
	}
	// Each multi-clause range is already parenthesized, so a disjunction of
	// several ranges only needs one wrapping pair. A single simple clause is
	// safe unparenthesized in an outer AND chain; IncludeNull below adds the
	// wrapping required by its OR.
	body := strings.Join(parts, " OR ")
	if len(parts) > 1 {
		body = "(" + body + ")"
	}

	if p.policy == IncludeNull {
		sql = "(" + body + " OR " + p.column + " IS NULL)"
	} else {
		sql = body
	}

	if strings.Count(sql, "?") != len(args) {
		return "", nil, fmt.Errorf(
			"internal error: range set predicate placeholder/argument mismatch")
	}
	return sql, args, nil
}
