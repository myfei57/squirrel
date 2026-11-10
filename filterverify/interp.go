// Package filterverify cross-checks CompileFilter output against an
// independent, in-memory three-valued-logic interpreter and a real SQLite
// database driven by the compiled SQL.
package filterverify

import (
	"bytes"
	"encoding/json"
	"fmt"

	sqrl "github.com/Masterminds/squirrel"
)

// tri is a SQL three-valued-logic truth value. The ordering matters: AND is
// min, OR is max, NOT is 2-t, so UNKNOWN stays UNKNOWN under NOT instead of
// collapsing into a binary TRUE.
type tri int

const (
	triFalse   tri = 0
	triUnknown tri = 1
	triTrue    tri = 2
)

func (t tri) String() string {
	switch t {
	case triFalse:
		return "FALSE"
	case triUnknown:
		return "UNKNOWN"
	default:
		return "TRUE"
	}
}

// Row is one finite dataset row keyed by the external field name. A nil value
// represents SQL NULL.
type Row map[string]interface{}

// Eval evaluates a filter tree against one row using SQL three-valued logic.
// The evaluation rules are written independently of squirrel's compiler: they
// only consume the tree data structure.
func Eval(node *sqrl.Filter, row Row) (tri, error) {
	if node == nil {
		return triFalse, fmt.Errorf("nil node")
	}
	switch node.Op {
	case sqrl.OpAnd, sqrl.OpOr:
		if len(node.Children) == 0 {
			return triFalse, fmt.Errorf("logical node without children")
		}
		out := triTrue
		if node.Op == sqrl.OpOr {
			out = triFalse
		}
		for _, child := range node.Children {
			v, err := Eval(child, row)
			if err != nil {
				return triFalse, err
			}
			if node.Op == sqrl.OpAnd {
				if v < out {
					out = v
				}
			} else {
				if v > out {
					out = v
				}
			}
		}
		return out, nil
	case sqrl.OpNot:
		if len(node.Children) != 1 {
			return triFalse, fmt.Errorf("not requires one child")
		}
		v, err := Eval(node.Children[0], row)
		if err != nil {
			return triFalse, err
		}
		return 2 - v, nil
	case sqrl.OpIsNull, sqrl.OpIsNotNull:
		v, ok := row[node.Field]
		isNull := !ok || v == nil
		if node.Op == sqrl.OpIsNotNull {
			return b2tri(!isNull), nil
		}
		return b2tri(isNull), nil
	case sqrl.OpIn, sqrl.OpNotIn:
		v, err := evalIn(node, row)
		if err != nil {
			return triFalse, err
		}
		if node.Op == sqrl.OpNotIn {
			return 2 - v, nil
		}
		return v, nil
	default:
		return evalScalar(node, row)
	}
}

func b2tri(b bool) tri {
	if b {
		return triTrue
	}
	return triFalse
}

func evalScalar(node *sqrl.Filter, row Row) (tri, error) {
	left, ok := row[node.Field]
	if !ok || left == nil {
		// Any comparison with NULL is UNKNOWN, including for <> and >=.
		return triUnknown, nil
	}
	if node.Value == nil {
		return triFalse, fmt.Errorf("nil scalar value at %s", node.Field)
	}
	cmp, ok := compare(left, node.Value)
	if !ok {
		return triFalse, fmt.Errorf("incompatible types for %s: %T vs %T", node.Field, left, node.Value)
	}
	switch node.Op {
	case sqrl.OpEq:
		return b2tri(cmp == 0), nil
	case sqrl.OpNeq:
		// SQL: NULL <> x is UNKNOWN (handled above); otherwise plain inequality.
		return b2tri(cmp != 0), nil
	case sqrl.OpLt:
		return b2tri(cmp < 0), nil
	case sqrl.OpLte:
		return b2tri(cmp <= 0), nil
	case sqrl.OpGt:
		return b2tri(cmp > 0), nil
	case sqrl.OpGte:
		return b2tri(cmp >= 0), nil
	default:
		return triFalse, fmt.Errorf("unknown op %q", node.Op)
	}
}

func evalIn(node *sqrl.Filter, row Row) (tri, error) {
	elements := node.Values
	// Independent mirror of the schema's BytesAsList contract: a []byte Value
	// expands into one integer element per byte. The compiler-side rule is the
	// authority; this keeps the interpreter honest about the same input shape.
	if len(elements) == 0 {
		if packed, ok := node.Value.([]byte); ok {
			elements = make([]interface{}, len(packed))
			for i, b := range packed {
				elements[i] = int64(b)
			}
		}
	}

	left, ok := row[node.Field]
	if !ok || left == nil {
		// NULL IN (...) is always UNKNOWN, even for the empty list edge which
		// is encoded separately below; empty lists never contain NULL rows'
		// terms but SQL FALSE is the vacuous result, so handle empty first.
		if len(elements) == 0 {
			return triFalse, nil
		}
		return triUnknown, nil
	}
	if len(elements) == 0 {
		// Fixed empty-set semantics: x IN () is FALSE for every row.
		return triFalse, nil
	}
	out := triFalse
	for _, elem := range elements {
		if elem == nil {
			// x = NULL is UNKNOWN; it can only upgrade FALSE to UNKNOWN.
			if out == triFalse {
				out = triUnknown
			}
			continue
		}
		cmp, ok := compare(left, elem)
		if !ok {
			return triFalse, fmt.Errorf("incompatible IN element types for %s", node.Field)
		}
		if cmp == 0 {
			return triTrue, nil
		}
	}
	return out, nil
}

// compare returns -1/0/1 and ok=false when the types are not comparable.
// Integer-to-integer comparisons (including exact json.Number integers) stay
// exact; mixed int/float uses float64.
func compare(a, b interface{}) (int, bool) {
	ai, aIsInt := asInt64(a)
	bi, bIsInt := asInt64(b)
	switch {
	case aIsInt && bIsInt:
		switch {
		case ai < bi:
			return -1, true
		case ai > bi:
			return 1, true
		default:
			return 0, true
		}
	case isNumber(a) && isNumber(b):
		af := toFloat(a)
		bf := toFloat(b)
		switch {
		case af < bf:
			return -1, true
		case af > bf:
			return 1, true
		default:
			return 0, true
		}
	}

	as, aIsStr := a.(string)
	bs, bIsStr := b.(string)
	if aIsStr && bIsStr {
		return bytes.Compare([]byte(as), []byte(bs)), true
	}

	ab, aIsBytes := a.([]byte)
	bb, bIsBytes := b.([]byte)
	if aIsBytes && bIsBytes {
		return bytes.Compare(ab, bb), true
	}

	abool, aIsBool := a.(bool)
	bbool, bIsBool := b.(bool)
	if aIsBool && bIsBool {
		switch {
		case abool == bbool:
			return 0, true
		case !abool:
			return -1, true
		default:
			return 1, true
		}
	}
	return 0, false
}

func asInt64(v interface{}) (int64, bool) {
	switch n := v.(type) {
	case int64:
		return n, true
	case json.Number:
		if i, err := n.Int64(); err == nil {
			return i, true
		}
	}
	return 0, false
}

func isNumber(v interface{}) bool {
	switch v.(type) {
	case int64, float64, json.Number:
		return true
	}
	return false
}

func toFloat(v interface{}) float64 {
	switch n := v.(type) {
	case int64:
		return float64(n)
	case float64:
		return n
	case json.Number:
		f, _ := n.Float64()
		return f
	}
	return 0
}

// MatchingRows evaluates the tree over the dataset and returns the rids of
// rows where the predicate is TRUE: FALSE and UNKNOWN rows are both excluded,
// exactly as a SQL WHERE clause does.
func MatchingRows(node *sqrl.Filter, rows map[int]Row) ([]int, error) {
	var matched []int
	for rid := 0; rid < len(rows); rid++ {
		v, err := Eval(node, rows[rid])
		if err != nil {
			return nil, fmt.Errorf("rid %d: %w", rid, err)
		}
		if v == triTrue {
			matched = append(matched, rid)
		}
	}
	return matched, nil
}
