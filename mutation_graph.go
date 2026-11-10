package squirrel

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// DefaultMaxMutationNodes is the default upper bound enforced on the number of
// nodes accepted by a MutationGraph. It exists to keep validation, topological
// sorting and per-execution planning work bounded.
const DefaultMaxMutationNodes = 64

// MutationCapabilities declares what the target SQL dialect and driver are able
// to provide. The caller is responsible for declaring capabilities truthfully:
// the graph never probes the driver on its own and never invents a zero value
// when a capability is missing.
type MutationCapabilities struct {
	// LastInsertId reports whether sql.Result.LastInsertId is supported by the
	// driver (for example MySQL and SQLite). Drivers such as the standard
	// PostgreSQL drivers do not support it.
	LastInsertId bool

	// Returning reports whether the dialect supports the RETURNING clause on
	// INSERT/UPDATE/DELETE statements (for example PostgreSQL and SQLite >= 3.35).
	Returning bool
}

// MutationFactory builds the Sqlizer (an Insert/Update/Delete builder) for one
// graph node.
//
// Implementations must close over frozen inputs only: the factory is invoked
// once per graph execution and must return an independent builder every time,
// so that executions never share mutable state and a failed execution can never
// be replayed through stale builder state.
type MutationFactory func() (Sqlizer, error)

// MutationHook is invoked around a node execution. The hook receives the node
// name but never the bound arguments, so hook implementations cannot leak
// sensitive values. Returning an error aborts the graph: a Before failure means
// the node statement is never executed; an After failure means the statement
// already executed but the graph still aborts and rolls back an owned
// transaction.
type MutationHook func(ctx context.Context, nodeName string) error

// MutationOutputSpec declares which verified outputs a node exposes to its
// successors. Only outputs declared here can be referenced through a
// MutationRef.
//
// A node is executed either through Exec (LastInsertId/RowsAffected) or through
// Query (Returning); the two modes cannot be combined on the same node.
type MutationOutputSpec struct {
	// LastInsertId exposes sql.Result.LastInsertId of the node.
	LastInsertId bool

	// RowsAffected exposes sql.Result.RowsAffected of the node.
	RowsAffected bool

	// Returning lists the columns of the single row returned by the node's
	// RETURNING clause. The statement must return exactly one row.
	Returning []string
}

// MutationNode is one vertex of the mutation DAG.
type MutationNode struct {
	// Name uniquely identifies the node. It must be non-empty.
	Name string

	// DependsOn lists the names of predecessor nodes. Every referenced output
	// must come from a node listed here.
	DependsOn []string

	// Factory builds the statement executed for this node.
	Factory MutationFactory

	// Outputs declares the values successors are allowed to reference.
	Outputs MutationOutputSpec

	// Before is optionally run immediately before the node statement.
	Before MutationHook

	// After is optionally run immediately after the node statement succeeds.
	After MutationHook
}

// MutationRef is a placeholder for a predecessor output. It is designed to be
// passed wherever a bound argument is accepted (for example Insert.Values or
// Where/Expr arguments). It always materializes as a "?" placeholder in the
// generated SQL and is only resolved at the actual parameter binding position:
// return values are never interpolated into the SQL string.
type MutationRef interface {
	refNode() string
	refKind() mutationRefKind
	refColumn() string
}

type mutationRefKind int

const (
	mutationRefLastInsertId mutationRefKind = iota
	mutationRefRowsAffected
	mutationRefReturning
)

type mutationRef struct {
	node   string
	kind   mutationRefKind
	column string
}

func (r mutationRef) refNode() string          { return r.node }
func (r mutationRef) refKind() mutationRefKind { return r.kind }
func (r mutationRef) refColumn() string        { return r.column }

// LastInsertIdRef references the LastInsertId output of a predecessor node.
func LastInsertIdRef(node string) MutationRef {
	return mutationRef{node: node, kind: mutationRefLastInsertId}
}

// RowsAffectedRef references the RowsAffected output of a predecessor node.
func RowsAffectedRef(node string) MutationRef {
	return mutationRef{node: node, kind: mutationRefRowsAffected}
}

// ReturningRef references one column of the single RETURNING row of a
// predecessor node.
func ReturningRef(node, column string) MutationRef {
	return mutationRef{node: node, kind: mutationRefReturning, column: column}
}

// Sentinel errors. Callers can use errors.Is to branch on them; the wrapping
// MutationError preserves node identity and the underlying cause.
var (
	// ErrMutationGraphInvalid wraps every pre-execution validation failure.
	ErrMutationGraphInvalid = errors.New("squirrel: mutation graph is invalid")

	// ErrMutationNoRunner is returned when neither RunWith nor RunWithTx was
	// configured.
	ErrMutationNoRunner = errors.New("squirrel: mutation graph has no runner; use RunWith or RunWithTx")

	// ErrMutationCommitUncertain is returned when Commit of an owned
	// transaction fails. The transaction outcome is unknown. The graph never
	// retries the commit and never replays already executed statements.
	ErrMutationCommitUncertain = errors.New("squirrel: mutation transaction commit failed; the outcome is uncertain, do not retry automatically")

	// ErrMutationRollback is reported alongside the original failure when
	// rolling back an owned transaction also fails.
	ErrMutationRollback = errors.New("squirrel: mutation transaction rollback failed")

	// ErrMutationLastInsertIdUnsupported is returned when a driver rejects
	// LastInsertId instead of the graph fabricating a zero identifier.
	ErrMutationLastInsertIdUnsupported = errors.New("squirrel: driver does not support LastInsertId")

	// ErrMutationNoRows is returned when a RETURNING node yields no row.
	ErrMutationNoRows = errors.New("squirrel: RETURNING node did not return a row")

	// ErrMutationMultipleRows is returned when a RETURNING node yields more
	// than one row.
	ErrMutationMultipleRows = errors.New("squirrel: RETURNING node returned more than one row")

	// ErrMutationNullOutput is returned when a referenced RETURNING column
	// scanned as NULL instead of binding a fabricated zero/empty value.
	ErrMutationNullOutput = errors.New("squirrel: referenced RETURNING value is NULL")
)

// Mutation operation labels carried by MutationError.Op.
const (
	mutationOpValidate     = "validate"
	mutationOpBegin        = "begin"
	mutationOpFactory      = "factory"
	mutationOpBuild        = "build"
	mutationOpContext      = "context"
	mutationOpHookBefore   = "hook:before"
	mutationOpHookAfter    = "hook:after"
	mutationOpReference    = "reference"
	mutationOpExec         = "exec"
	mutationOpQuery        = "query"
	mutationOpScan         = "scan"
	mutationOpRowsAffected = "rows_affected"
	mutationOpLastInsertID = "last_insert_id"
	mutationOpRowsClose    = "rows_close"
	mutationOpCommit       = "commit"
)

// MutationError wraps a graph failure with the responsible node and the
// operation that failed. It never contains bound arguments.
type MutationError struct {
	// Node is the node the failure is attributed to. It is empty for
	// graph-level validation or runner failures.
	Node string

	// Op is one of the mutationOp* constants.
	Op string

	// Err is the underlying cause.
	Err error

	// RollbackErr is set when an owned transaction had to be rolled back due
	// to Err and the rollback itself also failed.
	RollbackErr error
}

func (e *MutationError) Error() string {
	var b strings.Builder
	b.WriteString("squirrel mutation graph")
	if e.Node != "" {
		fmt.Fprintf(&b, " node %q", e.Node)
	}
	if e.Op != "" {
		fmt.Fprintf(&b, " failed during %s", e.Op)
	}
	if e.Err != nil {
		fmt.Fprintf(&b, ": %v", e.Err)
	}
	if e.RollbackErr != nil {
		fmt.Fprintf(&b, "; additionally %v: %v", ErrMutationRollback, e.RollbackErr)
	}
	return b.String()
}

// Unwrap returns the underlying cause so errors.Is/errors.As keep working.
func (e *MutationError) Unwrap() error { return e.Err }

// Is lets errors.Is match ErrMutationRollback when a rollback also failed.
func (e *MutationError) Is(target error) bool {
	return target == ErrMutationRollback && e.RollbackErr != nil
}

// MutationOutput holds the verified outputs of one executed node.
type MutationOutput struct {
	hasLastInsertID bool
	lastInsertID    int64

	hasRowsAffected bool
	rowsAffected    int64

	returning map[string]interface{}
}

// LastInsertId returns the verified LastInsertId. The boolean is false when the
// node did not declare that output.
func (o MutationOutput) LastInsertId() (int64, bool) {
	return o.lastInsertID, o.hasLastInsertID
}

// RowsAffected returns the verified RowsAffected. The boolean is false when the
// node did not declare that output.
func (o MutationOutput) RowsAffected() (int64, bool) {
	return o.rowsAffected, o.hasRowsAffected
}

// Returning returns the scanned value of one declared RETURNING column. The
// boolean is false when the column was not declared.
func (o MutationOutput) Returning(column string) (interface{}, bool) {
	v, ok := o.returning[column]
	return v, ok
}

// MutationGraphResult is the owned, verified output of a successful graph run.
type MutationGraphResult struct {
	outputs map[string]MutationOutput
}

// Node returns the output of one node.
func (r *MutationGraphResult) Node(name string) (MutationOutput, bool) {
	o, ok := r.outputs[name]
	return o, ok
}

// NodeNames returns the names of all executed nodes, sorted for stable
// iteration.
func (r *MutationGraphResult) NodeNames() []string {
	names := make([]string, 0, len(r.outputs))
	for name := range r.outputs {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// MutationGraph organizes a finite set of Insert/Update/Delete constructors as
// a directed acyclic graph and executes them in a single transaction, in a
// stable topological order. Successor statements may reference outputs that
// predecessors explicitly declared; references are resolved at parameter
// binding time only.
type MutationGraph struct {
	nodes    []MutationNode
	maxNodes int
	caps     MutationCapabilities

	db *sql.DB
	tx *sql.Tx
}

// NewMutationGraph creates an empty graph. By default it allows up to
// DefaultMaxMutationNodes nodes.
func NewMutationGraph() *MutationGraph {
	return &MutationGraph{maxNodes: DefaultMaxMutationNodes}
}

// AddNode appends a node. The node and its declaration slices are copied, so
// later caller-side mutations do not affect the graph. Structural validation
// happens before execution.
func (g *MutationGraph) AddNode(node MutationNode) *MutationGraph {
	copied := MutationNode{
		Name:    node.Name,
		Factory: node.Factory,
		Before:  node.Before,
		After:   node.After,
	}
	if node.DependsOn != nil {
		copied.DependsOn = append([]string(nil), node.DependsOn...)
	}
	copied.Outputs = MutationOutputSpec{
		LastInsertId: node.Outputs.LastInsertId,
		RowsAffected: node.Outputs.RowsAffected,
	}
	if node.Outputs.Returning != nil {
		copied.Outputs.Returning = append([]string(nil), node.Outputs.Returning...)
	}
	g.nodes = append(g.nodes, copied)
	return g
}

// MaxNodes overrides the maximum number of nodes. A non-positive value restores
// the default.
func (g *MutationGraph) MaxNodes(n int) *MutationGraph {
	if n <= 0 {
		g.maxNodes = DefaultMaxMutationNodes
	} else {
		g.maxNodes = n
	}
	return g
}

// Capabilities declares the dialect/driver capabilities of the runner.
func (g *MutationGraph) Capabilities(caps MutationCapabilities) *MutationGraph {
	g.caps = caps
	return g
}

// RunWith configures the graph to open and own a transaction on db. The
// transaction is committed on full success and rolled back when any node, hook
// or context fails.
func (g *MutationGraph) RunWith(db *sql.DB) *MutationGraph {
	g.db = db
	g.tx = nil
	return g
}

// RunWithTx configures the graph to run inside an existing caller-owned
// transaction. The graph never commits or rolls it back: the caller stays in
// control of the transaction lifecycle.
func (g *MutationGraph) RunWithTx(tx *sql.Tx) *MutationGraph {
	g.tx = tx
	g.db = nil
	return g
}

// Exec runs the graph with context.Background.
func (g *MutationGraph) Exec() (*MutationGraphResult, error) {
	return g.run(context.Background())
}

// preparedNode is the frozen, validated execution plan for one node.
type preparedNode struct {
	spec      MutationNode
	query     string
	args      []interface{}
	returning bool
}

// prepare runs every pre-execution check and materializes all factories before
// any database connection is touched or any transaction begins.
func (g *MutationGraph) prepare() ([]*preparedNode, error) {
	if g.db == nil && g.tx == nil {
		return nil, &MutationError{Op: mutationOpValidate, Err: fmt.Errorf("%w: no runner configured", ErrMutationNoRunner)}
	}
	if len(g.nodes) == 0 {
		return nil, g.validationError("graph contains no nodes")
	}
	if len(g.nodes) > g.maxNodes {
		return nil, g.validationError("graph has %d nodes which exceeds the limit of %d", len(g.nodes), g.maxNodes)
	}

	indexByName := make(map[string]int, len(g.nodes))
	for i, node := range g.nodes {
		if strings.TrimSpace(node.Name) == "" {
			return nil, g.validationError("node at index %d has an empty name", i)
		}
		if _, exists := indexByName[node.Name]; exists {
			return nil, g.validationError("duplicate node name %q", node.Name)
		}
		if node.Factory == nil {
			return nil, g.validationError("node %q has no factory", node.Name)
		}
		indexByName[node.Name] = i
	}

	// Validate dependency declarations and output specs.
	indegree := make([]int, len(g.nodes))
	dependents := make([][]int, len(g.nodes))
	seenDeps := make(map[string]struct{})
	for i, node := range g.nodes {
		if len(node.Outputs.Returning) > 0 && (node.Outputs.LastInsertId || node.Outputs.RowsAffected) {
			return nil, g.validationError("node %q cannot combine RETURNING outputs with LastInsertId or RowsAffected", node.Name)
		}
		if node.Outputs.LastInsertId && !g.caps.LastInsertId {
			return nil, g.validationError("node %q declares LastInsertId but the driver capability is not enabled", node.Name)
		}
		if len(node.Outputs.Returning) > 0 && !g.caps.Returning {
			return nil, g.validationError("node %q declares RETURNING outputs but the dialect capability is not enabled", node.Name)
		}
		if dup := firstDuplicate(node.Outputs.Returning); dup != "" {
			return nil, g.validationError("node %q declares RETURNING column %q more than once", node.Name, dup)
		}

		for key := range seenDeps {
			delete(seenDeps, key)
		}
		for _, dep := range node.DependsOn {
			if dep == node.Name {
				return nil, g.validationError("node %q depends on itself", node.Name)
			}
			if _, ok := indexByName[dep]; !ok {
				return nil, g.validationError("node %q depends on unknown node %q", node.Name, dep)
			}
			if _, dup := seenDeps[dep]; dup {
				return nil, g.validationError("node %q lists dependency %q more than once", node.Name, dep)
			}
			seenDeps[dep] = struct{}{}
			depIdx := indexByName[dep]
			dependents[depIdx] = append(dependents[depIdx], i)
			indegree[i]++
		}
	}

	order, cycle := stableTopoOrder(g.nodes, indegree, dependents)
	if cycle != nil {
		return nil, g.validationError("dependency cycle detected involving nodes %s", strings.Join(cycle, " -> "))
	}

	// Materialize the frozen plan: every factory runs exactly once per
	// execution and every statement is rendered before the transaction starts.
	prepared := make([]*preparedNode, len(g.nodes))
	for i, node := range g.nodes {
		sqlizer, err := node.Factory()
		if err != nil {
			return nil, &MutationError{Node: node.Name, Op: mutationOpFactory, Err: err}
		}
		if sqlizer == nil {
			return nil, &MutationError{Node: node.Name, Op: mutationOpFactory, Err: errors.New("factory returned a nil Sqlizer")}
		}
		query, args, err := sqlizer.ToSql()
		if err != nil {
			return nil, &MutationError{Node: node.Name, Op: mutationOpBuild, Err: err}
		}
		prepared[i] = &preparedNode{
			spec:      node,
			query:     query,
			args:      args,
			returning: len(node.Outputs.Returning) > 0,
		}
	}

	// Validate every reference against predecessor declarations.
	for _, pn := range prepared {
		deps := make(map[string]struct{}, len(pn.spec.DependsOn))
		for _, dep := range pn.spec.DependsOn {
			deps[dep] = struct{}{}
		}
		var verify func(args []interface{}) error
		verify = func(args []interface{}) error {
			for _, arg := range args {
				switch v := arg.(type) {
				case MutationRef:
					if err := g.verifyRef(pn, v, deps); err != nil {
						return err
					}
				case []interface{}:
					if err := verify(v); err != nil {
						return err
					}
				}
			}
			return nil
		}
		if err := verify(pn.args); err != nil {
			return nil, err
		}
	}

	ordered := make([]*preparedNode, 0, len(prepared))
	for _, idx := range order {
		ordered = append(ordered, prepared[idx])
	}
	return ordered, nil
}

func (g *MutationGraph) verifyRef(pn *preparedNode, ref MutationRef, deps map[string]struct{}) error {
	target := ref.refNode()
	if _, ok := deps[target]; !ok {
		return g.validationError("node %q references node %q which is not listed in DependsOn", pn.spec.Name, target)
	}
	targetSpec := g.nodes[indexOf(g.nodes, target)].Outputs
	switch ref.refKind() {
	case mutationRefLastInsertId:
		if !targetSpec.LastInsertId {
			return g.validationError("node %q references LastInsertId of node %q which did not declare that output", pn.spec.Name, target)
		}
	case mutationRefRowsAffected:
		if !targetSpec.RowsAffected {
			return g.validationError("node %q references RowsAffected of node %q which did not declare that output", pn.spec.Name, target)
		}
	case mutationRefReturning:
		column := ref.refColumn()
		if !containsString(targetSpec.Returning, column) {
			return g.validationError("node %q references RETURNING column %q of node %q which did not declare it", pn.spec.Name, column, target)
		}
	}
	return nil
}

func (g *MutationGraph) validationError(format string, args ...interface{}) error {
	return &MutationError{
		Op:  mutationOpValidate,
		Err: fmt.Errorf("%w: %s", ErrMutationGraphInvalid, fmt.Sprintf(format, args...)),
	}
}

// stableTopoOrder runs Kahn's algorithm breaking ties by insertion index, so
// the order only depends on the declared graph and never on map iteration.
func stableTopoOrder(nodes []MutationNode, indegree []int, dependents [][]int) ([]int, []string) {
	remaining := append([]int(nil), indegree...)
	emitted := make([]int, 0, len(nodes))
	for len(emitted) < len(nodes) {
		pick := -1
		for i := range nodes {
			if remaining[i] == 0 {
				pick = i
				break
			}
		}
		if pick == -1 {
			var cyclic []string
			for i, n := range remaining {
				if n > 0 {
					cyclic = append(cyclic, nodes[i].Name)
				}
			}
			sort.Strings(cyclic)
			return nil, cyclic
		}
		remaining[pick] = -1
		emitted = append(emitted, pick)
		for _, next := range dependents[pick] {
			remaining[next]--
		}
	}
	return emitted, nil
}

// resolveArgs produces a fresh argument slice for one execution, replacing
// MutationRef values at their binding positions with verified predecessor
// outputs. The SQL text is never touched.
func resolveArgs(args []interface{}, outputs map[string]MutationOutput) ([]interface{}, error) {
	resolved := make([]interface{}, len(args))
	for i, arg := range args {
		switch v := arg.(type) {
		case MutationRef:
			value, err := resolveRef(v, outputs)
			if err != nil {
				return nil, err
			}
			resolved[i] = value
		case []interface{}:
			nested, err := resolveArgs(v, outputs)
			if err != nil {
				return nil, err
			}
			resolved[i] = nested
		default:
			resolved[i] = arg
		}
	}
	return resolved, nil
}

func resolveRef(ref MutationRef, outputs map[string]MutationOutput) (interface{}, error) {
	out, ok := outputs[ref.refNode()]
	if !ok {
		return nil, fmt.Errorf("output of predecessor node %q is not available", ref.refNode())
	}
	switch ref.refKind() {
	case mutationRefLastInsertId:
		if !out.hasLastInsertID {
			return nil, fmt.Errorf("%w for node %q", ErrMutationLastInsertIdUnsupported, ref.refNode())
		}
		return out.lastInsertID, nil
	case mutationRefRowsAffected:
		if !out.hasRowsAffected {
			return nil, fmt.Errorf("RowsAffected output of node %q is not available", ref.refNode())
		}
		return out.rowsAffected, nil
	case mutationRefReturning:
		value, ok := out.returning[ref.refColumn()]
		if !ok {
			return nil, fmt.Errorf("RETURNING column %q of node %q is not available", ref.refColumn(), ref.refNode())
		}
		if value == nil {
			return nil, fmt.Errorf("%w: node %q column %q", ErrMutationNullOutput, ref.refNode(), ref.refColumn())
		}
		return value, nil
	default:
		return nil, fmt.Errorf("unknown mutation reference kind for node %q", ref.refNode())
	}
}

func firstDuplicate(values []string) string {
	seen := make(map[string]struct{}, len(values))
	for _, v := range values {
		if _, ok := seen[v]; ok {
			return v
		}
		seen[v] = struct{}{}
	}
	return ""
}

func containsString(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

func indexOf(nodes []MutationNode, name string) int {
	for i := range nodes {
		if nodes[i].Name == name {
			return i
		}
	}
	return -1
}
