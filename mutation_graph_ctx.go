//go:build go1.8
// +build go1.8

package squirrel

import (
	"context"
	"database/sql"
	"fmt"
)

// ExecContext validates and runs the graph.
//
// When the graph was configured with RunWith, a transaction is opened on the
// database/sql DB and owned by the graph: any node, hook, scan or context
// failure rolls the whole transaction back and a successful run commits it. A
// commit failure is reported as uncertain through ErrMutationCommitUncertain
// and is never retried.
//
// When the graph was configured with RunWithTx, the caller-owned transaction is
// used as-is and the graph never commits or rolls it back.
//
// Nodes execute sequentially in a stable topological order on the single
// transaction connection. A failing node prevents every successor from running.
func (g *MutationGraph) ExecContext(ctx context.Context) (*MutationGraphResult, error) {
	return g.run(ctx)
}

func (g *MutationGraph) run(ctx context.Context) (*MutationGraphResult, error) {
	// All validation and factory materialization completes before any
	// transaction is opened or statement is sent to the database.
	ordered, err := g.prepare()
	if err != nil {
		return nil, err
	}

	ownTx := g.tx == nil
	var tx *sql.Tx
	if ownTx {
		tx, err = g.db.BeginTx(ctx, nil)
		if err != nil {
			return nil, &MutationError{Op: mutationOpBegin, Err: err}
		}
	} else {
		tx = g.tx
	}

	outputs := make(map[string]MutationOutput, len(ordered))
	for _, pn := range ordered {
		// Stop before touching the next node when the context is done.
		if err := ctx.Err(); err != nil {
			return nil, g.fail(tx, ownTx, pn, mutationOpContext, err)
		}

		if hook := pn.spec.Before; hook != nil {
			if err := hook(ctx, pn.spec.Name); err != nil {
				return nil, g.fail(tx, ownTx, pn, mutationOpHookBefore, err)
			}
		}

		// References are resolved at binding positions only; the query text
		// keeps its placeholders untouched.
		boundArgs, err := resolveArgs(pn.args, outputs)
		if err != nil {
			return nil, g.fail(tx, ownTx, pn, mutationOpReference, err)
		}

		var out MutationOutput
		var op string
		if pn.returning {
			out, op, err = g.runReturning(ctx, tx, pn, boundArgs)
		} else {
			out, op, err = g.runExec(ctx, tx, pn, boundArgs)
		}
		if err != nil {
			return nil, g.fail(tx, ownTx, pn, op, err)
		}

		if hook := pn.spec.After; hook != nil {
			if err := hook(ctx, pn.spec.Name); err != nil {
				return nil, g.fail(tx, ownTx, pn, mutationOpHookAfter, err)
			}
		}

		outputs[pn.spec.Name] = out
	}

	// Do not attempt a commit when the context is already canceled; roll back
	// so the transactional outcome stays explicit.
	if err := ctx.Err(); err != nil {
		return nil, g.fail(tx, ownTx, ordered[len(ordered)-1], mutationOpContext, err)
	}

	result := buildResult(outputs)
	if !ownTx {
		// Caller-owned transaction: lifecycle stays with the caller.
		return result, nil
	}
	if err := tx.Commit(); err != nil {
		// The commit may or may not have reached the server. Never retry and
		// never replay already executed statements.
		return nil, &MutationError{
			Op:  mutationOpCommit,
			Err: fmt.Errorf("%w: %v", ErrMutationCommitUncertain, err),
		}
	}
	return result, nil
}

// fail attributes cause to pn and rolls back an owned transaction. The original
// cause is always preserved; a rollback failure is attached as additional
// context instead of replacing it.
func (g *MutationGraph) fail(tx *sql.Tx, ownTx bool, pn *preparedNode, op string, cause error) error {
	me := &MutationError{Node: pn.spec.Name, Op: op, Err: cause}
	if !ownTx {
		return me
	}
	if rbErr := tx.Rollback(); rbErr != nil && rbErr != sql.ErrTxDone {
		me.RollbackErr = rbErr
	}
	return me
}

func (g *MutationGraph) runExec(ctx context.Context, tx *sql.Tx, pn *preparedNode, args []interface{}) (MutationOutput, string, error) {
	res, err := tx.ExecContext(ctx, pn.query, args...)
	if err != nil {
		return MutationOutput{}, mutationOpExec, err
	}

	out := MutationOutput{}
	if pn.spec.Outputs.RowsAffected {
		n, err := res.RowsAffected()
		if err != nil {
			return MutationOutput{}, mutationOpRowsAffected, err
		}
		out.hasRowsAffected = true
		out.rowsAffected = n
	}
	if pn.spec.Outputs.LastInsertId {
		id, err := res.LastInsertId()
		if err != nil {
			// Fail explicitly instead of silently continuing with id == 0.
			return MutationOutput{}, mutationOpLastInsertID,
				fmt.Errorf("%w: %v", ErrMutationLastInsertIdUnsupported, err)
		}
		out.hasLastInsertID = true
		out.lastInsertID = id
	}
	return out, "", nil
}

// runReturning executes a node through QueryContext, requires exactly one row,
// scans the declared columns and closes the rows before returning, so the
// transaction connection is free before the next node starts.
func (g *MutationGraph) runReturning(ctx context.Context, tx *sql.Tx, pn *preparedNode, args []interface{}) (MutationOutput, string, error) {
	rows, err := tx.QueryContext(ctx, pn.query, args...)
	if err != nil {
		return MutationOutput{}, mutationOpQuery, err
	}

	closed := false
	closeRows := func() error {
		if closed {
			return nil
		}
		closed = true
		return rows.Close()
	}
	defer func() { _ = closeRows() }()

	columns, err := rows.Columns()
	if err != nil {
		_ = closeRows()
		return MutationOutput{}, mutationOpScan, err
	}

	if !rows.Next() {
		if err := rows.Err(); err != nil {
			_ = closeRows()
			return MutationOutput{}, mutationOpScan, err
		}
		if err := closeRows(); err != nil {
			return MutationOutput{}, mutationOpRowsClose, err
		}
		return MutationOutput{}, mutationOpScan, ErrMutationNoRows
	}

	values := make([]interface{}, len(columns))
	dest := make([]interface{}, len(columns))
	for i := range values {
		dest[i] = &values[i]
	}
	if err := rows.Scan(dest...); err != nil {
		_ = closeRows()
		return MutationOutput{}, mutationOpScan, err
	}

	if rows.Next() {
		_ = closeRows()
		return MutationOutput{}, mutationOpScan, ErrMutationMultipleRows
	}
	if err := rows.Err(); err != nil {
		_ = closeRows()
		return MutationOutput{}, mutationOpScan, err
	}

	columnIndex := make(map[string]int, len(columns))
	for i, name := range columns {
		columnIndex[name] = i
	}

	out := MutationOutput{returning: make(map[string]interface{}, len(pn.spec.Outputs.Returning))}
	for _, column := range pn.spec.Outputs.Returning {
		idx, ok := columnIndex[column]
		if !ok {
			_ = closeRows()
			return MutationOutput{}, mutationOpScan, fmt.Errorf("RETURNING column %q is missing from the result of node %q", column, pn.spec.Name)
		}
		out.returning[column] = values[idx]
	}

	if err := closeRows(); err != nil {
		return MutationOutput{}, mutationOpRowsClose, err
	}
	return out, "", nil
}

func buildResult(outputs map[string]MutationOutput) *MutationGraphResult {
	copied := make(map[string]MutationOutput, len(outputs))
	for name, out := range outputs {
		if out.returning != nil {
			returning := make(map[string]interface{}, len(out.returning))
			for column, value := range out.returning {
				returning[column] = value
			}
			out.returning = returning
		}
		copied[name] = out
	}
	return &MutationGraphResult{outputs: copied}
}
