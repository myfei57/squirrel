package squirrel

import (
	"context"
	"database/sql/driver"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var errFakeBoom = errors.New("boom: simulated failure")

// Frozen builder factories: every invocation returns an independent builder.
func parentInsertFactory() MutationFactory {
	return func() (Sqlizer, error) {
		return Insert("parent").Columns("name").Values("alice"), nil
	}
}

func childInsertFactory() MutationFactory {
	return func() (Sqlizer, error) {
		return Insert("child").Columns("parent_id", "note").Values(LastInsertIdRef("parent"), "toy"), nil
	}
}

func expectedSQL(t *testing.T, s Sqlizer) string {
	t.Helper()
	query, _, err := s.ToSql()
	require.NoError(t, err)
	return query
}

func mustMutationError(t *testing.T, err error) *MutationError {
	t.Helper()
	require.Error(t, err)
	me := &MutationError{}
	require.True(t, errors.As(err, &me), "expected *MutationError, got %T: %v", err, err)
	return me
}

func callsFor(calls []fakeCall, query string) []fakeCall {
	var out []fakeCall
	for _, c := range calls {
		if c.query == query {
			out = append(out, c)
		}
	}
	return out
}

func TestMutationGraphParentChildLastInsertId(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	parentSQL := expectedSQL(t, Insert("parent").Columns("name").Values("alice"))
	childSQL := expectedSQL(t, Insert("child").Columns("parent_id", "note").Values(LastInsertIdRef("parent"), "toy"))
	rec.idFor[parentSQL] = 42

	g := NewMutationGraph().
		Capabilities(MutationCapabilities{LastInsertId: true}).
		RunWith(db).
		AddNode(MutationNode{
			Name:    "parent",
			Factory: parentInsertFactory(),
			Outputs: MutationOutputSpec{LastInsertId: true},
		}).
		AddNode(MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory:   childInsertFactory(),
		})

	result, err := g.ExecContext(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	calls := rec.snapshotCalls()
	require.Len(t, calls, 2)

	// Stable order: parent first.
	assert.Equal(t, parentSQL, calls[0].query)
	assert.Equal(t, childSQL, calls[1].query)

	// SQL text keeps placeholders: return values are never interpolated.
	assert.Contains(t, calls[1].query, "?")
	assert.NotContains(t, calls[1].query, "42")

	// Reference resolved at the binding position only.
	assert.Equal(t, []interface{}{"alice"}, calls[0].args)
	assert.Equal(t, []interface{}{int64(42), "toy"}, calls[1].args)

	parentOut, present := result.Node("parent")
	assert.True(t, present)
	id, ok := parentOut.LastInsertId()
	assert.True(t, ok)
	assert.Equal(t, int64(42), id)
	childOut, present := result.Node("child")
	assert.True(t, present)
	_, ok = childOut.LastInsertId()
	assert.False(t, ok)
	assert.Equal(t, []string{"child", "parent"}, result.NodeNames())

	begins, commits, rollbacks, _ := rec.counters()
	assert.Equal(t, 1, begins)
	assert.Equal(t, 1, commits)
	assert.Equal(t, 0, rollbacks)
}

func TestMutationGraphRowsAffectedRef(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	mark := Update("parent").Set("name", "bob").Where("id = ?", 1)
	cleanup := Delete("child").Where("rid = ?", RowsAffectedRef("mark"))
	markSQL := expectedSQL(t, mark)
	cleanupSQL := expectedSQL(t, cleanup)
	rec.rowsFor[markSQL] = 3

	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{
			Name:    "mark",
			Factory: func() (Sqlizer, error) { return mark, nil },
			Outputs: MutationOutputSpec{RowsAffected: true},
		}).
		AddNode(MutationNode{
			Name:      "cleanup",
			DependsOn: []string{"mark"},
			Factory:   func() (Sqlizer, error) { return cleanup, nil },
		})

	result, err := g.Exec()
	require.NoError(t, err)

	markOut, present := result.Node("mark")
	assert.True(t, present)
	n, ok := markOut.RowsAffected()
	assert.True(t, ok)
	assert.Equal(t, int64(3), n)

	calls := rec.snapshotCalls()
	require.Len(t, calls, 2)
	assert.Equal(t, cleanupSQL, calls[1].query)
	assert.Equal(t, []interface{}{int64(3)}, calls[1].args)
}

func TestMutationGraphReturningRef(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	parent := Insert("parent").Columns("name").Values("alice").Suffix("RETURNING id")
	child := Insert("child").Columns("parent_id", "note").Values(ReturningRef("parent", "id"), "toy")
	parentSQL := expectedSQL(t, parent)
	rec.returningCols[parentSQL] = []string{"id"}
	rec.returningRows[parentSQL] = [][]driver.Value{{int64(7)}}

	g := NewMutationGraph().
		Capabilities(MutationCapabilities{Returning: true}).
		RunWith(db).
		AddNode(MutationNode{
			Name:    "parent",
			Factory: func() (Sqlizer, error) { return parent, nil },
			Outputs: MutationOutputSpec{Returning: []string{"id"}},
		}).
		AddNode(MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory:   func() (Sqlizer, error) { return child, nil },
		})

	result, err := g.ExecContext(context.Background())
	require.NoError(t, err)

	parentOut, present := result.Node("parent")
	assert.True(t, present)
	id, ok := parentOut.Returning("id")
	assert.True(t, ok)
	assert.Equal(t, int64(7), id)

	calls := rec.snapshotCalls()
	require.Len(t, calls, 2)
	assert.Equal(t, "query", calls[0].kind)
	assert.Equal(t, "exec", calls[1].kind)
	assert.Equal(t, []interface{}{int64(7), "toy"}, calls[1].args)
}

func TestMutationGraphRefInWhereClause(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	parentSQL := expectedSQL(t, Insert("parent").Columns("name").Values("alice"))
	rec.idFor[parentSQL] = 99
	update := Update("child").Set("note", "x").Where("parent_id = ?", LastInsertIdRef("parent"))
	updateSQL := expectedSQL(t, update)

	g := NewMutationGraph().
		Capabilities(MutationCapabilities{LastInsertId: true}).
		RunWith(db).
		AddNode(MutationNode{
			Name:    "parent",
			Factory: parentInsertFactory(),
			Outputs: MutationOutputSpec{LastInsertId: true},
		}).
		AddNode(MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory:   func() (Sqlizer, error) { return update, nil },
		})

	_, err := g.ExecContext(context.Background())
	require.NoError(t, err)

	calls := rec.snapshotCalls()
	require.Len(t, calls, 2)
	assert.Equal(t, updateSQL, calls[1].query)
	assert.Contains(t, calls[1].query, "?")
	assert.Equal(t, []interface{}{"x", int64(99)}, calls[1].args)
}

func TestMutationGraphIndependentNodesAndStableOrder(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	first := Insert("a").Values(1)
	second := Insert("b").Values(2)
	firstSQL := expectedSQL(t, first)
	secondSQL := expectedSQL(t, second)

	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{Name: "first", Factory: func() (Sqlizer, error) { return first, nil }}).
		AddNode(MutationNode{Name: "second", Factory: func() (Sqlizer, error) { return second, nil }})

	_, err := g.ExecContext(context.Background())
	require.NoError(t, err)

	calls := rec.snapshotCalls()
	require.Len(t, calls, 2)
	assert.Equal(t, firstSQL, calls[0].query)
	assert.Equal(t, secondSQL, calls[1].query)

	begins, commits, _, _ := rec.counters()
	assert.Equal(t, 1, begins)
	assert.Equal(t, 1, commits)
}

func TestMutationGraphDiamondTopoOrder(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	queries := map[string]string{}
	factory := func(name string) MutationFactory {
		b := Insert(name).Values(1)
		queries[name] = expectedSQL(t, b)
		return func() (Sqlizer, error) { return b, nil }
	}

	// Insertion order intentionally differs from dependency order.
	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{Name: "leaf1", DependsOn: []string{"root"}, Factory: factory("leaf1")}).
		AddNode(MutationNode{Name: "leaf2", DependsOn: []string{"root"}, Factory: factory("leaf2")}).
		AddNode(MutationNode{Name: "root", Factory: factory("root")}).
		AddNode(MutationNode{Name: "join", DependsOn: []string{"leaf1", "leaf2"}, Factory: factory("join")})

	_, err := g.ExecContext(context.Background())
	require.NoError(t, err)

	calls := rec.snapshotCalls()
	require.Len(t, calls, 4)
	assert.Equal(t, queries["root"], calls[0].query)
	assert.Equal(t, queries["leaf1"], calls[1].query)
	assert.Equal(t, queries["leaf2"], calls[2].query)
	assert.Equal(t, queries["join"], calls[3].query)
}

func TestMutationGraphValidation(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	validFactory := func() (Sqlizer, error) { return Insert("t").Values(1), nil }
	refFactory := func() (Sqlizer, error) {
		return Insert("t").Values(LastInsertIdRef("p")), nil
	}

	cases := []struct {
		name  string
		graph func() *MutationGraph
	}{
		{
			name: "no runner",
			graph: func() *MutationGraph {
				return NewMutationGraph().AddNode(MutationNode{Name: "a", Factory: validFactory})
			},
		},
		{
			name: "empty graph",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db)
			},
		},
		{
			name: "empty node name",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "  ", Factory: validFactory})
			},
		},
		{
			name: "duplicate node name",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "a", Factory: validFactory}).
					AddNode(MutationNode{Name: "a", Factory: validFactory})
			},
		},
		{
			name: "nil factory",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "a"})
			},
		},
		{
			name: "unknown dependency",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "a", DependsOn: []string{"ghost"}, Factory: validFactory})
			},
		},
		{
			name: "self dependency",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "a", DependsOn: []string{"a"}, Factory: validFactory})
			},
		},
		{
			name: "duplicate dependency",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "a", Factory: validFactory}).
					AddNode(MutationNode{Name: "b", DependsOn: []string{"a", "a"}, Factory: validFactory})
			},
		},
		{
			name: "cycle",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "a", DependsOn: []string{"c"}, Factory: validFactory}).
					AddNode(MutationNode{Name: "b", DependsOn: []string{"a"}, Factory: validFactory}).
					AddNode(MutationNode{Name: "c", DependsOn: []string{"b"}, Factory: validFactory})
			},
		},
		{
			name: "max nodes exceeded",
			graph: func() *MutationGraph {
				g := NewMutationGraph().MaxNodes(2).RunWith(db)
				for _, n := range []string{"a", "b", "c"} {
					g.AddNode(MutationNode{Name: n, Factory: validFactory})
				}
				return g
			},
		},
		{
			name: "reference to non dependency",
			graph: func() *MutationGraph {
				return NewMutationGraph().
					Capabilities(MutationCapabilities{LastInsertId: true}).RunWith(db).
					AddNode(MutationNode{Name: "p", Factory: validFactory, Outputs: MutationOutputSpec{LastInsertId: true}}).
					AddNode(MutationNode{Name: "c", Factory: refFactory})
			},
		},
		{
			name: "undeclared last insert id",
			graph: func() *MutationGraph {
				return NewMutationGraph().
					Capabilities(MutationCapabilities{LastInsertId: true}).RunWith(db).
					AddNode(MutationNode{Name: "p", Factory: validFactory}).
					AddNode(MutationNode{Name: "c", DependsOn: []string{"p"}, Factory: refFactory})
			},
		},
		{
			name: "missing last insert id capability",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{Name: "p", Factory: validFactory, Outputs: MutationOutputSpec{LastInsertId: true}})
			},
		},
		{
			name: "missing returning capability",
			graph: func() *MutationGraph {
				return NewMutationGraph().RunWith(db).
					AddNode(MutationNode{
						Name:    "p",
						Factory: validFactory,
						Outputs: MutationOutputSpec{Returning: []string{"id"}},
					})
			},
		},
		{
			name: "mixed output modes",
			graph: func() *MutationGraph {
				return NewMutationGraph().
					Capabilities(MutationCapabilities{LastInsertId: true, Returning: true}).RunWith(db).
					AddNode(MutationNode{
						Name:    "p",
						Factory: validFactory,
						Outputs: MutationOutputSpec{LastInsertId: true, Returning: []string{"id"}},
					})
			},
		},
		{
			name: "duplicate returning column",
			graph: func() *MutationGraph {
				return NewMutationGraph().
					Capabilities(MutationCapabilities{Returning: true}).RunWith(db).
					AddNode(MutationNode{
						Name:    "p",
						Factory: validFactory,
						Outputs: MutationOutputSpec{Returning: []string{"id", "id"}},
					})
			},
		},
		{
			name: "undeclared returning column reference",
			graph: func() *MutationGraph {
				return NewMutationGraph().
					Capabilities(MutationCapabilities{Returning: true}).RunWith(db).
					AddNode(MutationNode{Name: "p", Factory: validFactory, Outputs: MutationOutputSpec{Returning: []string{"id"}}}).
					AddNode(MutationNode{
						Name:      "c",
						DependsOn: []string{"p"},
						Factory: func() (Sqlizer, error) {
							return Insert("t").Values(ReturningRef("p", "missing")), nil
						},
					})
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.graph().ExecContext(context.Background())
			me := mustMutationError(t, err)
			assert.Equal(t, mutationOpValidate, me.Op)
			if !errors.Is(err, ErrMutationNoRunner) {
				assert.True(t, errors.Is(err, ErrMutationGraphInvalid), "expected ErrMutationGraphInvalid, got %v", err)
			}
			begins, _, _, _ := rec.counters()
			assert.Equal(t, 0, begins, "validation must not open a transaction")
		})
	}
}

func TestMutationGraphFactoryAndBuildErrors(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{
			Name:    "bad-factory",
			Factory: func() (Sqlizer, error) { return nil, errFakeBoom },
		})
	_, err := g.ExecContext(context.Background())
	me := mustMutationError(t, err)
	assert.Equal(t, "bad-factory", me.Node)
	assert.Equal(t, mutationOpFactory, me.Op)
	assert.True(t, errors.Is(err, errFakeBoom))

	g2 := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{
			Name: "bad-sql",
			Factory: func() (Sqlizer, error) {
				return Insert(""), nil // missing table
			},
		})
	_, err = g2.ExecContext(context.Background())
	me = mustMutationError(t, err)
	assert.Equal(t, "bad-sql", me.Node)
	assert.Equal(t, mutationOpBuild, me.Op)

	begins, _, _, _ := rec.counters()
	assert.Equal(t, 0, begins)
}

func TestMutationGraphFailureStopsSuccessorsAndRollsBack(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	parentSQL := expectedSQL(t, Insert("parent").Columns("name").Values("alice"))
	childSQL := expectedSQL(t, Insert("child").Columns("parent_id", "note").Values(LastInsertIdRef("parent"), "toy"))
	grandSQL := expectedSQL(t, Insert("grand").Values(1))
	rec.idFor[parentSQL] = 42
	rec.execErr[childSQL] = errFakeBoom

	g := NewMutationGraph().
		Capabilities(MutationCapabilities{LastInsertId: true}).
		RunWith(db).
		AddNode(MutationNode{Name: "parent", Factory: parentInsertFactory(), Outputs: MutationOutputSpec{LastInsertId: true}}).
		AddNode(MutationNode{Name: "child", DependsOn: []string{"parent"}, Factory: childInsertFactory()}).
		AddNode(MutationNode{Name: "grand", DependsOn: []string{"child"}, Factory: func() (Sqlizer, error) {
			return Insert("grand").Values(1), nil
		}})

	_, err := g.ExecContext(context.Background())
	me := mustMutationError(t, err)
	assert.Equal(t, "child", me.Node)
	assert.Equal(t, mutationOpExec, me.Op)
	assert.True(t, errors.Is(err, errFakeBoom))

	calls := rec.snapshotCalls()
	assert.Len(t, callsFor(calls, parentSQL), 1)
	assert.Len(t, callsFor(calls, childSQL), 1)
	assert.Empty(t, callsFor(calls, grandSQL), "successors of a failed node must not execute")

	begins, commits, rollbacks, _ := rec.counters()
	assert.Equal(t, 1, begins)
	assert.Equal(t, 0, commits)
	assert.Equal(t, 1, rollbacks)
}

func TestMutationGraphCommitUncertain(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	rec.commitErr = driver.ErrBadConn

	parentSQL := expectedSQL(t, Insert("parent").Values(1))
	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{Name: "parent", Factory: func() (Sqlizer, error) {
			return Insert("parent").Values(1), nil
		}})

	result, err := g.ExecContext(context.Background())
	assert.Nil(t, result)
	me := mustMutationError(t, err)
	assert.Equal(t, mutationOpCommit, me.Op)
	assert.True(t, errors.Is(err, ErrMutationCommitUncertain))

	// Exactly one commit attempt, no replay of statements.
	_, commits, _, _ := rec.counters()
	assert.Equal(t, 1, commits)
	assert.Len(t, callsFor(rec.snapshotCalls(), parentSQL), 1)
}

func TestMutationGraphRollbackFailure(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	childSQL := expectedSQL(t, Insert("child").Values(1))
	rec.execErr[childSQL] = errFakeBoom
	rollbackErr := errors.New("rollback broken")
	rec.rollbackErr = rollbackErr

	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{Name: "parent", Factory: func() (Sqlizer, error) { return Insert("parent").Values(1), nil }}).
		AddNode(MutationNode{Name: "child", DependsOn: []string{"parent"}, Factory: func() (Sqlizer, error) {
			return Insert("child").Values(1), nil
		}})

	_, err := g.ExecContext(context.Background())
	me := mustMutationError(t, err)
	assert.Equal(t, "child", me.Node)
	assert.True(t, errors.Is(err, errFakeBoom), "original cause must be preserved")
	assert.True(t, errors.Is(err, ErrMutationRollback))
	assert.True(t, me.RollbackErr == rollbackErr)
}

func TestMutationGraphCallerTxSuccess(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	tx, err := db.Begin()
	require.NoError(t, err)

	g := NewMutationGraph().RunWithTx(tx).
		AddNode(MutationNode{Name: "a", Factory: func() (Sqlizer, error) { return Insert("t").Values(1), nil }})

	result, err := g.ExecContext(context.Background())
	require.NoError(t, err)
	require.NotNil(t, result)

	_, commits, rollbacks, _ := rec.counters()
	assert.Equal(t, 0, commits, "caller-owned tx must not be committed by the graph")
	assert.Equal(t, 0, rollbacks)

	require.NoError(t, tx.Commit())
	_, commits, _, _ = rec.counters()
	assert.Equal(t, 1, commits)
}

func TestMutationGraphCallerTxFailure(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	childSQL := expectedSQL(t, Insert("child").Values(1))
	rec.execErr[childSQL] = errFakeBoom

	tx, err := db.Begin()
	require.NoError(t, err)

	g := NewMutationGraph().RunWithTx(tx).
		AddNode(MutationNode{Name: "parent", Factory: func() (Sqlizer, error) { return Insert("parent").Values(1), nil }}).
		AddNode(MutationNode{Name: "child", DependsOn: []string{"parent"}, Factory: func() (Sqlizer, error) {
			return Insert("child").Values(1), nil
		}})

	_, err = g.ExecContext(context.Background())
	me := mustMutationError(t, err)
	assert.Equal(t, "child", me.Node)

	_, commits, rollbacks, _ := rec.counters()
	assert.Equal(t, 0, commits)
	assert.Equal(t, 0, rollbacks, "caller-owned tx must not be rolled back by the graph")

	require.NoError(t, tx.Rollback())
	_, _, rollbacks, _ = rec.counters()
	assert.Equal(t, 1, rollbacks)
}

func TestMutationGraphContextPreCanceled(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{Name: "a", Factory: func() (Sqlizer, error) { return Insert("t").Values(1), nil }})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := g.ExecContext(ctx)
	me := mustMutationError(t, err)
	// The transaction can never begin, so the failure is attributed to begin,
	// not to a node that never ran.
	assert.Equal(t, "", me.Node)
	assert.Equal(t, mutationOpBegin, me.Op)
	assert.True(t, errors.Is(err, context.Canceled))

	assert.Empty(t, rec.snapshotCalls())
	begins, _, rollbacks, _ := rec.counters()
	assert.Equal(t, 0, begins)
	assert.Equal(t, 0, rollbacks)
}

func TestMutationGraphContextCancelMidQuery(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	parentSQL := expectedSQL(t, Insert("parent").Values(1))
	rec.block[parentSQL] = make(chan struct{}) // never released

	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{Name: "parent", Factory: func() (Sqlizer, error) { return Insert("parent").Values(1), nil }}).
		AddNode(MutationNode{Name: "child", DependsOn: []string{"parent"}, Factory: func() (Sqlizer, error) {
			return Insert("child").Values(1), nil
		}})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := g.ExecContext(ctx)
		done <- err
	}()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && len(rec.snapshotCalls()) == 0 {
		time.Sleep(2 * time.Millisecond)
	}
	require.NotEmpty(t, rec.snapshotCalls(), "graph never reached the blocked node")

	cancel()
	select {
	case err := <-done:
		me := mustMutationError(t, err)
		assert.Equal(t, "parent", me.Node)
		assert.True(t, errors.Is(err, context.Canceled))
	case <-time.After(2 * time.Second):
		t.Fatal("graph did not stop after context cancellation")
	}

	calls := rec.snapshotCalls()
	assert.Len(t, calls, 1, "successor must never run after cancellation")
	_, commits, rollbacks, _ := rec.counters()
	assert.Equal(t, 1, rollbacks)
	assert.Equal(t, 0, commits)

	// The transaction connection must be released back to the pool.
	assert.Equal(t, 0, db.Stats().InUse)
}

func TestMutationGraphContextCancelBetweenNodes(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	childSQL := expectedSQL(t, Insert("child").Values(1))

	ctx, cancel := context.WithCancel(context.Background())
	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{
			Name: "parent",
			After: func(context.Context, string) error {
				cancel()
				return nil
			},
			Factory: func() (Sqlizer, error) { return Insert("parent").Values(1), nil },
		}).
		AddNode(MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory:   func() (Sqlizer, error) { return Insert("child").Values(1), nil },
		})

	_, err := g.ExecContext(ctx)
	me := mustMutationError(t, err)
	assert.Equal(t, "child", me.Node)
	assert.Equal(t, mutationOpContext, me.Op)
	assert.True(t, errors.Is(err, context.Canceled))
	assert.Empty(t, callsFor(rec.snapshotCalls(), childSQL))
	_, _, rollbacks, _ := rec.counters()
	assert.Equal(t, 1, rollbacks)
	assert.Equal(t, 0, db.Stats().InUse)
}

func TestMutationGraphRowsClosedBeforeNextNode(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	first := Insert("a").Values(1).Suffix("RETURNING id")
	second := Insert("b").Values(2).Suffix("RETURNING id")
	firstSQL := expectedSQL(t, first)
	secondSQL := expectedSQL(t, second)
	rec.returningCols[firstSQL] = []string{"id"}
	rec.returningRows[firstSQL] = [][]driver.Value{{int64(1)}}
	rec.returningCols[secondSQL] = []string{"id"}
	rec.returningRows[secondSQL] = [][]driver.Value{{int64(2)}}

	g := NewMutationGraph().
		Capabilities(MutationCapabilities{Returning: true}).
		RunWith(db).
		AddNode(MutationNode{Name: "a", DependsOn: []string(nil), Factory: func() (Sqlizer, error) { return first, nil }, Outputs: MutationOutputSpec{Returning: []string{"id"}}}).
		AddNode(MutationNode{Name: "b", DependsOn: []string{"a"}, Factory: func() (Sqlizer, error) { return second, nil }, Outputs: MutationOutputSpec{Returning: []string{"id"}}})

	_, err := g.ExecContext(context.Background())
	require.NoError(t, err)

	events := rec.snapshotEvents()
	require.Equal(t, []fakeEvent{
		{kind: "query", query: firstSQL},
		{kind: "close", query: firstSQL},
		{kind: "query", query: secondSQL},
		{kind: "close", query: secondSQL},
	}, events)
	assert.Equal(t, 0, db.Stats().InUse)
}

func TestMutationGraphLastInsertIdRuntimeUnsupported(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	rec.lastInsertIDSupported = false
	childSQL := expectedSQL(t, Insert("child").Values(1))

	g := NewMutationGraph().
		Capabilities(MutationCapabilities{LastInsertId: true}).
		RunWith(db).
		AddNode(MutationNode{Name: "parent", Factory: parentInsertFactory(), Outputs: MutationOutputSpec{LastInsertId: true}}).
		AddNode(MutationNode{Name: "child", DependsOn: []string{"parent"}, Factory: childInsertFactory()})

	_, err := g.ExecContext(context.Background())
	me := mustMutationError(t, err)
	assert.Equal(t, "parent", me.Node)
	assert.Equal(t, mutationOpLastInsertID, me.Op)
	assert.True(t, errors.Is(err, ErrMutationLastInsertIdUnsupported))

	assert.Empty(t, callsFor(rec.snapshotCalls(), childSQL), "must not continue with a fabricated zero id")
	_, _, rollbacks, _ := rec.counters()
	assert.Equal(t, 1, rollbacks)
}

func TestMutationGraphReturningRowShapeAndNull(t *testing.T) {
	cases := []struct {
		name    string
		rows    [][]driver.Value
		wantErr error
		wantOp  string
		nullRef bool
	}{
		{name: "no row", rows: nil, wantErr: ErrMutationNoRows, wantOp: mutationOpScan},
		{name: "two rows", rows: [][]driver.Value{{int64(1)}, {int64(2)}}, wantErr: ErrMutationMultipleRows, wantOp: mutationOpScan},
		{name: "null referenced", rows: [][]driver.Value{{nil}}, wantErr: ErrMutationNullOutput, wantOp: mutationOpReference, nullRef: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := newFakeRecorder()
			db := openFakeDB(t, rec)
			parent := Insert("parent").Values(1).Suffix("RETURNING id")
			parentSQL := expectedSQL(t, parent)
			rec.returningCols[parentSQL] = []string{"id"}
			rec.returningRows[parentSQL] = tc.rows

			g := NewMutationGraph().
				Capabilities(MutationCapabilities{Returning: true}).
				RunWith(db).
				AddNode(MutationNode{
					Name:    "parent",
					Factory: func() (Sqlizer, error) { return parent, nil },
					Outputs: MutationOutputSpec{Returning: []string{"id"}},
				})

			if tc.nullRef {
				g.AddNode(MutationNode{
					Name:      "child",
					DependsOn: []string{"parent"},
					Factory: func() (Sqlizer, error) {
						return Insert("child").Values(ReturningRef("parent", "id")), nil
					},
				})
			}

			_, err := g.ExecContext(context.Background())
			me := mustMutationError(t, err)
			assert.True(t, errors.Is(err, tc.wantErr), "want %v, got %v", tc.wantErr, err)
			assert.Equal(t, tc.wantOp, me.Op)
			_, _, rollbacks, _ := rec.counters()
			assert.Equal(t, 1, rollbacks)
		})
	}
}

func TestMutationGraphReturningMissingColumn(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	parent := Insert("parent").Values(1).Suffix("RETURNING id")
	parentSQL := expectedSQL(t, parent)
	rec.returningCols[parentSQL] = []string{"other"}
	rec.returningRows[parentSQL] = [][]driver.Value{{int64(1)}}

	g := NewMutationGraph().
		Capabilities(MutationCapabilities{Returning: true}).
		RunWith(db).
		AddNode(MutationNode{
			Name:    "parent",
			Factory: func() (Sqlizer, error) { return parent, nil },
			Outputs: MutationOutputSpec{Returning: []string{"id"}},
		})

	_, err := g.ExecContext(context.Background())
	me := mustMutationError(t, err)
	assert.Equal(t, mutationOpScan, me.Op)
}

func TestMutationGraphHooks(t *testing.T) {
	t.Run("before hook failure skips statement", func(t *testing.T) {
		rec := newFakeRecorder()
		db := openFakeDB(t, rec)
		parentSQL := expectedSQL(t, Insert("parent").Values(1))
		childSQL := expectedSQL(t, Insert("child").Values(1))

		g := NewMutationGraph().RunWith(db).
			AddNode(MutationNode{Name: "parent", Factory: func() (Sqlizer, error) { return Insert("parent").Values(1), nil }}).
			AddNode(MutationNode{
				Name:      "child",
				DependsOn: []string{"parent"},
				Before:    func(context.Context, string) error { return errFakeBoom },
				Factory:   func() (Sqlizer, error) { return Insert("child").Values(1), nil },
			})

		_, err := g.ExecContext(context.Background())
		me := mustMutationError(t, err)
		assert.Equal(t, "child", me.Node)
		assert.Equal(t, mutationOpHookBefore, me.Op)
		assert.NotEmpty(t, callsFor(rec.snapshotCalls(), parentSQL))
		assert.Empty(t, callsFor(rec.snapshotCalls(), childSQL))
		_, _, rollbacks, _ := rec.counters()
		assert.Equal(t, 1, rollbacks)
	})

	t.Run("after hook failure keeps node identity", func(t *testing.T) {
		rec := newFakeRecorder()
		db := openFakeDB(t, rec)
		childSQL := expectedSQL(t, Insert("child").Values(1))

		g := NewMutationGraph().RunWith(db).
			AddNode(MutationNode{Name: "parent", Factory: func() (Sqlizer, error) { return Insert("parent").Values(1), nil }}).
			AddNode(MutationNode{
				Name:      "child",
				DependsOn: []string{"parent"},
				After:     func(context.Context, string) error { return errFakeBoom },
				Factory:   func() (Sqlizer, error) { return Insert("child").Values(1), nil },
			})

		_, err := g.ExecContext(context.Background())
		me := mustMutationError(t, err)
		assert.Equal(t, "child", me.Node)
		assert.Equal(t, mutationOpHookAfter, me.Op)
		assert.NotEmpty(t, callsFor(rec.snapshotCalls(), childSQL), "statement already ran; hook failure does not erase it")
		_, _, rollbacks, _ := rec.counters()
		assert.Equal(t, 1, rollbacks)
	})

	t.Run("hooks receive node name but no args", func(t *testing.T) {
		rec := newFakeRecorder()
		db := openFakeDB(t, rec)
		var mu sync.Mutex
		var seen []string
		hook := func(_ context.Context, name string) error {
			mu.Lock()
			seen = append(seen, name)
			mu.Unlock()
			return nil
		}
		g := NewMutationGraph().RunWith(db).
			AddNode(MutationNode{Name: "a", Before: hook, After: hook, Factory: func() (Sqlizer, error) {
				return Insert("t").Values(1), nil
			}})
		_, err := g.ExecContext(context.Background())
		require.NoError(t, err)
		assert.Equal(t, []string{"a", "a"}, seen)
	})
}

func TestMutationGraphSensitiveArgsNotInError(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	secret := "super-secret-value-9f3k2m"
	childSQL := expectedSQL(t, Insert("child").Values(1))
	rec.execErr[childSQL] = errFakeBoom

	g := NewMutationGraph().RunWith(db).
		AddNode(MutationNode{
			Name:    "parent",
			Factory: func() (Sqlizer, error) { return Insert("parent").Values(secret), nil },
		}).
		AddNode(MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory:   func() (Sqlizer, error) { return Insert("child").Values(1), nil },
		})

	_, err := g.ExecContext(context.Background())
	require.Error(t, err)
	assert.NotContains(t, err.Error(), secret)
}

func TestMutationGraphFactoryIndependenceAcrossExecs(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)
	parentSQL := expectedSQL(t, Insert("parent").Columns("name").Values("alice"))
	childSQL := expectedSQL(t, Insert("child").Columns("parent_id", "note").Values(LastInsertIdRef("parent"), "toy"))
	rec.idFor[parentSQL] = 42

	var parentCalls, childCalls int
	var mu sync.Mutex
	g := NewMutationGraph().
		Capabilities(MutationCapabilities{LastInsertId: true}).
		RunWith(db).
		AddNode(MutationNode{
			Name: "parent",
			Factory: func() (Sqlizer, error) {
				mu.Lock()
				parentCalls++
				mu.Unlock()
				return Insert("parent").Columns("name").Values("alice"), nil
			},
			Outputs: MutationOutputSpec{LastInsertId: true},
		}).
		AddNode(MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory: func() (Sqlizer, error) {
				mu.Lock()
				childCalls++
				mu.Unlock()
				return Insert("child").Columns("parent_id", "note").Values(LastInsertIdRef("parent"), "toy"), nil
			},
		})

	for i := 0; i < 2; i++ {
		_, err := g.ExecContext(context.Background())
		require.NoError(t, err)
	}

	mu.Lock()
	assert.Equal(t, 2, parentCalls)
	assert.Equal(t, 2, childCalls)
	mu.Unlock()

	calls := rec.snapshotCalls()
	assert.Len(t, callsFor(calls, parentSQL), 2)
	assert.Len(t, callsFor(calls, childSQL), 2)
	for _, c := range callsFor(calls, childSQL) {
		assert.Equal(t, []interface{}{int64(42), "toy"}, c.args)
	}
	_, commits, _, _ := rec.counters()
	assert.Equal(t, 2, commits)
}

func TestMutationGraphLegacyRunWithCompatibility(t *testing.T) {
	// The existing builder RunWith path must keep its exact behavior.
	stub := &DBStub{}
	b := Insert("test").Values(1).RunWith(stub)
	_, err := b.Exec()
	require.NoError(t, err)
	assert.Equal(t, "INSERT INTO test VALUES (?)", stub.LastExecSql)
	assert.Equal(t, []interface{}{1}, stub.LastExecArgs)
}

func TestMutationGraphAddNodeCopiesDeclarations(t *testing.T) {
	rec := newFakeRecorder()
	db := openFakeDB(t, rec)

	deps := []string{"a"}
	returning := []string{"id"}
	g := NewMutationGraph().
		Capabilities(MutationCapabilities{Returning: true}).
		RunWith(db).
		AddNode(MutationNode{Name: "a", Factory: func() (Sqlizer, error) { return Insert("t").Values(1).Suffix("RETURNING id"), nil }, Outputs: MutationOutputSpec{Returning: returning}}).
		AddNode(MutationNode{Name: "b", DependsOn: deps, Factory: func() (Sqlizer, error) { return Insert("u").Values(ReturningRef("a", "id")), nil }})

	// Mutate the caller-side slices after registration; the graph must keep
	// its own copy.
	deps[0] = "tampered"
	returning[0] = "tampered"

	rec.returningCols["INSERT INTO t VALUES (?) RETURNING id"] = []string{"id"}
	rec.returningRows["INSERT INTO t VALUES (?) RETURNING id"] = [][]driver.Value{{int64(5)}}

	_, err := g.ExecContext(context.Background())
	require.NoError(t, err)
}

func TestMutationGraphErrorStringFormat(t *testing.T) {
	me := &MutationError{Node: "child", Op: mutationOpExec, Err: errFakeBoom}
	s := me.Error()
	assert.Contains(t, s, `"child"`)
	assert.Contains(t, s, mutationOpExec)
	assert.Contains(t, s, errFakeBoom.Error())

	wrapped := &MutationError{Node: "child", Op: mutationOpExec, Err: errFakeBoom, RollbackErr: errors.New("rb")}
	assert.True(t, errors.Is(wrapped, ErrMutationRollback))
	assert.False(t, errors.Is(me, ErrMutationRollback))
}

func TestMutationGraphMaxNodesDefaultRestored(t *testing.T) {
	g := NewMutationGraph().MaxNodes(0)
	assert.Equal(t, DefaultMaxMutationNodes, g.maxNodes)
}
