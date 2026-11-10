// Package graphintegration exercises MutationGraph against a real, pure-Go
// SQLite database (modernc.org/sqlite), covering foreign-key parent/child
// chains, RETURNING semantics, atomicity and transaction ownership.
package graphintegration

import (
	"context"
	"database/sql"
	"errors"
	"testing"
	"time"

	sqrl "github.com/Masterminds/squirrel"
	_ "modernc.org/sqlite"
)

const schema = `
CREATE TABLE parent (
	id   INTEGER PRIMARY KEY AUTOINCREMENT,
	name TEXT NOT NULL,
	tag  TEXT
);
CREATE TABLE child (
	id        INTEGER PRIMARY KEY AUTOINCREMENT,
	parent_id INTEGER NOT NULL REFERENCES parent(id),
	note      TEXT
);`

func openGraphDB(t *testing.T) *sql.DB {
	t.Helper()
	// _pragma turns foreign keys on for every connection; the in-memory DSN
	// keeps each test isolated. MaxOpenConns(1) pins the in-memory database to
	// one connection, which also matches the single-connection transaction
	// contract of MutationGraph.
	db, err := sql.Open("sqlite", "file::memory:?_pragma=foreign_keys(1)")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("create schema: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func countRows(t *testing.T, db *sql.DB, table string) int {
	t.Helper()
	var n int
	if err := db.QueryRow("SELECT COUNT(*) FROM " + table).Scan(&n); err != nil {
		t.Fatalf("count %s: %v", table, err)
	}
	return n
}

func mustGraphError(t *testing.T, err error) *sqrl.MutationError {
	t.Helper()
	if err == nil {
		t.Fatal("expected MutationError, got nil")
	}
	me := &sqrl.MutationError{}
	if !errors.As(err, &me) {
		t.Fatalf("expected *MutationError, got %T: %v", err, err)
	}
	return me
}

// TestSQLiteParentChildLastInsertId runs the canonical parent-then-child chain
// through a real foreign-key relationship and verifies the bound identifier.
func TestSQLiteParentChildLastInsertId(t *testing.T) {
	db := openGraphDB(t)

	// The generated SQL keeps placeholders and the ref travels as an argument.
	childBuilder := sqrl.Insert("child").Columns("parent_id", "note").
		Values(sqrl.LastInsertIdRef("parent"), "toy")
	query, args, err := childBuilder.ToSql()
	if err != nil {
		t.Fatal(err)
	}
	if query != "INSERT INTO child (parent_id,note) VALUES (?,?)" {
		t.Fatalf("unexpected SQL: %q", query)
	}
	if _, ok := args[0].(sqrl.MutationRef); !ok {
		t.Fatalf("reference must stay a bound argument, got %T", args[0])
	}

	g := sqrl.NewMutationGraph().
		Capabilities(sqrl.MutationCapabilities{LastInsertId: true}).
		RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "parent",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("parent").Columns("name").Values("alice"), nil
			},
			Outputs: sqrl.MutationOutputSpec{LastInsertId: true},
		}).
		AddNode(sqrl.MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("child").Columns("parent_id", "note").
					Values(sqrl.LastInsertIdRef("parent"), "toy"), nil
			},
		})

	result, err := g.ExecContext(context.Background())
	if err != nil {
		t.Fatalf("graph exec: %v", err)
	}
	parentOut, ok := result.Node("parent")
	if !ok {
		t.Fatal("missing parent output")
	}
	parentID, ok := parentOut.LastInsertId()
	if !ok {
		t.Fatal("missing verified LastInsertId")
	}

	var childParentID int64
	if err := db.QueryRow(
		"SELECT parent_id FROM child WHERE note = ?", "toy",
	).Scan(&childParentID); err != nil {
		t.Fatalf("read child: %v", err)
	}
	if childParentID != parentID {
		t.Fatalf("child parent_id = %d, want %d", childParentID, parentID)
	}
	if countRows(t, db, "parent") != 1 || countRows(t, db, "child") != 1 {
		t.Fatal("expected exactly one parent and one child row")
	}
}

// TestSQLiteParentChildReturning drives the same chain through the RETURNING
// capability instead of LastInsertId.
func TestSQLiteParentChildReturning(t *testing.T) {
	db := openGraphDB(t)

	g := sqrl.NewMutationGraph().
		Capabilities(sqrl.MutationCapabilities{Returning: true}).
		RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "parent",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("parent").Columns("name").Values("bea").
					Suffix("RETURNING id"), nil
			},
			Outputs: sqrl.MutationOutputSpec{Returning: []string{"id"}},
		}).
		AddNode(sqrl.MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("child").Columns("parent_id", "note").
					Values(sqrl.ReturningRef("parent", "id"), "block"), nil
			},
		})

	result, err := g.ExecContext(context.Background())
	if err != nil {
		t.Fatalf("graph exec: %v", err)
	}
	parentOut, ok := result.Node("parent")
	if !ok {
		t.Fatal("missing parent output")
	}
	returnedID, ok := parentOut.Returning("id")
	if !ok {
		t.Fatal("missing RETURNING output")
	}

	var childParentID int64
	if err := db.QueryRow(
		"SELECT parent_id FROM child WHERE note = ?", "block",
	).Scan(&childParentID); err != nil {
		t.Fatalf("read child: %v", err)
	}
	if childParentID != returnedID.(int64) {
		t.Fatalf("child parent_id = %d, want RETURNING id %v", childParentID, returnedID)
	}
}

// TestSQLiteForeignKeyAtomicity proves a failing successor rolls the whole
// owned transaction back, including the already-inserted predecessor.
func TestSQLiteForeignKeyAtomicity(t *testing.T) {
	db := openGraphDB(t)

	g := sqrl.NewMutationGraph().
		RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "parent",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("parent").Columns("name").Values("alice"), nil
			},
		}).
		AddNode(sqrl.MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory: func() (sqrl.Sqlizer, error) {
				// 9999 does not exist: the foreign key must reject this row.
				return sqrl.Insert("child").Columns("parent_id", "note").
					Values(9999, "illegal"), nil
			},
		})

	_, err := g.ExecContext(context.Background())
	me := mustGraphError(t, err)
	if me.Node != "child" {
		t.Fatalf("error node = %q, want child", me.Node)
	}

	if countRows(t, db, "parent") != 0 {
		t.Fatal("parent row survived although the transaction must roll back")
	}
	if countRows(t, db, "child") != 0 {
		t.Fatal("child row must not exist")
	}
}

// TestSQLiteIndependentNodes commits unrelated nodes together in one graph.
func TestSQLiteIndependentNodes(t *testing.T) {
	db := openGraphDB(t)

	g := sqrl.NewMutationGraph().RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "p1",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("parent").Columns("name").Values("one"), nil
			},
		}).
		AddNode(sqrl.MutationNode{
			Name: "p2",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("parent").Columns("name").Values("two"), nil
			},
		})

	if _, err := g.ExecContext(context.Background()); err != nil {
		t.Fatalf("graph exec: %v", err)
	}
	if countRows(t, db, "parent") != 2 {
		t.Fatal("both independent nodes must commit")
	}
}

// TestSQLiteCallerOwnedTransaction checks that RunWithTx never commits or rolls
// back a transaction owned by the caller.
func TestSQLiteCallerOwnedTransaction(t *testing.T) {
	db := openGraphDB(t)

	t.Run("success needs caller commit", func(t *testing.T) {
		tx, err := db.Begin()
		if err != nil {
			t.Fatal(err)
		}
		g := sqrl.NewMutationGraph().RunWithTx(tx).
			AddNode(sqrl.MutationNode{
				Name: "parent",
				Factory: func() (sqrl.Sqlizer, error) {
					return sqrl.Insert("parent").Columns("name").Values("tx"), nil
				},
			})
		if _, err := g.ExecContext(context.Background()); err != nil {
			t.Fatalf("graph exec: %v", err)
		}
		// Visible inside the caller transaction, not committed yet.
		var inside int
		if err := tx.QueryRow("SELECT COUNT(*) FROM parent").Scan(&inside); err != nil {
			t.Fatal(err)
		}
		if inside != 1 {
			t.Fatalf("inside-tx count = %d, want 1", inside)
		}
		if err := tx.Commit(); err != nil {
			t.Fatal(err)
		}
		if countRows(t, db, "parent") != 1 {
			t.Fatal("caller commit must publish the row")
		}
	})

	t.Run("failure leaves rollback to caller", func(t *testing.T) {
		tx, err := db.Begin()
		if err != nil {
			t.Fatal(err)
		}
		g := sqrl.NewMutationGraph().RunWithTx(tx).
			AddNode(sqrl.MutationNode{
				Name: "parent",
				Factory: func() (sqrl.Sqlizer, error) {
					return sqrl.Insert("parent").Columns("name").Values("kept"), nil
				},
			}).
			AddNode(sqrl.MutationNode{
				Name:      "child",
				DependsOn: []string{"parent"},
				Factory: func() (sqrl.Sqlizer, error) {
					return sqrl.Insert("child").Columns("parent_id", "note").
						Values(9999, "illegal"), nil
				},
			})
		if _, err := g.ExecContext(context.Background()); err == nil {
			t.Fatal("expected FK failure")
		}
		if err := tx.Rollback(); err != nil {
			t.Fatal(err)
		}
		if countRows(t, db, "parent") != 1 { // only the row from the previous subtest
			t.Fatal("caller rollback must undo graph writes")
		}
	})
}

// TestSQLiteContextCancelReleasesConnection cancels a blocked graph run and
// verifies no connection stays in use and nothing is persisted.
func TestSQLiteContextCancelReleasesConnection(t *testing.T) {
	db := openGraphDB(t)

	entered := make(chan struct{})
	release := make(chan struct{})
	ctx, cancel := context.WithCancel(context.Background())

	g := sqrl.NewMutationGraph().RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "parent",
			Before: func(context.Context, string) error {
				close(entered)
				select {
				case <-release:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			},
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("parent").Columns("name").Values("canceled"), nil
			},
		})

	done := make(chan error, 1)
	go func() {
		_, err := g.ExecContext(ctx)
		done <- err
	}()

	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("hook never entered")
	}
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("graph did not stop after cancellation")
	}
	close(release)

	// The SQLite driver may finalize its cancellation-triggered rollback
	// asynchronously, so poll for the connection to be released instead of
	// reading Stats exactly once.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && db.Stats().InUse != 0 {
		time.Sleep(5 * time.Millisecond)
	}
	if stats := db.Stats(); stats.InUse != 0 {
		t.Fatalf("connection still in use after cancellation: %+v", stats)
	}
	if countRows(t, db, "parent") != 0 {
		t.Fatal("canceled graph must not leave rows behind")
	}
}

// TestSQLiteReturningMultipleRowsRollsBack ensures the single-row RETURNING
// contract is enforced against a real database and the update is rolled back.
func TestSQLiteReturningMultipleRowsRollsBack(t *testing.T) {
	db := openGraphDB(t)
	for _, name := range []string{"a", "b"} {
		if _, err := db.Exec("INSERT INTO parent(name) VALUES (?)", name); err != nil {
			t.Fatal(err)
		}
	}

	g := sqrl.NewMutationGraph().
		Capabilities(sqrl.MutationCapabilities{Returning: true}).
		RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "rename",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Update("parent").Set("name", "same").
					Suffix("RETURNING id"), nil
			},
			Outputs: sqrl.MutationOutputSpec{Returning: []string{"id"}},
		})

	_, err := g.ExecContext(context.Background())
	if !errors.Is(err, sqrl.ErrMutationMultipleRows) {
		t.Fatalf("want ErrMutationMultipleRows, got %v", err)
	}

	var unchanged int
	if err := db.QueryRow(
		"SELECT COUNT(*) FROM parent WHERE name IN (?, ?)", "a", "b",
	).Scan(&unchanged); err != nil {
		t.Fatal(err)
	}
	if unchanged != 2 {
		t.Fatal("multi-row RETURNING failure must roll the update back")
	}
}

// TestSQLiteNullReturningFails ensures a NULL referenced output is reported
// instead of silently binding an empty value.
func TestSQLiteNullReturningFails(t *testing.T) {
	db := openGraphDB(t)

	g := sqrl.NewMutationGraph().
		Capabilities(sqrl.MutationCapabilities{Returning: true}).
		RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "parent",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("parent").Columns("name", "tag").
					Values("nullable", nil).Suffix("RETURNING id, tag"), nil
			},
			Outputs: sqrl.MutationOutputSpec{Returning: []string{"id", "tag"}},
		}).
		AddNode(sqrl.MutationNode{
			Name:      "child",
			DependsOn: []string{"parent"},
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Insert("child").Columns("parent_id", "note").
					Values(sqrl.ReturningRef("parent", "tag"), "note"), nil
			},
		})

	_, err := g.ExecContext(context.Background())
	if !errors.Is(err, sqrl.ErrMutationNullOutput) {
		t.Fatalf("want ErrMutationNullOutput, got %v", err)
	}
	if countRows(t, db, "child") != 0 {
		t.Fatal("child must not be written with a fabricated NULL replacement")
	}
	if countRows(t, db, "parent") != 0 {
		t.Fatal("owned transaction must roll the parent insert back too")
	}
}

// TestSQLiteRowsAffected verifies the verified RowsAffected output against real
// row counts.
func TestSQLiteRowsAffected(t *testing.T) {
	db := openGraphDB(t)
	res, err := db.Exec("INSERT INTO parent(name) VALUES (?)", "root")
	if err != nil {
		t.Fatal(err)
	}
	parentID, _ := res.LastInsertId()
	for _, note := range []string{"x", "y"} {
		if _, err := db.Exec(
			"INSERT INTO child(parent_id, note) VALUES (?, ?)", parentID, note,
		); err != nil {
			t.Fatal(err)
		}
	}

	g := sqrl.NewMutationGraph().RunWith(db).
		AddNode(sqrl.MutationNode{
			Name: "touch",
			Factory: func() (sqrl.Sqlizer, error) {
				return sqrl.Update("child").Set("note", "z").
					Where("parent_id = ?", parentID), nil
			},
			Outputs: sqrl.MutationOutputSpec{RowsAffected: true},
		})

	result, err := g.ExecContext(context.Background())
	if err != nil {
		t.Fatalf("graph exec: %v", err)
	}
	touchOut, ok := result.Node("touch")
	if !ok {
		t.Fatal("missing touch output")
	}
	affected, ok := touchOut.RowsAffected()
	if !ok || affected != 2 {
		t.Fatalf("RowsAffected = %d (ok=%v), want 2", affected, ok)
	}
}
