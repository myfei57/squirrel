package squirrel

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"sync"
	"testing"
)

// fakeMutationDriverName is registered once for the whole test binary.
const fakeMutationDriverName = "squirrel_mutation_fake"

func init() {
	sql.Register(fakeMutationDriverName, &fakeDriver{})
}

// fakeCall records one Exec/Query with a copy of its bound arguments.
type fakeCall struct {
	kind  string
	query string
	args  []interface{}
}

// fakeEvent records query start and rows close ordering.
type fakeEvent struct {
	kind  string
	query string
}

// fakeRecorder is the shared, mutex-protected observation surface of one fake
// database connection tree.
type fakeRecorder struct {
	mu sync.Mutex

	calls  []fakeCall
	events []fakeEvent

	openConns int
	begins    int
	commits   int
	rollbacks int

	execErr  map[string]error
	queryErr map[string]error
	block    map[string]chan struct{}

	returningCols map[string][]string
	returningRows map[string][][]driver.Value

	lastInsertIDSupported bool
	lastInsertIDErr       error
	idFor                 map[string]int64
	rowsFor               map[string]int64

	commitErr   error
	rollbackErr error
}

func newFakeRecorder() *fakeRecorder {
	return &fakeRecorder{
		lastInsertIDSupported: true,
		idFor:                 map[string]int64{},
		rowsFor:               map[string]int64{},
		execErr:               map[string]error{},
		queryErr:              map[string]error{},
		block:                 map[string]chan struct{}{},
		returningCols:         map[string][]string{},
		returningRows:         map[string][][]driver.Value{},
	}
}

func (r *fakeRecorder) recordCall(kind, query string, args []driver.NamedValue) {
	bound := make([]interface{}, len(args))
	for i, arg := range args {
		bound[i] = arg.Value
	}
	r.mu.Lock()
	r.calls = append(r.calls, fakeCall{kind: kind, query: query, args: bound})
	r.mu.Unlock()
}

func (r *fakeRecorder) recordEvent(kind, query string) {
	r.mu.Lock()
	r.events = append(r.events, fakeEvent{kind: kind, query: query})
	r.mu.Unlock()
}

func (r *fakeRecorder) waitBlock(ctx context.Context, query string) error {
	r.mu.Lock()
	ch, ok := r.block[query]
	r.mu.Unlock()
	if !ok {
		return nil
	}
	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *fakeRecorder) errorFor(m map[string]error, query string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return m[query]
}

func (r *fakeRecorder) snapshotCalls() []fakeCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]fakeCall, len(r.calls))
	copy(out, r.calls)
	return out
}

func (r *fakeRecorder) snapshotEvents() []fakeEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]fakeEvent, len(r.events))
	copy(out, r.events)
	return out
}

func (r *fakeRecorder) counters() (begins, commits, rollbacks, openConns int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.begins, r.commits, r.rollbacks, r.openConns
}

// fakeDriver resolves DSN tokens to recorders so every test gets an isolated
// observation surface while sharing one registered driver name.
type fakeDriver struct{}

var fakeDriverRegistry = struct {
	sync.Mutex
	recs map[string]*fakeRecorder
}{recs: map[string]*fakeRecorder{}}

func (d *fakeDriver) Open(name string) (driver.Conn, error) {
	fakeDriverRegistry.Lock()
	rec, ok := fakeDriverRegistry.recs[name]
	fakeDriverRegistry.Unlock()
	if !ok {
		return nil, fmt.Errorf("fake driver: unknown data source %q", name)
	}
	rec.mu.Lock()
	rec.openConns++
	rec.mu.Unlock()
	return &fakeConn{rec: rec}, nil
}

func openFakeDB(t *testing.T, rec *fakeRecorder) *sql.DB {
	t.Helper()
	dsn := fmt.Sprintf("fake-%p", rec)
	fakeDriverRegistry.Lock()
	fakeDriverRegistry.recs[dsn] = rec
	fakeDriverRegistry.Unlock()
	db, err := sql.Open(fakeMutationDriverName, dsn)
	if err != nil {
		t.Fatalf("open fake db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

type fakeConn struct {
	rec *fakeRecorder
}

var (
	_ driver.Conn           = (*fakeConn)(nil)
	_ driver.ExecerContext  = (*fakeConn)(nil)
	_ driver.QueryerContext = (*fakeConn)(nil)
	_ driver.ConnBeginTx    = (*fakeConn)(nil)
)

func (c *fakeConn) Prepare(query string) (driver.Stmt, error) {
	return &fakeStmt{rec: c.rec, query: query}, nil
}

func (c *fakeConn) Close() error {
	c.rec.mu.Lock()
	c.rec.openConns--
	c.rec.mu.Unlock()
	return nil
}

func (c *fakeConn) Begin() (driver.Tx, error) {
	return c.beginTx()
}

func (c *fakeConn) BeginTx(_ context.Context, _ driver.TxOptions) (driver.Tx, error) {
	return c.beginTx()
}

func (c *fakeConn) beginTx() (driver.Tx, error) {
	c.rec.mu.Lock()
	c.rec.begins++
	c.rec.mu.Unlock()
	return &fakeTx{rec: c.rec}, nil
}

func (c *fakeConn) CheckNamedValue(_ *driver.NamedValue) error { return nil }

func (c *fakeConn) ExecContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	c.rec.recordCall("exec", query, args)
	if err := c.rec.waitBlock(ctx, query); err != nil {
		return nil, err
	}
	if err := c.rec.errorFor(c.rec.execErr, query); err != nil {
		return nil, err
	}
	return &fakeResult{rec: c.rec, query: query}, nil
}

func (c *fakeConn) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	c.rec.recordCall("query", query, args)
	c.rec.recordEvent("query", query)
	if err := c.rec.waitBlock(ctx, query); err != nil {
		return nil, err
	}
	if err := c.rec.errorFor(c.rec.queryErr, query); err != nil {
		return nil, err
	}
	c.rec.mu.Lock()
	cols := append([]string(nil), c.rec.returningCols[query]...)
	data := c.rec.returningRows[query]
	c.rec.mu.Unlock()
	return &fakeRows{rec: c.rec, query: query, cols: cols, data: data, idx: -1}, nil
}

type fakeStmt struct {
	rec   *fakeRecorder
	query string
}

var (
	_ driver.Stmt = (*fakeStmt)(nil)
)

func (s *fakeStmt) Close() error  { return nil }
func (s *fakeStmt) NumInput() int { return -1 }
func (s *fakeStmt) Exec(_ []driver.Value) (driver.Result, error) {
	return nil, errors.New("fakeStmt.Exec must not be used by the graph")
}
func (s *fakeStmt) Query(_ []driver.Value) (driver.Rows, error) {
	return nil, errors.New("fakeStmt.Query must not be used by the graph")
}

type fakeTx struct {
	rec *fakeRecorder
}

var _ driver.Tx = (*fakeTx)(nil)

func (t *fakeTx) Commit() error {
	t.rec.mu.Lock()
	t.rec.commits++
	err := t.rec.commitErr
	t.rec.mu.Unlock()
	return err
}

func (t *fakeTx) Rollback() error {
	t.rec.mu.Lock()
	t.rec.rollbacks++
	err := t.rec.rollbackErr
	t.rec.mu.Unlock()
	return err
}

type fakeResult struct {
	rec   *fakeRecorder
	query string
}

var _ driver.Result = (*fakeResult)(nil)

func (r *fakeResult) LastInsertId() (int64, error) {
	if r.rec.lastInsertIDErr != nil {
		return 0, r.rec.lastInsertIDErr
	}
	if !r.rec.lastInsertIDSupported {
		return 0, errors.New("driver does not support LastInsertId")
	}
	r.rec.mu.Lock()
	defer r.rec.mu.Unlock()
	if id, ok := r.rec.idFor[r.query]; ok {
		return id, nil
	}
	return 1, nil
}

func (r *fakeResult) RowsAffected() (int64, error) {
	r.rec.mu.Lock()
	defer r.rec.mu.Unlock()
	if n, ok := r.rec.rowsFor[r.query]; ok {
		return n, nil
	}
	return 1, nil
}

type fakeRows struct {
	rec    *fakeRecorder
	query  string
	cols   []string
	data   [][]driver.Value
	idx    int
	closed bool
}

var _ driver.Rows = (*fakeRows)(nil)

func (r *fakeRows) Columns() []string { return r.cols }

func (r *fakeRows) Close() error {
	r.rec.mu.Lock()
	r.closed = true
	r.rec.events = append(r.rec.events, fakeEvent{kind: "close", query: r.query})
	r.rec.mu.Unlock()
	return nil
}

func (r *fakeRows) Next(dest []driver.Value) error {
	r.idx++
	if r.idx >= len(r.data) {
		return io.EOF
	}
	copy(dest, r.data[r.idx])
	return nil
}
