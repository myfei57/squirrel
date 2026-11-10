//go:build go1.8
// +build go1.8

package squirrel

import (
	"context"
	"database/sql"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// --- fakes -----------------------------------------------------------------

var (
	errPrimaryMarker = errors.New("primary runner marker")
	errReplicaMarker = errors.New("replica runner marker")
	errBoom          = errors.New("boom")
)

type fakeCall struct {
	ctx   context.Context
	query string
	args  []interface{}
}

// fakeRunner is a controlled primary/replica runner. Query calls return a
// distinctive marker error so tests can prove which runner was selected
// without constructing real *sql.Rows values.
type fakeRunner struct {
	name     string
	queryErr error
	execErr  error
	scanner  RowScanner

	mu        sync.Mutex
	execs     []fakeCall
	queries   []fakeCall
	queryRows []fakeCall

	committed  bool
	rolledBack bool
}

func newFakeRunner(name string) *fakeRunner {
	f := &fakeRunner{name: name}
	switch name {
	case "primary", "tx":
		f.queryErr = errPrimaryMarker
	default:
		f.queryErr = errReplicaMarker
	}
	f.scanner = &markerScanner{origin: name}
	return f
}

func (f *fakeRunner) Exec(query string, args ...interface{}) (sql.Result, error) {
	return f.ExecContext(context.Background(), query, args...)
}

func (f *fakeRunner) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.execs = append(f.execs, fakeCall{ctx: ctx, query: query, args: args})
	return nil, f.execErr
}

func (f *fakeRunner) Query(query string, args ...interface{}) (*sql.Rows, error) {
	return f.QueryContext(context.Background(), query, args...)
}

func (f *fakeRunner) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queries = append(f.queries, fakeCall{ctx: ctx, query: query, args: args})
	return nil, f.queryErr
}

func (f *fakeRunner) QueryRow(query string, args ...interface{}) RowScanner {
	return f.QueryRowContext(context.Background(), query, args...)
}

func (f *fakeRunner) QueryRowContext(ctx context.Context, query string, args ...interface{}) RowScanner {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.queryRows = append(f.queryRows, fakeCall{ctx: ctx, query: query, args: args})
	return f.scanner
}

// txRunner is a fakeRunner that also looks like *sql.Tx. Plain fakeRunners
// deliberately do not expose Commit/Rollback so they are not mistaken for
// transactions by session auto-detection.
type txRunner struct {
	*fakeRunner
}

func (t *txRunner) Commit() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.committed = true
	return nil
}

func (t *txRunner) Rollback() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rolledBack = true
	return nil
}

func newTxRunner() *txRunner {
	return &txRunner{fakeRunner: newFakeRunner("tx")}
}

func (f *fakeRunner) snapshotExecs() []fakeCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]fakeCall, len(f.execs))
	copy(out, f.execs)
	return out
}

func (f *fakeRunner) snapshotQueries() []fakeCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]fakeCall, len(f.queries))
	copy(out, f.queries)
	return out
}

func (f *fakeRunner) snapshotQueryRows() []fakeCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]fakeCall, len(f.queryRows))
	copy(out, f.queryRows)
	return out
}

type markerScanner struct {
	origin  string
	scanErr error
}

func (m *markerScanner) Scan(...interface{}) error { return m.scanErr }

// seqFence is an opaque token for the fake provider. The field is unexported
// on purpose: the causal library must never interpret the token, only the
// fake provider may.
type seqFence struct{ seq int64 }

func (f seqFence) FenceToken() interface{} { return f.seq }

type waitRecord struct {
	ctx      context.Context
	identity SessionIdentity
	seq      int64
}

// fakeProvider implements a controllable replication watermark. Its
// partial order is the integer sequence it owns; merging means taking the
// maximum sequence. The library under test never sees these details.
type fakeProvider struct {
	mu        sync.Mutex
	cond      *sync.Cond
	watermark map[string]int64
	nextSeq   int64

	identities []SessionIdentity
	commits    []int64
	waits      []waitRecord
	merges     [][]int64

	commitErr error
	waitErr   error
	mergeErr  error

	waitStarted chan int64
}

func newFakeProvider() *fakeProvider {
	p := &fakeProvider{
		watermark:   make(map[string]int64),
		waitStarted: make(chan int64, 32),
	}
	p.cond = sync.NewCond(&p.mu)
	return p
}

func identityKey(id SessionIdentity) string {
	return id.Tenant + "/" + id.Database
}

func (p *fakeProvider) CommitFence(ctx context.Context, id SessionIdentity, result sql.Result) (Fence, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.identities = append(p.identities, id)
	if p.commitErr != nil {
		return nil, p.commitErr
	}
	p.nextSeq++
	p.commits = append(p.commits, p.nextSeq)
	return seqFence{seq: p.nextSeq}, nil
}

func (p *fakeProvider) WaitForReplicated(ctx context.Context, id SessionIdentity, replica ReplicaReader, required Fence) error {
	seq := required.(seqFence).seq

	p.mu.Lock()
	p.waits = append(p.waits, waitRecord{ctx: ctx, identity: id, seq: seq})
	select {
	case p.waitStarted <- seq:
	default:
	}

	stop := make(chan struct{})
	defer close(stop)
	go func() {
		select {
		case <-ctx.Done():
			p.mu.Lock()
			p.cond.Broadcast()
			p.mu.Unlock()
		case <-stop:
		}
	}()

	if p.waitErr != nil {
		err := p.waitErr
		p.mu.Unlock()
		return err
	}
	for p.watermark[identityKey(id)] < seq {
		if err := ctx.Err(); err != nil {
			p.mu.Unlock()
			return err
		}
		p.cond.Wait()
	}
	p.mu.Unlock()
	return nil
}

func (p *fakeProvider) Merge(id SessionIdentity, fences ...Fence) (Fence, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.mergeErr != nil {
		return nil, p.mergeErr
	}
	seqs := make([]int64, 0, len(fences))
	var max int64
	for _, f := range fences {
		s := f.(seqFence).seq
		seqs = append(seqs, s)
		if s > max {
			max = s
		}
	}
	p.merges = append(p.merges, seqs)
	return seqFence{seq: max}, nil
}

// advance moves the replica watermark for identity to seq and wakes waiters.
func (p *fakeProvider) advance(id SessionIdentity, seq int64) {
	p.mu.Lock()
	p.watermark[identityKey(id)] = seq
	p.cond.Broadcast()
	p.mu.Unlock()
}

func (p *fakeProvider) snapshotCommits() []int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]int64, len(p.commits))
	copy(out, p.commits)
	return out
}

func (p *fakeProvider) snapshotWaits() []waitRecord {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]waitRecord, len(p.waits))
	copy(out, p.waits)
	return out
}

func (p *fakeProvider) snapshotMerges() [][]int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([][]int64, len(p.merges))
	copy(out, p.merges)
	return out
}

func (p *fakeProvider) snapshotIdentities() []SessionIdentity {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]SessionIdentity, len(p.identities))
	copy(out, p.identities)
	return out
}

func (p *fakeProvider) snapshotWaitSeqs() []int64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]int64, len(p.waits))
	for i, w := range p.waits {
		out[i] = w.seq
	}
	return out
}

var testIdentity = SessionIdentity{Tenant: "tenant-a", Database: "db-a"}

func newCausalHarness(t *testing.T, policy FallbackPolicy) (*fakeRunner, *fakeRunner, *fakeProvider, *CausalRunner) {
	t.Helper()
	primary := newFakeRunner("primary")
	replica := newFakeRunner("replica")
	prov := newFakeProvider()
	runner := NewCausalRunner(primary, replica, prov, WithFallbackPolicy(policy))
	return primary, replica, prov, runner
}

// --- tests -----------------------------------------------------------------

func TestCausalReadOnlyWithoutPriorWriteGoesStraightToReplica(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	ctx := WithRouteIntent(context.Background(), RouteReadOnly)
	_, err := sess.QueryContext(ctx, "SELECT 1")

	assert.True(t, errors.Is(err, errReplicaMarker), "expected replica marker, got %v", err)
	assert.Empty(t, primary.snapshotQueries())
	assert.Len(t, replica.snapshotQueries(), 1)
	assert.Empty(t, prov.snapshotWaits(), "no fence exists yet, so no wait is needed")
}

func TestCausalReadAfterWriteWaitsForAuthoritativeFenceThenReadsReplica(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	res, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)
	assert.Nil(t, res)
	assert.Equal(t, []fakeCall{{ctx: context.Background(), query: "UPDATE t SET v = 1"}}, stripArgs(primary.snapshotExecs()))
	assert.Equal(t, []int64{1}, prov.snapshotCommits(), "fence must come from the provider, not a local counter")

	// Replica is behind: a read must block until the watermark advances.
	prov.advance(testIdentity, 1)

	ctx := WithRouteIntent(context.Background(), RouteReadOnly)
	_, err = sess.QueryContext(ctx, "SELECT v FROM t")
	assert.True(t, errors.Is(err, errReplicaMarker))
	assert.Equal(t, []int64{1}, prov.snapshotWaitSeqs())
	assert.Equal(t, testIdentity, prov.snapshotWaits()[0].identity)
	assert.Empty(t, primary.snapshotQueries())
	assert.Len(t, replica.snapshotQueries(), 1)

	// QueryRow must be served by the same selected runner and keep Scan semantics.
	scanner := sess.QueryRowContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT v FROM t LIMIT 1")
	row, ok := scanner.(*markerScanner)
	assert.True(t, ok)
	assert.Equal(t, "replica", row.origin)
}

func TestCausalDefaultAndPrimaryRequiredIntentsStayOnPrimary(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)
	prov.advance(testIdentity, 1)

	// Undeclared intent: primary, even though the replica is caught up.
	_, err = sess.QueryContext(context.Background(), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))

	// Explicit PrimaryRequired: primary without any fence wait.
	_, err = sess.QueryContext(WithRouteIntent(context.Background(), RoutePrimaryRequired), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))

	assert.Empty(t, replica.snapshotQueries())
	assert.Empty(t, prov.snapshotWaits())
	assert.Len(t, primary.snapshotQueries(), 2)

	// Exec always goes to primary regardless of any intent in the context.
	_, err = sess.ExecContext(WithRouteIntent(context.Background(), RouteReadOnly), "UPDATE t SET v = 2")
	assert.NoError(t, err)
	assert.Len(t, primary.snapshotExecs(), 2)
	assert.Empty(t, replica.snapshotExecs())
}

func TestCausalReplicaLagFallbackToPrimarySharesContext(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FallbackToPrimary)
	sess := runner.NewSession(testIdentity)

	_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)
	// Watermark deliberately left at 0: the replica is behind fence 1.

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	readCtx := WithRouteIntent(ctx, RouteReadOnly)
	_, err = sess.QueryContext(readCtx, "SELECT v FROM t")

	assert.True(t, errors.Is(err, errPrimaryMarker), "fallback must serve from primary, got %v", err)
	assert.Empty(t, replica.snapshotQueries(), "replica must not be queried after a failed wait")
	assert.Len(t, primary.snapshotQueries(), 1)
	// The fallback driver call and the fence wait share the original context
	// and its remaining time; no fresh timeout context is created.
	assert.Equal(t, readCtx, primary.snapshotQueries()[0].ctx)
	assert.Equal(t, readCtx, prov.snapshotWaits()[0].ctx)
}

func TestCausalReplicaLagFailClosedReturnsError(t *testing.T) {
	primary, replica, _, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = sess.QueryContext(WithRouteIntent(ctx, RouteReadOnly), "SELECT v FROM t")
	assert.True(t, errors.Is(err, ErrReplicaNotCaughtUp), "got %v", err)
	assert.Empty(t, primary.snapshotQueries())
	assert.Empty(t, replica.snapshotQueries())

	// QueryRow routing errors must surface through Scan, not a nil scanner.
	scanner := sess.QueryRowContext(WithRouteIntent(ctx, RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(scanner.Scan(), ErrReplicaNotCaughtUp))
}

func TestCausalProviderWaitErrorFollowsPolicy(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FallbackToPrimary)
	sess := runner.NewSession(testIdentity)
	_, _ = sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	prov.waitErr = errBoom

	_, err := sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker), "fallback policy should route to primary, got %v", err)
	assert.Len(t, primary.snapshotQueries(), 1)

	// A second runner with the fail-closed policy must report the failure
	// instead. Its own write gives it a fence that the wait must honor.
	failClosed := NewCausalRunner(primary, replica, prov, WithFallbackPolicy(FailClosed))
	sess2 := failClosed.NewSession(testIdentity)
	_, err = sess2.ExecContext(context.Background(), "UPDATE t SET v = 2")
	assert.NoError(t, err)
	prov.waitErr = errBoom
	_, err = sess2.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, ErrReplicaNotCaughtUp))
}

func TestCausalCanceledContextIsNotFallback(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FallbackToPrimary)
	sess := runner.NewSession(testIdentity)
	_, _ = sess.ExecContext(context.Background(), "UPDATE t SET v = 1")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := sess.QueryContext(WithRouteIntent(ctx, RouteReadOnly), "SELECT 1")
		done <- err
	}()

	select {
	case seq := <-prov.waitStarted:
		assert.Equal(t, int64(1), seq)
	case <-time.After(time.Second):
		t.Fatal("fence wait never started")
	}
	cancel()

	select {
	case err := <-done:
		assert.True(t, errors.Is(err, context.Canceled), "got %v", err)
	case <-time.After(time.Second):
		t.Fatal("read did not return after cancellation")
	}
	assert.Empty(t, primary.snapshotQueries(), "no time remains, fallback must not run")
	assert.Empty(t, replica.snapshotQueries())
}

func TestCausalConcurrentReadSnapshotsFenceAtStartBoundary(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	// Write 1 establishes fence 1; the replica watermark stays at 0.
	_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)

	readDone := make(chan error, 1)
	go func() {
		_, e := sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
		readDone <- e
	}()
	select {
	case seq := <-prov.waitStarted:
		assert.Equal(t, int64(1), seq)
	case <-time.After(time.Second):
		t.Fatal("first read never started waiting")
	}

	// A second write completes while the read is already waiting. It must not
	// extend the read's wait: the read's boundary fence is fence 1.
	_, err = sess.ExecContext(context.Background(), "UPDATE t SET v = 2")
	assert.NoError(t, err)
	assert.Equal(t, []int64{1, 2}, prov.snapshotCommits())

	// Catching up to fence 1 must release the read; fence 2 is not chased.
	prov.advance(testIdentity, 1)
	select {
	case err := <-readDone:
		assert.True(t, errors.Is(err, errReplicaMarker), "read must finish on replica, got %v", err)
	case <-time.After(time.Second):
		t.Fatal("read did not unblock at its boundary fence")
	}
	assert.Equal(t, []int64{1}, prov.snapshotWaitSeqs(), "the read must not chase fence 2")
	assert.Len(t, replica.snapshotQueries(), 1)
	assert.Empty(t, primary.snapshotQueries())

	// A later read picks up the merged boundary (fence 2 dominates fence 1).
	prov.advance(testIdentity, 2)
	_, err = sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 2")
	assert.True(t, errors.Is(err, errReplicaMarker))
	assert.Equal(t, []int64{1, 2}, prov.snapshotWaitSeqs())
}

func TestCausalConcurrentWritesMergeUnderProviderPartialOrder(t *testing.T) {
	_, _, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
			assert.NoError(t, err)
		}()
	}
	close(start)
	wg.Wait()

	assert.Equal(t, []int64{1, 2}, prov.snapshotCommits())
	assert.Equal(t, [][]int64{{1, 2}}, prov.snapshotMerges(), "merge is defined by the provider, never string-maxed by the library")

	sess.mu.Lock()
	req := sess.req
	sess.mu.Unlock()
	assert.Equal(t, seqFence{seq: 2}, req)
}

func TestCausalWriteErrorMarksSessionStickyWithoutRetry(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)
	primary.execErr = errBoom

	_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.True(t, errors.Is(err, errBoom))
	assert.Len(t, primary.snapshotExecs(), 1, "an uncertain write must never be auto-retried")
	assert.Empty(t, prov.snapshotCommits(), "no fence is claimed for a failed write")
	assert.True(t, sess.PrimarySticky())

	// Sticky sessions keep using the primary; the old (non-existent here)
	// watermark can never drive a replica read.
	_, err = sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))
	assert.Empty(t, replica.snapshotQueries())
	assert.Empty(t, prov.snapshotWaits())
}

func TestCausalFenceErrorMarksSessionSticky(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)
	prov.commitErr = errBoom

	res, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.True(t, errors.Is(err, ErrFenceUnavailable))
	assert.Nil(t, res)
	assert.True(t, sess.PrimarySticky())
	assert.Len(t, primary.snapshotExecs(), 1, "the committed write is not retried")

	_, err = sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))
	assert.Empty(t, replica.snapshotQueries())
	assert.Empty(t, prov.snapshotWaits())
}

func TestCausalMergeErrorMarksSessionSticky(t *testing.T) {
	_, _, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)

	prov.mergeErr = errBoom
	_, err = sess.ExecContext(context.Background(), "UPDATE t SET v = 2")
	assert.True(t, errors.Is(err, ErrFenceUnavailable))
	assert.True(t, sess.PrimarySticky())

	_, err = sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))
}

func TestCausalTransactionalSessionIsPinnedToPrimaryTx(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	tx := newTxRunner()
	sess := runner.TransactionalSession(testIdentity, tx)
	assert.True(t, sess.Transactional())

	_, err := sess.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)
	_, err = sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))
	scanner := sess.QueryRowContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.Equal(t, "tx", scanner.(*markerScanner).origin)

	assert.Len(t, tx.snapshotExecs(), 1)
	assert.Len(t, tx.snapshotQueries(), 1)
	assert.Len(t, tx.snapshotQueryRows(), 1)
	assert.Empty(t, replica.snapshotQueries())
	assert.Empty(t, primary.snapshotExecs())
	assert.Empty(t, prov.snapshotWaits())
}

func TestCausalAutoDetectsTransactionalPrimary(t *testing.T) {
	replica := newFakeRunner("replica")
	prov := newFakeProvider()
	tx := newTxRunner()
	runner := NewCausalRunner(tx, replica, prov)

	sess := runner.NewSession(testIdentity)
	assert.True(t, sess.Transactional())

	_, err := sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))
	assert.Empty(t, replica.snapshotQueries())
}

func TestCausalSessionsAreIsolatedByIdentity(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	other := SessionIdentity{Tenant: "tenant-b", Database: "db-b"}

	sessA := runner.NewSession(testIdentity)
	_, err := sessA.ExecContext(context.Background(), "UPDATE t SET v = 1")
	assert.NoError(t, err)

	// A different tenant/database has no causal relationship with A's write.
	sessB := runner.NewSession(other)
	_, err = sessB.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errReplicaMarker))
	assert.Empty(t, prov.snapshotWaits())
	assert.Len(t, replica.snapshotQueries(), 1)
	assert.Empty(t, primary.snapshotQueries())

	// The provider is always told which identity a fence belongs to.
	assert.Equal(t, []SessionIdentity{testIdentity}, prov.snapshotIdentities())
}

func TestCausalRunnerUsesSessionHandleFromContext(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	// A write through the runner with a session handle advances that session.
	ctx := ContextWithSession(context.Background(), sess)
	_, err := runner.ExecContext(ctx, "UPDATE t SET v = 1")
	assert.NoError(t, err)
	prov.advance(testIdentity, 1)

	readCtx := WithRouteIntent(ctx, RouteReadOnly)
	_, err = runner.QueryContext(readCtx, "SELECT 1")
	assert.True(t, errors.Is(err, errReplicaMarker))
	assert.Len(t, replica.snapshotQueries(), 1)
	assert.Equal(t, []int64{1}, prov.snapshotWaitSeqs())
	assert.Equal(t, readCtx, prov.snapshotWaits()[0].ctx)

	// Without a handle, intent alone cannot establish a causal chain: primary.
	_, err = runner.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))
	assert.Len(t, primary.snapshotQueries(), 1)

	got, ok := SessionFromContext(readCtx)
	assert.True(t, ok)
	assert.True(t, got == sess, "context must carry the same session handle")
	_, ok = SessionFromContext(context.Background())
	assert.False(t, ok)
}

func TestCausalNoContextMethodsStayOnPrimary(t *testing.T) {
	primary, replica, prov, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	_, err := sess.Exec("UPDATE t SET v = 1")
	assert.NoError(t, err)
	_, err = sess.Query("SELECT 1")
	assert.True(t, errors.Is(err, errPrimaryMarker))
	scanner := sess.QueryRow("SELECT 1")
	assert.Equal(t, "primary", scanner.(*markerScanner).origin)

	assert.Len(t, primary.snapshotExecs(), 1)
	assert.Len(t, primary.snapshotQueries(), 1)
	assert.Len(t, primary.snapshotQueryRows(), 1)
	assert.Empty(t, replica.snapshotQueries())
	assert.Empty(t, prov.snapshotWaits())
}

func TestCausalRowsAndScanErrorsArePassedThrough(t *testing.T) {
	_, replica, _, runner := newCausalHarness(t, FailClosed)
	sess := runner.NewSession(testIdentity)

	// The selected runner's Query error is returned as-is; nothing is
	// buffered for routing and there is no wrapping.
	_, err := sess.QueryContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1")
	assert.True(t, err == errReplicaMarker, "expected the replica marker error untouched, got %v", err)

	// A scanner error still reaches the caller through Scan.
	replica.scanner = &markerScanner{origin: "replica", scanErr: sql.ErrNoRows}
	err = sess.QueryRowContext(WithRouteIntent(context.Background(), RouteReadOnly), "SELECT 1").Scan()
	assert.True(t, err == sql.ErrNoRows, "expected sql.ErrNoRows untouched, got %v", err)
}

// --- helpers ---------------------------------------------------------------

func stripArgs(calls []fakeCall) []fakeCall {
	out := make([]fakeCall, len(calls))
	for i, c := range calls {
		out[i] = fakeCall{ctx: c.ctx, query: c.query}
	}
	return out
}
