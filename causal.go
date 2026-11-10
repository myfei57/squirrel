//go:build go1.8
// +build go1.8

package squirrel

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sync"
)

// RouteIntent is the routing intent declared explicitly by a trusted caller.
//
// The library never infers intent from the SQL string (for example by looking
// for a "SELECT" prefix): a read-only looking statement may still have side
// effects, so only an explicit declaration can safely route work to a replica.
type RouteIntent int

const (
	// RouteUnspecified declares no routing intent. Work is served from the
	// primary, exactly like an Exec call. This is the zero value.
	RouteUnspecified RouteIntent = iota

	// RouteReadOnly declares that the query has no side effects and may be
	// served from a replica once the replica has reached the session's
	// required fence.
	RouteReadOnly

	// RoutePrimaryRequired declares that the query must run on the primary
	// even when the replica is already caught up.
	RoutePrimaryRequired
)

// FallbackPolicy decides what happens when a declared read-only query cannot
// be served from a replica because the fence wait fails (including a wait
// timeout). Explicit context cancellation is never rerouted.
type FallbackPolicy int

const (
	// FailClosed returns the fence wait error instead of serving a possibly
	// stale read. This is the zero value and the safe default.
	FailClosed FallbackPolicy = iota

	// FallbackToPrimary serves the read from the primary, spending whatever
	// time remains in the original context.
	FallbackToPrimary
)

// Errors returned by the causal routing layer.
var (
	// ErrFenceUnavailable is returned when the authoritative fence for a
	// successful primary write could not be obtained or merged. The session
	// is marked primary-sticky in that case.
	ErrFenceUnavailable = errors.New("squirrel: authoritative commit fence is unavailable")

	// ErrReplicaNotCaughtUp wraps a fence wait failure when the policy is
	// FailClosed.
	ErrReplicaNotCaughtUp = errors.New("squirrel: replica did not reach the required fence")
)

// SessionIdentity namespaces causal state. Sessions created for different
// tenants or databases never share fences or sticky state.
type SessionIdentity struct {
	Tenant   string
	Database string
}

// Fence is an opaque replication progress token issued by a FenceProvider.
//
// The causal routing library never inspects, compares or serializes the
// token value: it has no notion of fence ordering. All partial-order
// decisions live in the FenceProvider implementation.
type Fence interface {
	// FenceToken returns the provider-defined opaque progress token.
	FenceToken() interface{}
}

// ReplicaReader is the view of a replica runner handed to a FenceProvider so
// it can query the replica's replication watermark itself.
type ReplicaReader interface {
	QueryerContext
	QueryRowerContext
}

// FenceProvider supplies authoritative replication fences and defines the
// partial order between them.
//
// Implementations must not derive progress from wall-clock time or process
// local counters: CommitFence must return the fence associated with the
// durable commit result of the write. This package does not implement
// database replication or any network service; production implementations
// are expected to query the underlying database's own replication metadata
// (for example GTID sets or LSNs).
type FenceProvider interface {
	// CommitFence returns the authoritative fence associated with the
	// result of a successful primary write.
	CommitFence(ctx context.Context, identity SessionIdentity, result sql.Result) (Fence, error)

	// WaitForReplicated blocks until the given replica has applied every
	// change up to and including the required fence, or until ctx is done.
	// The implementation owns the comparison between the replica watermark
	// and the required token; the library assumes nothing about token
	// ordering.
	WaitForReplicated(ctx context.Context, identity SessionIdentity, replica ReplicaReader, required Fence) error

	// Merge returns a single fence that dominates every input fence under
	// the provider's partial order. Merging is the provider's
	// responsibility precisely because arbitrary tokens cannot be ordered
	// by the library (for example with a string max).
	Merge(identity SessionIdentity, fences ...Fence) (Fence, error)
}

// CausalOption customizes a CausalRunner at construction time.
type CausalOption func(*CausalRunner)

// WithFallbackPolicy sets the explicit policy used when a replica fence wait
// fails with time remaining in the context.
func WithFallbackPolicy(policy FallbackPolicy) CausalOption {
	return func(r *CausalRunner) {
		r.policy = policy
	}
}

// CausalRunner routes explicitly declared read-only queries to a replica while
// keeping read-your-writes causality within request-local CausalSessions.
//
// A CausalRunner is safe for concurrent use. It holds no per-request state
// itself: all causal state lives in CausalSession values that callers pass
// explicitly (either by calling the session directly or by attaching it to a
// context with ContextWithSession).
type CausalRunner struct {
	primary RunnerContext
	replica RunnerContext
	fences  FenceProvider
	policy  FallbackPolicy
}

// NewCausalRunner builds a runner backed by the given primary and replica.
// The fence provider is required: it is the only source of authoritative
// replication progress.
func NewCausalRunner(primary, replica RunnerContext, fences FenceProvider, opts ...CausalOption) *CausalRunner {
	r := &CausalRunner{
		primary: primary,
		replica: replica,
		fences:  fences,
		policy:  FailClosed,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// NewSession starts a request-local causal session bound to identity.
//
// If the configured primary runner looks like a transaction (it implements
// Commit and Rollback, as *sql.Tx and its wrappers do) the session is pinned
// to that transaction: every operation runs on it and replica routing is
// disabled. Use TransactionalSession to pin an explicitly wrapped
// transaction that does not expose those methods.
func (r *CausalRunner) NewSession(identity SessionIdentity) *CausalSession {
	s := &CausalSession{
		runner:   r,
		identity: identity,
	}
	if _, ok := r.primary.(txLike); ok {
		s.inTx = true
		s.tx = r.primary
	}
	return s
}

// TransactionalSession starts a session pinned to an existing primary
// transaction. An open transaction always stays on its original primary
// connection: the library never switches a transactional session to a
// replica.
func (r *CausalRunner) TransactionalSession(identity SessionIdentity, tx RunnerContext) *CausalSession {
	return &CausalSession{
		runner:   r,
		identity: identity,
		inTx:     true,
		tx:       tx,
	}
}

// txLike is the shape shared by *sql.Tx and runners that embed it.
type txLike interface {
	Commit() error
	Rollback() error
}

// CausalSession holds the causal state of one request: the authoritative
// fence that replica reads must wait for, and the primary-sticky flag.
//
// A session is safe for concurrent use. It must not be shared across tenants
// or databases; create one session per SessionIdentity instead.
type CausalSession struct {
	runner   *CausalRunner
	identity SessionIdentity

	mu     sync.Mutex
	req    Fence // latest fence dominating every acknowledged primary write
	sticky bool  // set when a write or fence outcome is uncertain
	inTx   bool
	tx     RunnerContext
}

// Identity returns the tenant/database identity the session is bound to.
func (s *CausalSession) Identity() SessionIdentity {
	return s.identity
}

// PrimarySticky reports whether the session has been pinned to the primary
// after an uncertain write or fence outcome. Sticky sessions never return to
// replica routing.
func (s *CausalSession) PrimarySticky() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sticky
}

// Transactional reports whether the session is pinned to a primary
// transaction.
func (s *CausalSession) Transactional() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inTx
}

func (s *CausalSession) primaryTarget() RunnerContext {
	if s.inTx {
		return s.tx
	}
	return s.runner.primary
}

// markSticky pins the session to the primary. It is called whenever the
// outcome of a write or of fence acquisition is uncertain, so that a stale
// replica watermark can never be used for subsequent reads.
func (s *CausalSession) markSticky() {
	s.mu.Lock()
	s.sticky = true
	s.mu.Unlock()
}

// recordCommit stores the authoritative fence of a confirmed primary write,
// merging it with the previously required fence under the provider's
// partial order.
func (s *CausalSession) recordCommit(f Fence) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.sticky {
		// A sticky session stays on the primary; no point merging fences it
		// will never wait on.
		return nil
	}
	if s.req == nil {
		s.req = f
		return nil
	}
	merged, err := s.runner.fences.Merge(s.identity, s.req, f)
	if err != nil {
		s.sticky = true
		return err
	}
	s.req = merged
	return nil
}

// ExecContext always runs on the primary. On success it asks the
// FenceProvider for the authoritative fence associated with the commit
// result. Any uncertainty (write error, fence error or merge error) marks
// the session primary-sticky; the write is never retried automatically.
func (s *CausalSession) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	res, err := s.primaryTarget().ExecContext(ctx, query, args...)
	if err != nil {
		// The write may have executed despite the error; its commit
		// watermark is unknown, so replica reads are no longer safe.
		s.markSticky()
		return res, err
	}

	fence, ferr := s.runner.fences.CommitFence(ctx, s.identity, res)
	if ferr != nil {
		// The write committed, but we cannot prove which watermark it
		// reached. Pin future reads to the primary.
		s.markSticky()
		return res, fmt.Errorf("%w: %v", ErrFenceUnavailable, ferr)
	}

	if merr := s.recordCommit(fence); merr != nil {
		s.markSticky()
		return res, fmt.Errorf("%w: %v", ErrFenceUnavailable, merr)
	}
	return res, nil
}

// readPlan chooses the runner for a read and performs the fence wait when a
// replica read is permitted. The required fence is snapshotted at the read's
// start boundary: writes completing afterwards are not chased.
func (s *CausalSession) readPlan(ctx context.Context) (RunnerContext, error) {
	intent := routeIntentFromContext(ctx)

	s.mu.Lock()
	required := s.req
	sticky := s.sticky
	inTx := s.inTx
	s.mu.Unlock()

	if inTx || sticky || intent != RouteReadOnly {
		return s.primaryTarget(), nil
	}
	// No acknowledged write in this session: there is no watermark to wait
	// for, so a declared read-only query can use the replica directly.
	if required == nil {
		return s.runner.replica, nil
	}

	if err := s.runner.fences.WaitForReplicated(ctx, s.identity, s.runner.replica, required); err != nil {
		// Explicit cancellation is always propagated: no fallback work can
		// be meaningful once the caller has abandoned the request.
		if ctx.Err() == context.Canceled {
			return nil, context.Canceled
		}
		// Every other wait failure (notably a timeout) follows the explicit
		// fallback policy. The fallback shares ctx and whatever time it has
		// left; no fresh deadline is created.
		if s.runner.policy == FallbackToPrimary {
			return s.primaryTarget(), nil
		}
		return nil, fmt.Errorf("%w: %v", ErrReplicaNotCaughtUp, err)
	}
	return s.runner.replica, nil
}

// QueryContext serves the query from the runner chosen by readPlan. The
// returned *sql.Rows comes directly from the selected runner and is closed
// by the original caller; results are never buffered for routing.
func (s *CausalSession) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	target, err := s.readPlan(ctx)
	if err != nil {
		return nil, err
	}
	return target.QueryContext(ctx, query, args...)
}

// QueryRowContext serves the row from the runner chosen by readPlan. Any
// routing error is deferred to the returned RowScanner's Scan method,
// matching database/sql.QueryRow semantics.
func (s *CausalSession) QueryRowContext(ctx context.Context, query string, args ...interface{}) RowScanner {
	target, err := s.readPlan(ctx)
	if err != nil {
		return &Row{err: err}
	}
	return target.QueryRowContext(ctx, query, args...)
}

// Exec runs on the primary with a background context; see ExecContext.
func (s *CausalSession) Exec(query string, args ...interface{}) (sql.Result, error) {
	return s.ExecContext(context.Background(), query, args...)
}

// Query runs with a background context and no routing intent, hence on the
// primary; see QueryContext.
func (s *CausalSession) Query(query string, args ...interface{}) (*sql.Rows, error) {
	return s.QueryContext(context.Background(), query, args...)
}

// QueryRow runs with a background context and no routing intent, hence on
// the primary; see QueryRowContext.
func (s *CausalSession) QueryRow(query string, args ...interface{}) RowScanner {
	return s.QueryRowContext(context.Background(), query, args...)
}

// The context keys are unexported typed values: the session handle and route
// intent are explicit, request-scoped values, never goroutine-local or
// global state.
type causalContextKey int

const (
	sessionContextKey causalContextKey = iota
	intentContextKey
)

// ContextWithSession attaches an explicit session handle to ctx. Derived
// contexts carry the same handle; no new session is ever created from a
// context alone.
func ContextWithSession(ctx context.Context, session *CausalSession) context.Context {
	return context.WithValue(ctx, sessionContextKey, session)
}

// SessionFromContext retrieves the explicit session handle attached with
// ContextWithSession.
func SessionFromContext(ctx context.Context) (*CausalSession, bool) {
	s, ok := ctx.Value(sessionContextKey).(*CausalSession)
	return s, ok
}

// WithRouteIntent attaches an explicit routing intent to ctx.
func WithRouteIntent(ctx context.Context, intent RouteIntent) context.Context {
	return context.WithValue(ctx, intentContextKey, intent)
}

func routeIntentFromContext(ctx context.Context) RouteIntent {
	if intent, ok := ctx.Value(intentContextKey).(RouteIntent); ok {
		return intent
	}
	return RouteUnspecified
}

// ExecContext uses the session handle carried by ctx when present. Without a
// session there is no causal state to maintain, so the write goes straight to
// the primary and no fence is acquired.
func (r *CausalRunner) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	if s, ok := SessionFromContext(ctx); ok {
		return s.ExecContext(ctx, query, args...)
	}
	return r.primary.ExecContext(ctx, query, args...)
}

// QueryContext uses the session handle carried by ctx when present. Without
// a session, routing intent cannot be tied to any causal chain, so the query
// is served from the primary.
func (r *CausalRunner) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	if s, ok := SessionFromContext(ctx); ok {
		return s.QueryContext(ctx, query, args...)
	}
	return r.primary.QueryContext(ctx, query, args...)
}

// QueryRowContext uses the session handle carried by ctx when present.
// Without a session the query is served from the primary.
func (r *CausalRunner) QueryRowContext(ctx context.Context, query string, args ...interface{}) RowScanner {
	if s, ok := SessionFromContext(ctx); ok {
		return s.QueryRowContext(ctx, query, args...)
	}
	return r.primary.QueryRowContext(ctx, query, args...)
}

// Exec runs on the primary with a background context; see ExecContext.
func (r *CausalRunner) Exec(query string, args ...interface{}) (sql.Result, error) {
	return r.ExecContext(context.Background(), query, args...)
}

// Query runs with a background context and no routing intent, hence on the
// primary; see QueryContext.
func (r *CausalRunner) Query(query string, args ...interface{}) (*sql.Rows, error) {
	return r.QueryContext(context.Background(), query, args...)
}

// QueryRow runs with a background context and no routing intent, hence on
// the primary; see QueryRowContext.
func (r *CausalRunner) QueryRow(query string, args ...interface{}) RowScanner {
	return r.QueryRowContext(context.Background(), query, args...)
}

// Compile-time assertions that both types are full squirrel runners.
var (
	_ RunnerContext = (*CausalRunner)(nil)
	_ RunnerContext = (*CausalSession)(nil)
)
