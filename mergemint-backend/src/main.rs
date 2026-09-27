// mergemint-backend/src/main.rs
//
// Application entry-point: builds the Axum router with middleware and starts
// the HTTP server.
//
// ## Request body size limits and timeout middleware (#476)
//
// Two middleware layers are added to the router to protect the service from
// slow clients and excessively large payloads:
//
//   * `RequestBodyLimitLayer` — rejects bodies larger than `MAX_BODY_BYTES`.
//     Without this, a malicious client could stream an arbitrarily large body
//     and exhaust server memory before any handler logic runs.
//
//   * `TimeoutLayer` — cancels any request (including body reads and handler
//     execution) that takes longer than `REQUEST_TIMEOUT`.  This prevents slow
//     clients or downstream Horizon calls from holding connections indefinitely
//     and starving the thread pool.
//
// ## Request correlation IDs (#486)
//
// Every inbound request is stamped with a UUID v4 correlation ID by
// `SetRequestIdLayer`.  The ID is read from the `x-request-id` header when
// present (so callers can propagate their own trace ID), or generated fresh
// when absent.  `TraceLayer` then opens a tracing span for each request that
// includes the correlation ID, making it trivial to grep logs for a single
// user's flow even when requests are interleaved.
//
// ## Prometheus metrics (#867)
//
// A `metrics` middleware records a per-route request counter and a latency
// histogram, both labelled by `route` and `status`.  The indexer exposes a
// `mergemint_indexer_lag_ledgers` gauge for how far behind the latest ledger
// it is.  Everything is rendered in the Prometheus text exposition format at
// `GET /metrics`.
//
// ## Structured JSON logging (#868)
//
// The subscriber is configured to emit either the human-friendly pretty
// format (the default, for local development) or structured JSON when the
// `LOG_FORMAT=json` environment variable is set.  JSON output is what log
// aggregators expect: each record is a single object whose span fields —
// including the `request_id` correlation ID and the matched `route` — appear
// as top-level structured keys rather than being interpolated into free text.
//
// ## Graceful shutdown
//
// `axum::serve` is wired to `shutdown_signal`, which waits for SIGINT
// (Ctrl+C) or, on Unix, SIGTERM. Once either fires, Axum stops accepting new
// connections but lets in-flight requests finish — including a `self_claim`
// / `resolve_dispute` call that has already reached the point of submitting
// a chain transaction — instead of dropping them mid-flight when a deploy
// sends SIGTERM.
//
// ## Health checks
//
// `GET /health` returns `200 ok` once the server is listening. The runtime
// container image is distroless (no shell or curl), so the binary doubles as
// its own probe: `mergemint-backend healthcheck` requests `/health` on the
// local listener and exits non-zero if it doesn't get a 200.
//
// ## Readiness probe (#870)
//
// `GET /ready` pings the database and returns `200` when the connection is
// usable, or `503 Service Unavailable` when it is not.  Orchestrators such as
// Kubernetes or Docker Compose can use this to avoid routing traffic to an
// instance whose database connection is broken.  `/health` stays cheap and
// database-independent so it remains a pure liveness check.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{MatchedPath, State},
    http::{header::CONTENT_TYPE, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    limit::RequestBodyLimitLayer,
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::Level;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod db;
mod metrics;
mod rate_limit;
mod routes;

use db::{new_shared_db, new_shared_idempotency_store, ping, SharedDb};
use metrics::{metrics_handler, record_request};
use routes::bounties::{get_bounty_route, bounty_stream, claim_bounty, list_bounties, list_bounties_by_assignee};
use routes::tx::{new_shared_rate_limiter, resolve_dispute, self_claim, AppState};

/// Maximum allowed request body size (1 MiB).
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Maximum wall-clock time allowed for a single request, including body reads
/// and handler execution.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// The canonical header name used to carry the correlation ID across service
/// boundaries.  Clients may supply their own value; if absent a UUID v4 is
/// generated automatically by `SetRequestIdLayer`.
const REQUEST_ID_HEADER: &str = "x-request-id";

/// Address the HTTP server listens on.
const LISTEN_ADDR: &str = "0.0.0.0:8080";

/// Address the `healthcheck` subcommand probes.
const HEALTHCHECK_ADDR: &str = "127.0.0.1:8080";

/// Reward-token allowlist env var consumed by create-bounty flows.
const ALLOWLISTED_REWARD_TOKENS_ENV: &str = "ALLOWLISTED_REWARD_TOKENS";

/// Env var holding a comma-separated allow-list of origins permitted to make
/// cross-origin requests, e.g.
/// "https://app.mergemint.xyz,https://staging.mergemint.xyz".
const CORS_ALLOWED_ORIGINS_ENV: &str = "CORS_ALLOWED_ORIGINS";

/// Env var selecting the log output format.  When set to `json` (case
/// insensitive) the subscriber emits structured JSON; any other value — or an
/// unset variable — keeps the human-friendly pretty format used locally.
const LOG_FORMAT_ENV: &str = "LOG_FORMAT";

#[tokio::main]
async fn main() {
    if std::env::args().nth(1).as_deref() == Some("healthcheck") {
        std::process::exit(match healthcheck(HEALTHCHECK_ADDR) {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("healthcheck failed: {err}");
                1
            }
        });
    }

    // ---------------------------------------------------------------------------
    // Initialise structured logging (#486, #868)
    //
    // We use a layered subscriber so that:
    //  - RUST_LOG (or the compiled-in default "info") controls the verbosity.
    //  - The output format is selected by LOG_FORMAT: `json` emits one JSON
    //    object per record (for log aggregators), anything else keeps the
    //    pretty format for local development.
    //  - Span fields injected by TraceLayer — the `request_id` correlation ID
    //    and the matched `route` — are recorded as structured keys, so they
    //    surface as top-level fields in JSON output.
    // ---------------------------------------------------------------------------
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        // Default: info-level for our crate, warn for noisy deps.
        "mergemint_backend=info,tower_http=debug,axum::rejection=trace"
            .parse()
            .unwrap()
    });

    let registry = tracing_subscriber::registry().with(env_filter);

    if log_format_is_json() {
        registry.with(fmt::layer().json()).init();
    } else {
        registry.with(fmt::layer()).init();
    }

    warn_if_reward_token_allowlist_empty();

    let shared_db = new_shared_db();
    let idempotency = new_shared_idempotency_store();
    let (bounty_broadcast, _) = tokio::sync::broadcast::channel(100);
    let state = Arc::new(AppState {
        db: shared_db,
        idempotency,
        rate_limiter: new_shared_rate_limiter(),
        bounty_broadcast,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics_handler))
        .route("/tx/resolve-dispute", post(resolve_dispute))
        .route("/tx/self-claim", post(self_claim))
        .route("/bounties", get(list_bounties))
        .route("/bounties/:id", get(get_bounty_route))
        .route(
            "/bounties/assignee/:address",
            get(list_bounties_by_assignee),
        )
        // ── Bounty push channel (#482) ─────────────────────────────────────
        .route("/bounties/:id/claim", post(claim_bounty))
        .route("/bounties/stream", get(bounty_stream))
        .with_state(state)
        // ── Prometheus metrics middleware (#867) ───────────────────────────
        //
        // Records a per-route request counter and latency histogram, labelled
        // by the matched route and response status.
        .layer(middleware::from_fn(record_request))
        // ── Request correlation IDs (#486) ─────────────────────────────────
        .layer(PropagateRequestIdLayer::new(
            HeaderValue::from_static(REQUEST_ID_HEADER),
        ))
        .layer(SetRequestIdLayer::new(
            HeaderValue::from_static(REQUEST_ID_HEADER),
            MakeRequestUuid,
        ))
        // ── Tracing ────────────────────────────────────────────────────────
        .layer(TraceLayer::new_for_http())
        // ── CORS ───────────────────────────────────────────────────────────
        .layer(cors_layer())
        // ── Body size limit (#476) ─────────────────────────────────────────
        .layer(RequestBodyLimitLayer::new(MAX_BODY_BYTES))
        // ── Request timeout (#476) ─────────────────────────────────────────
        .layer(TimeoutLayer::new(REQUEST_TIMEOUT));

    let listener = tokio::net::TcpListener::bind(LISTEN_ADDR)
        .await
        .expect("failed to bind listener");

    tracing::info!(addr = LISTEN_ADDR, "mergemint-backend listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

/// Liveness probe: cheap, database-independent, always `200 ok` while the
/// process is running.
async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

/// Readiness probe (#870): pings the database and reports whether the
/// instance can serve traffic.  Returns `200 ok` when the ping succeeds and
/// `503 Service Unavailable` when the database is unreachable, so orchestrators
/// can stop routing traffic to a broken instance.
async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match ping(&state.db).await {
        Ok(()) => (StatusCode::OK, "ok"),
        Err(err) => {
            tracing::warn!(error = %err, "readiness probe failed: database unreachable");
            (StatusCode::SERVICE_UNAVAILABLE, "database unavailable")
        }
    }
}

/// Build the CORS layer from `CORS_ALLOWED_ORIGINS`.
fn cors_layer() -> CorsLayer {
    let origins = std::env::var(CORS_ALLOWED_ORIGINS_ENV).unwrap_or_default();
    let allowed: Vec<HeaderValue> = origins
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse::<HeaderValue>().ok())
        .collect();

    CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([CONTENT_TYPE])
        .allow_origin(AllowOrigin::list(allowed))
}

/// Returns `true` when `LOG_FORMAT=json` (case insensitive).
fn log_format_is_json() -> bool {
    std::env::var(LOG_FORMAT_ENV)
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

/// Warn once at startup when the reward-token allowlist is empty.
fn warn_if_reward_token_allowlist_empty() {
    let empty = std::env::var(ALLOWLISTED_REWARD_TOKENS_ENV)
        .map(|v| v.trim().is_empty())
        .unwrap_or(true);
    if empty {
        tracing::warn!(
            env = ALLOWLISTED_REWARD_TOKENS_ENV,
            "reward-token allowlist is empty; create-bounty flows will reject all tokens"
        );
    }
}

/// Wait for SIGINT (Ctrl+C) or, on Unix, SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }

    tracing::info!("shutdown signal received; draining in-flight requests");
}

/// Probe `/health` on the local listener; used by the `healthcheck` subcommand.
fn healthcheck(addr: &str) -> std::io::Result<()> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;

    if response.starts_with("HTTP/1.1 200") || response.starts_with("HTTP/1.0 200") {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("unexpected health response: {}", response.lines().next().unwrap_or("")),
        ))
    }
}
