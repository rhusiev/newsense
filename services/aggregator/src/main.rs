mod app_state;
mod auth;
mod handlers;
mod models;

use axum::{
    http::{HeaderValue, Method, header},
    routing::{get, post, put},
    Router,
};
use sqlx::postgres::PgPoolOptions;
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tower_sessions::{cookie::time::Duration, Expiry, SessionManagerLayer};
use tower_sessions_sqlx_store::PostgresStore;

use crate::{
    app_state::AppState,
    handlers::{
        get_all_items, get_all_unread_count, get_all_unread_counts, get_cluster_unread_count,
        get_clusters, get_feed_clusters, get_feed_items, get_feed_unread_count,
        mark_all_items_read, mark_feed_clusters_read, mark_feed_items_read, update_cluster_status,
        update_item_status,
    },
};

const SESSION_COOKIE_NAME: &str = "newsense_session";

#[tokio::main]
async fn main() {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://user:pass@localhost/authdb".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("Failed to connect to Postgres");

    let session_store = PostgresStore::new(pool.clone());
    session_store
        .migrate()
        .await
        .expect("Failed to migrate store");

    let is_dev = cfg!(debug_assertions);
    let is_production = !is_dev;

    println!(
        "Items Service running in {} mode (Secure Cookies: {})",
        if is_production {
            "PRODUCTION"
        } else {
            "DEVELOPMENT"
        },
        is_production
    );

    let cookie_domain = std::env::var("COOKIE_DOMAIN").unwrap_or_else(|_| ".localhost".to_string());

    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(is_production)
        .with_same_site(tower_sessions::cookie::SameSite::Lax)
        .with_expiry(Expiry::OnInactivity(Duration::new(3600, 0)))
        .with_name(SESSION_COOKIE_NAME)
        .with_domain(cookie_domain);

    let app_state = AppState { db: pool };

    let web_url = std::env::var("WEB_URL").unwrap_or_else(|_| "http://localhost:5173".to_string());

    let allowed_origins = [
        web_url.parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:5173".parse::<HeaderValue>().unwrap(),
        "http://localhost:5173".parse::<HeaderValue>().unwrap(),
    ];

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS, Method::PUT])
        .allow_headers([
            header::CONTENT_TYPE,
            header::HeaderName::from_static("x-csrf-token"),
        ])
        .allow_credentials(true);

    let app = Router::new()
        .route("/items/feed/{feed_id}", get(get_feed_items))
        .route(
            "/items/feed/{feed_id}/mark-read",
            post(mark_feed_items_read),
        )
        .route(
            "/items/feed/{feed_id}/unread-count",
            get(get_feed_unread_count),
        )
        .route("/items", get(get_all_items))
        .route("/items/mark-read", post(mark_all_items_read))
        .route("/items/unread-count", get(get_all_unread_count))
        .route("/items/unread-counts", get(get_all_unread_counts))
        .route("/items/{item_id}/status", put(update_item_status))
        .route("/clusters/feed/{feed_id}", get(get_feed_clusters))
        .route(
            "/clusters/feed/{feed_id}/mark-read",
            post(mark_feed_clusters_read),
        )
        .route("/clusters", get(get_clusters))
        .route("/clusters/{id}/status", put(update_cluster_status))
        .route("/clusters/unread-count", get(get_cluster_unread_count))
        .layer(session_layer)
        .layer(cors)
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3002")
        .await
        .expect("Failed to bind");

    println!("Items Service running on http://0.0.0.0:3002");

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .expect("Server failed");
}