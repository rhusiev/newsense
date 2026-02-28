mod state;
mod models;
mod utils;
mod handlers;
mod router;
mod extractors;
mod admin;

use axum::{
    http::{HeaderValue, Method, header},
};
use axum_csrf::{CsrfConfig, CsrfLayer};
use sqlx::postgres::PgPoolOptions;
use std::{net::SocketAddr, time};
use tower_governor::{GovernorLayer, governor::GovernorConfigBuilder};
use tower_http::cors::CorsLayer;
use tower_sessions::{Expiry, SessionManagerLayer, cookie::time::Duration};
use tower_sessions_sqlx_store::PostgresStore;

use crate::{state::AppState, utils::SESSION_COOKIE_NAME};

#[tokio::main]
async fn main() {
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://user:pass@localhost/authdb".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&database_url)
        .await
        .expect("Failed to connect to Postgres");

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create users table");

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS remember_tokens (
            series UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create remember_tokens table");

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS access_codes (
            code VARCHAR(255) PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create access_codes table");

    let is_dev = cfg!(debug_assertions);
    let is_production = !is_dev;

    println!(
        "Server starting in {} mode (Secure Cookies: {})",
        if is_production {
            "PRODUCTION"
        } else {
            "DEVELOPMENT"
        },
        is_production
    );

    let session_store = PostgresStore::new(pool.clone());
    session_store
        .migrate()
        .await
        .expect("Failed to migrate session store");

    let cookie_domain = std::env::var("COOKIE_DOMAIN").unwrap_or_else(|_| ".localhost".to_string());

    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(is_production)
        .with_same_site(tower_sessions::cookie::SameSite::Lax)
        .with_expiry(Expiry::OnInactivity(Duration::new(3600, 0)))
        .with_name(SESSION_COOKIE_NAME)
        .with_domain(cookie_domain.clone());

    let registration_enabled = std::env::var("REGISTRATION_ENABLED")
        .map(|val| match val.to_lowercase().as_str() {
            "true" | "1" | "yes" => true,
            "false" | "0" | "no" => false,
            _ => true,
        })
        .unwrap_or(true);

    let app_state = AppState {
        db: pool,
        is_production,
        registration_enabled,
        cookie_domain,
    };
    let csrf_config = CsrfConfig::default();

    let governor_config = GovernorConfigBuilder::default()
        .per_second(20)
        .burst_size(30)
        .finish()
        .unwrap();

    let governor_limiter = governor_config.limiter().clone();
    tokio::spawn(async move {
        let interval = time::Duration::from_secs(60);
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            governor_limiter.retain_recent();
        }
    });

    let web_url = std::env::var("WEB_URL").unwrap_or_else(|_| "http://localhost:5173".to_string());

    let mut allowed_origins = vec![
        web_url.parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:5173".parse::<HeaderValue>().unwrap(),
        "http://localhost:5173".parse::<HeaderValue>().unwrap(),
    ];

    if let Ok(domain) = std::env::var("COOKIE_DOMAIN") {
        if let Some(stripped) = domain.strip_prefix('.') {
            let prod_origin = format!("https://{}", stripped);
            if let Ok(val) = prod_origin.parse::<HeaderValue>() {
                allowed_origins.push(val);
            }
        }
    }

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([
            Method::GET,
            Method::POST,
            Method::PATCH,
            Method::PUT,
            Method::DELETE,
            Method::OPTIONS,
        ])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            header::HeaderName::from_static("x-csrf-token"),
        ])
        .allow_credentials(true);

    let app = router::routes()
        .layer(session_layer)
        .layer(GovernorLayer::new(governor_config))
        .layer(CsrfLayer::new(csrf_config))
        .layer(cors)
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("Failed to bind");

    println!("Server running on http://0.0.0.0:3000");

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .expect("Server failed");
}
