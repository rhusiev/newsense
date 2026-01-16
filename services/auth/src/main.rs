use argon2::{
    Argon2,
    password_hash::{
        PasswordHash, PasswordHasher, PasswordVerifier, SaltString,
        rand_core::{OsRng, RngCore},
    },
};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderValue, Method, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use axum_csrf::{CsrfConfig, CsrfLayer};
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD as BASE64};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, postgres::PgPoolOptions, types::time::OffsetDateTime};
use std::{net::SocketAddr, time};
use tower_governor::{GovernorLayer, governor::GovernorConfigBuilder};
use tower_http::cors::CorsLayer;
use tower_sessions::{Expiry, Session, SessionManagerLayer, cookie::time::Duration};
use tower_sessions_sqlx_store::PostgresStore;
use uuid::Uuid;

const REMEMBER_COOKIE_NAME: &str = "remember_me";
const REMEMBER_DURATION_DAYS: i64 = 30;
const SESSION_COOKIE_NAME: &str = "newsense_session";

#[derive(Clone)]
struct AppState {
    db: PgPool,
    is_production: bool,
    registration_enabled: bool,
}

#[derive(Deserialize)]
struct RegisterRequest {
    username: String,
    password: String,
}

#[derive(Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
    #[serde(default)]
    remember_me: bool,
}

#[derive(Serialize)]
struct AuthResponse {
    user_id: Uuid,
    username: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

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
        .with_domain(cookie_domain);

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
    };
    let csrf_config = CsrfConfig::default();

    let governor_config = GovernorConfigBuilder::default()
        .per_second(5)
        .burst_size(5)
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

    let allowed_origins = [
        web_url.parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:5173".parse::<HeaderValue>().unwrap(),
        "http://localhost:5173".parse::<HeaderValue>().unwrap(),
    ];

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            header::CONTENT_TYPE,
            header::HeaderName::from_static("x-csrf-token"),
        ])
        .allow_credentials(true);

    let app = Router::new()
        .route("/register", post(register))
        .route("/login", post(login))
        .route("/logout", post(logout))
        .route("/me", get(current_user))
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

fn generate_token_pair() -> (String, String) {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    let token = BASE64.encode(bytes);

    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let hash = BASE64.encode(hasher.finalize());

    (token, hash)
}

fn build_cookie_value(user_id: Uuid, series: Uuid, token: &str) -> String {
    format!("{}:{}:{}", user_id, series, token)
}

fn parse_cookie_value(value: &str) -> Option<(Uuid, Uuid, String)> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let user_id = Uuid::parse_str(parts[0]).ok()?;
    let series = Uuid::parse_str(parts[1]).ok()?;
    let token = parts[2].to_string();
    Some((user_id, series, token))
}

async fn register(
    State(state): State<AppState>,
    Json(payload): Json<RegisterRequest>,
) -> Result<(StatusCode, Json<AuthResponse>), (StatusCode, Json<ErrorResponse>)> {
    if !state.registration_enabled {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Registration is currently disabled".to_string(),
            }),
        ));
    }

    if payload.username.len() < 3 || payload.username.len() > 255 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Username must be 3-255 characters".to_string(),
            }),
        ));
    }
    if payload.password.len() < 8 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Password must be at least 8 characters".to_string(),
            }),
        ));
    }

    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(payload.password.as_bytes(), &salt)
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Hashing failed".to_string(),
                }),
            )
        })?
        .to_string();

    let user_id = Uuid::new_v4();

    let result = sqlx::query("INSERT INTO users (id, username, password_hash) VALUES ($1, $2, $3)")
        .bind(user_id)
        .bind(&payload.username)
        .bind(&password_hash)
        .execute(&state.db)
        .await;

    match result {
        Ok(_) => Ok((
            StatusCode::CREATED,
            Json(AuthResponse {
                user_id,
                username: payload.username,
            }),
        )),
        Err(e)
            if e.as_database_error()
                .map(|x| x.is_unique_violation())
                .unwrap_or(false) =>
        {
            Err((
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: "Username taken".to_string(),
                }),
            ))
        }
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
            }),
        )),
    }
}

async fn login(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
    Json(payload): Json<LoginRequest>,
) -> Result<(CookieJar, Json<AuthResponse>), (StatusCode, Json<ErrorResponse>)> {
    let user = sqlx::query_as::<_, (Uuid, String, String)>(
        "SELECT id, username, password_hash FROM users WHERE username = $1",
    )
    .bind(&payload.username)
    .fetch_optional(&state.db)
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "DB Error".to_string(),
            }),
        )
    })?;

    let Some((user_id, username, password_hash)) = user else {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid credentials".to_string(),
            }),
        ));
    };

    let parsed_hash = PasswordHash::new(&password_hash).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Hash Error".to_string(),
            }),
        )
    })?;

    if Argon2::default()
        .verify_password(payload.password.as_bytes(), &parsed_hash)
        .is_err()
    {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid credentials".to_string(),
            }),
        ));
    }

    session.insert("user_id", user_id).await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Session creation failed".to_string(),
            }),
        )
    })?;

    let mut response_jar = jar;
    if payload.remember_me {
        let series = Uuid::new_v4();
        let (token, token_hash) = generate_token_pair();
        let expires_at = OffsetDateTime::now_utc() + Duration::days(REMEMBER_DURATION_DAYS);

        sqlx::query("INSERT INTO remember_tokens (series, user_id, token_hash, expires_at) VALUES ($1, $2, $3, $4)")
            .bind(series).bind(user_id).bind(token_hash).bind(expires_at)
            .execute(&state.db).await
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Failed to set remember token".to_string() })))?;

        let value = build_cookie_value(user_id, series, &token);
        let mut cookie = Cookie::new(REMEMBER_COOKIE_NAME, value);
        cookie.set_http_only(true);

        cookie.set_secure(state.is_production);

        cookie.set_same_site(SameSite::Lax);
        cookie.set_path("/");
        cookie.set_max_age(Duration::days(REMEMBER_DURATION_DAYS));

        response_jar = response_jar.add(cookie);
    }

    Ok((response_jar, Json(AuthResponse { user_id, username })))
}

async fn logout(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
) -> impl IntoResponse {
    let _ = session.delete().await;

    let mut response_jar = jar;

    if let Some(cookie) = response_jar.get(REMEMBER_COOKIE_NAME) {
        if let Some((_, series, _)) = parse_cookie_value(cookie.value()) {
            let _ = sqlx::query("DELETE FROM remember_tokens WHERE series = $1")
                .bind(series)
                .execute(&state.db)
                .await;
        }
        response_jar = response_jar.remove(Cookie::from(REMEMBER_COOKIE_NAME));
    }

    (response_jar, StatusCode::NO_CONTENT)
}

async fn current_user(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
) -> Result<(CookieJar, Json<AuthResponse>), (StatusCode, Json<ErrorResponse>)> {
    let mut response_jar = jar.clone();

    let mut maybe_user_id: Option<Uuid> = session.get("user_id").await.ok().flatten();

    if maybe_user_id.is_none() {
        if let Some(cookie) = response_jar.get(REMEMBER_COOKIE_NAME) {
            if let Some((user_id, series, token)) = parse_cookie_value(cookie.value()) {
                if let Ok(Some((db_hash, db_uid, expires))) = sqlx::query_as::<
                    _,
                    (String, Uuid, OffsetDateTime),
                >(
                    "SELECT token_hash, user_id, expires_at FROM remember_tokens WHERE series = $1",
                )
                .bind(series)
                .fetch_optional(&state.db)
                .await
                {
                    let mut hasher = Sha256::new();
                    hasher.update(token.as_bytes());
                    let incoming_hash = BASE64.encode(hasher.finalize());

                    if db_uid == user_id
                        && incoming_hash == db_hash
                        && OffsetDateTime::now_utc() < expires
                    {
                        let (new_token, new_hash) = generate_token_pair();

                        sqlx::query("UPDATE remember_tokens SET token_hash = $1, last_used_at = NOW() WHERE series = $2")
                            .bind(new_hash).bind(series).execute(&state.db).await.ok();

                        session.insert("user_id", user_id).await.ok();
                        maybe_user_id = Some(user_id);

                        let value = build_cookie_value(user_id, series, &new_token);
                        let mut c = Cookie::new(REMEMBER_COOKIE_NAME, value);
                        c.set_http_only(true);

                        c.set_secure(state.is_production);

                        c.set_path("/");
                        c.set_max_age(Duration::days(REMEMBER_DURATION_DAYS));
                        response_jar = response_jar.add(c);
                    }
                }
            }
        }
    }

    let user_id = maybe_user_id.ok_or((
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "Not authenticated".to_string(),
        }),
    ))?;

    let user = sqlx::query_as::<_, (Uuid, String)>("SELECT id, username FROM users WHERE id = $1")
        .bind(user_id)
        .fetch_optional(&state.db)
        .await
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "DB Error".to_string(),
                }),
            )
        })?
        .ok_or((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "User not found".to_string(),
            }),
        ))?;

    Ok((
        response_jar,
        Json(AuthResponse {
            user_id: user.0,
            username: user.1,
        }),
    ))
}
