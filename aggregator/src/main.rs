use axum::{
    Json, RequestPartsExt, Router,
    extract::{FromRequestParts, Path, Query, State},
    http::{StatusCode, request::Parts},
    routing::{get, put},
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::net::SocketAddr;
use tower_sessions::{Expiry, Session, SessionManagerLayer, cookie::time::Duration};
use tower_sessions_sqlx_store::PostgresStore;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    db: PgPool,
}

#[derive(Deserialize)]
struct GetItemsQuery {
    limit: Option<i64>,
    since: Option<time::OffsetDateTime>,
}

#[derive(Deserialize)]
struct UpdateItemStatusRequest {
    is_read: Option<bool>,
    liked: Option<f32>,
}

#[derive(Serialize)]
struct ItemResponse {
    id: Uuid,
    feed_id: Uuid,
    title: String,
    link: String,
    content: Option<String>,
    author: Option<String>,
    #[serde(with = "time::serde::iso8601::option")]
    published_at: Option<time::OffsetDateTime>,
    cluster_id: Option<Uuid>,
    #[serde(with = "time::serde::iso8601::option")]
    created_at: Option<time::OffsetDateTime>,
    is_read: Option<bool>,
    liked: Option<f32>,
}

#[derive(Serialize)]
struct ReadStatusResponse {
    user_id: Uuid,
    item_id: Uuid,
    is_read: bool,
    liked: f32,
    #[serde(with = "time::serde::iso8601::option")]
    marked_at: Option<time::OffsetDateTime>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

struct AuthUser(Uuid);

impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<ErrorResponse>);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let session = parts.extract::<Session>().await.map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Session extraction failed".to_string(),
                }),
            )
        })?;

        let user_id = session.get::<Uuid>("user_id").await.map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Session read error".to_string(),
                }),
            )
        })?;

        match user_id {
            Some(id) => Ok(AuthUser(id)),
            None => Err((
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Unauthorized".to_string(),
                }),
            )),
        }
    }
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

    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(is_production)
        .with_same_site(tower_sessions::cookie::SameSite::Lax)
        .with_expiry(Expiry::OnInactivity(Duration::new(3600, 0)));

    let app_state = AppState { db: pool };

    let app = Router::new()
        .route("/feeds/{feed_id}/items", get(get_feed_items))
        .route("/items", get(get_all_items))
        .route("/items/{item_id}/status", put(update_item_status))
        .layer(session_layer)
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

async fn get_feed_items(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(feed_id): Path<Uuid>,
    Query(params): Query<GetItemsQuery>,
) -> Result<Json<Vec<ItemResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied to this feed".to_string(),
            }),
        ));
    }

    let limit = params.limit.unwrap_or(50).min(1000);

    let items = if let Some(since) = params.since {
        sqlx::query_as!(
            ItemResponse,
            r#"
            SELECT
                i.id as "id!", i.feed_id as "feed_id!", i.title, i.link, i.content, i.author,
                i.published_at, i.cluster_id, i.created_at,
                ir.is_read as "is_read?",
                ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?"
            FROM items i
            LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
            WHERE i.feed_id = $2 AND i.published_at > $3
            ORDER BY i.published_at ASC
            LIMIT $4
            "#,
            user_id,
            feed_id,
            since,
            limit
        )
        .fetch_all(&state.db)
        .await
    } else {
        sqlx::query_as!(
            ItemResponse,
            r#"
            SELECT
                i.id as "id!", i.feed_id as "feed_id!", i.title, i.link, i.content, i.author,
                i.published_at, i.cluster_id, i.created_at,
                ir.is_read as "is_read?",
                ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?"
            FROM items i
            LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
            WHERE i.feed_id = $2
            ORDER BY i.published_at DESC
            LIMIT $3
            "#,
            user_id,
            feed_id,
            limit
        )
        .fetch_all(&state.db)
        .await
    }
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(items))
}

async fn get_all_items(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Query(params): Query<GetItemsQuery>,
) -> Result<Json<Vec<ItemResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let limit = params.limit.unwrap_or(50).min(1000);

    let items = sqlx::query_as!(
        ItemResponse,
        r#"
        SELECT
            i.id as "id!", i.feed_id as "feed_id!", i.title, i.link, i.content, i.author,
            i.published_at, i.cluster_id, i.created_at,
            ir.is_read as "is_read?",
            ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?"
        FROM items i
        INNER JOIN feeds f ON i.feed_id = f.id
        LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        WHERE f.owner_id = $1 OR fs.user_id = $1
        ORDER BY i.published_at DESC
        LIMIT $2
        "#,
        user_id,
        limit
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    Ok(Json(items))
}

async fn update_item_status(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(item_id): Path<Uuid>,
    Json(payload): Json<UpdateItemStatusRequest>,
) -> Result<Json<ReadStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    // 1. Access Check (Explicitly typed for safety)
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM items i
            INNER JOIN feeds f ON i.feed_id = f.id
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE i.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        item_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e: sqlx::Error| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    if !has_access.has_access {
        return Err((StatusCode::FORBIDDEN, Json(ErrorResponse { error: "Access denied".into() })));
    }

    // 2. The UPSERT with Explicit Type Casting
    // We cast $3 to BOOLEAN and $4 to REAL so SQLx knows how to bind payload fields
    let status = sqlx::query_as!(
        ReadStatusResponse,
        r#"
        INSERT INTO item_reads (user_id, item_id, is_read, liked)
        VALUES ($1, $2, COALESCE($3::BOOLEAN, true), COALESCE($4::REAL, 0.0))
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET 
            is_read = COALESCE($3::BOOLEAN, item_reads.is_read), 
            liked = COALESCE($4::REAL, item_reads.liked),
            marked_at = CURRENT_TIMESTAMP
        RETURNING 
            user_id, 
            item_id, 
            is_read, 
            ROUND(liked)::REAL as "liked!", 
            marked_at
        "#,
        user_id,
        item_id,
        payload.is_read, // Option<bool> maps to BOOLEAN
        payload.liked    // Option<f32> maps to REAL
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e: sqlx::Error| ( // Added explicit type to error to solve inference
        StatusCode::INTERNAL_SERVER_ERROR, 
        Json(ErrorResponse { error: e.to_string() })
    ))?;

    Ok(Json(status))
}
