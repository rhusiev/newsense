use axum::{
    Json, RequestPartsExt, Router,
    extract::{FromRequestParts, Path, State},
    http::{StatusCode, request::Parts},
    routing::{delete, get, post, put},
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
struct AddFeedRequest {
    url: String,
    title: Option<String>,
    description: Option<String>,
    is_public: Option<bool>,
}

#[derive(Deserialize)]
struct UpdateFeedRequest {
    title: Option<String>,
    description: Option<String>,
    is_public: Option<bool>,
}

#[derive(Serialize)]
struct FeedResponse {
    id: Uuid,
    owner_id: Option<Uuid>,
    url: String,
    title: Option<String>,
    description: Option<String>,
    is_public: Option<bool>,
    #[serde(with = "time::serde::iso8601::option")]
    created_at: Option<time::OffsetDateTime>,
}

#[derive(Serialize)]
struct FeedSubscriptionResponse {
    user_id: Uuid,
    feed_id: Uuid,
    #[serde(with = "time::serde::iso8601::option")]
    created_at: Option<time::OffsetDateTime>,
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

    let session_layer = SessionManagerLayer::new(session_store)
        .with_secure(true)
        .with_same_site(tower_sessions::cookie::SameSite::Strict)
        .with_expiry(Expiry::OnInactivity(Duration::new(3600, 0)));

    let app_state = AppState { db: pool };

    let app = Router::new()
        .route("/feeds", post(add_feed))
        .route("/feeds", get(list_feeds))
        .route("/feeds/{id}", put(update_feed))
        .route("/feeds/{id}", delete(delete_feed))
        .route("/feeds/{id}/subscription", post(subscribe_feed))
        .layer(session_layer)
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001")
        .await
        .expect("Failed to bind");

    println!("Users/Feeds Service running on http://0.0.0.0:3001");

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .expect("Server failed");
}

async fn add_feed(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Json(payload): Json<AddFeedRequest>,
) -> Result<(StatusCode, Json<FeedResponse>), (StatusCode, Json<ErrorResponse>)> {
    let existing_feed = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT id, owner_id, url, title, description, is_public, created_at
        FROM feeds WHERE url = $1 AND owner_id = $2
        "#,
        payload.url,
        user_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let feed = match existing_feed {
        Some(f) => f,
        None => {
            let new_feed = sqlx::query_as!(
                FeedResponse,
                r#"
                INSERT INTO feeds (owner_id, url, title, description, is_public)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, owner_id, url, title, description, is_public, created_at
                "#,
                user_id,
                payload.url,
                payload.title,
                payload.description,
                payload.is_public.unwrap_or(false)
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

            new_feed
        }
    };

    Ok((StatusCode::CREATED, Json(feed)))
}

async fn list_feeds(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
) -> Result<Json<Vec<FeedResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let feeds = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT f.id, f.owner_id, f.url, f.title, f.description, f.is_public, f.created_at
        FROM feeds f
        JOIN feed_subscriptions fs ON f.id = fs.feed_id
        WHERE fs.user_id = $1
        ORDER BY f.created_at DESC
        "#,
        user_id
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(feeds))
}

async fn update_feed(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(feed_id): Path<Uuid>,
    Json(payload): Json<UpdateFeedRequest>,
) -> Result<Json<FeedResponse>, (StatusCode, Json<ErrorResponse>)> {
    let feed = sqlx::query!("SELECT owner_id FROM feeds WHERE id = $1", feed_id)
        .fetch_optional(&state.db)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    let feed_record = match feed {
        Some(f) => f,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Feed not found".to_string(),
                }),
            ));
        }
    };

    if feed_record.owner_id != Some(user_id) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "You are not the owner of this feed".to_string(),
            }),
        ));
    }

    let updated_feed = sqlx::query_as!(
        FeedResponse,
        r#"
        UPDATE feeds
        SET
            title = COALESCE($1, title),
            description = COALESCE($2, description),
            is_public = COALESCE($3, is_public)
        WHERE id = $4
        RETURNING id, owner_id, url, title, description, is_public, created_at
        "#,
        payload.title,
        payload.description,
        payload.is_public,
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

    Ok(Json(updated_feed))
}

async fn delete_feed(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(feed_id): Path<Uuid>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let feed = sqlx::query!("SELECT owner_id FROM feeds WHERE id = $1", feed_id)
        .fetch_optional(&state.db)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;
    let feed_record = match feed {
        Some(f) => f,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Feed not found".to_string(),
                }),
            ));
        }
    };
    if feed_record.owner_id != Some(user_id) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "You are not the owner of this feed".to_string(),
            }),
        ));
    }
    let result = sqlx::query!("DELETE FROM feeds WHERE id = $1", feed_id)
        .execute(&state.db)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

    if result.rows_affected() == 0 {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Feed not found".to_string(),
            }),
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

async fn subscribe_feed(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(feed_id): Path<Uuid>,
) -> Result<(StatusCode, Json<FeedSubscriptionResponse>), (StatusCode, Json<ErrorResponse>)> {
    let feed = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT id, owner_id, url, title, description, is_public, created_at
        FROM feeds WHERE id = $1
        "#,
        feed_id,
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let feed_record = match feed {
        Some(f) => f,
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Feed not found".to_string(),
                }),
            ));
        }
    };

    if feed_record.owner_id != Some(user_id) && feed_record.is_public != Some(true) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "You can't access this feed".to_string(),
            }),
        ));
    }

    let existing_subscription = sqlx::query_as!(
        FeedSubscriptionResponse,
        r#"
        SELECT user_id, feed_id, created_at
        FROM feed_subscriptions WHERE user_id = $1 AND feed_id = $2
        "#,
        user_id,
        feed_id,
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let subscription = match existing_subscription {
        Some(s) => s,
        None => {
            let new_subscription = sqlx::query_as!(
                FeedSubscriptionResponse,
                r#"
                INSERT INTO feed_subscriptions (user_id, feed_id)
                VALUES ($1, $2)
                RETURNING user_id, feed_id, created_at
                "#,
                user_id,
                feed_id,
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

            new_subscription
        }
    };

    Ok((StatusCode::CREATED, Json(subscription)))
}
