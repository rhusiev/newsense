use axum::{
    Json, RequestPartsExt, Router,
    extract::{FromRequestParts, Path, Query, State},
    http::{StatusCode, Method, HeaderValue, request::Parts, header},
    routing::{delete, get, post, put},
};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::net::SocketAddr;
use tower_sessions::{Expiry, Session, SessionManagerLayer, cookie::time::Duration};
use tower_sessions_sqlx_store::PostgresStore;
use uuid::Uuid;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
struct AppState {
    db: PgPool,
    http_client: reqwest::Client,
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

#[derive(Serialize)]
struct SubscriberCountResponse {
    feed_id: Uuid,
    subscriber_count: i64,
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
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
        "Users Service running in {} mode (Secure Cookies: {})",
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

    let app_state = AppState {
        db: pool,
        http_client: reqwest::Client::new(),
    };

    let web_url = std::env::var("WEB_URL")
        .unwrap_or_else(|_| "http://localhost:5173".to_string());

    let allowed_origins = [
        web_url.parse::<HeaderValue>().unwrap(),
        "http://127.0.0.1:5173".parse::<HeaderValue>().unwrap(),
        "http://localhost:5173".parse::<HeaderValue>().unwrap(),
    ];

    let cors = CorsLayer::new()
        .allow_origin(allowed_origins) 
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::HeaderName::from_static("x-csrf-token")])
        .allow_credentials(true);

    let app = Router::new()
        .route("/feeds", post(add_feed))
        .route("/feeds/{id}", put(update_feed))
        .route("/feeds/{id}", get(get_feed))
        .route("/feeds/{id}", delete(delete_feed))
        .route("/feeds/{id}/subscription", post(subscribe_feed))
        .route("/feeds/{id}/subscription", delete(unsubscribe_feed))
        .route(
            "/feeds/{id}/subscribers/count",
            get(get_feed_subscriber_count),
        )
        .route("/feeds/owned", get(list_owned_feeds))
        .route("/feeds/subscribed", get(list_subscribed_feeds))
        .route("/feeds/search", get(search_public_feeds))
        .layer(session_layer)
        .layer(cors)
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001")
        .await
        .expect("Failed to bind");

    println!("Users Service running on http://0.0.0.0:3001");

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
    let source = sqlx::query!(
        r#"
        INSERT INTO sources (url)
        VALUES ($1)
        ON CONFLICT (url) DO UPDATE SET url = EXCLUDED.url
        RETURNING id, url
        "#,
        payload.url
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

    let existing_feed = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT f.id, f.owner_id, s.url, f.title, f.description, f.is_public, f.created_at
        FROM feeds f
        JOIN sources s ON f.source_id = s.id
        WHERE f.owner_id = $1 AND f.source_id = $2
        "#,
        user_id,
        source.id
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

    if let Some(feed) = existing_feed {
        return Ok((StatusCode::OK, Json(feed)));
    }

    let new_feed = sqlx::query_as!(
        FeedResponse,
        r#"
        INSERT INTO feeds (owner_id, source_id, title, description, is_public)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id, owner_id, $6::TEXT as "url!", title, description, is_public, created_at
        "#,
        user_id,
        source.id,
        payload.title,
        payload.description,
        payload.is_public.unwrap_or(false),
        source.url
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

    sqlx::query!(
        "INSERT INTO feed_subscriptions (user_id, feed_id) VALUES ($1, $2)",
        user_id,
        new_feed.id
    )
    .execute(&state.db)
    .await
    .ok();

    let client = state.http_client.clone();
    let feed_id = new_feed.id;
    tokio::spawn(async move {
        let url = format!("http://fetcher:3003/feeds/{}/refresh", feed_id);
        client.post(&url).send().await.ok();
    });

    Ok((StatusCode::CREATED, Json(new_feed)))
}

async fn list_subscribed_feeds(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
) -> Result<Json<Vec<FeedResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let feeds = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT f.id, f.owner_id, s.url, f.title, f.description, f.is_public, f.created_at
        FROM feeds f
        JOIN sources s ON f.source_id = s.id
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

async fn list_owned_feeds(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
) -> Result<Json<Vec<FeedResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let feeds = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT f.id, f.owner_id, s.url, f.title, f.description, f.is_public, f.created_at
        FROM feeds f
        JOIN sources s ON f.source_id = s.id
        WHERE f.owner_id = $1
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

async fn get_feed_subscriber_count(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(feed_id): Path<Uuid>,
) -> Result<Json<SubscriberCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    let feed = sqlx::query!(
        "SELECT owner_id, is_public FROM feeds WHERE id = $1",
        feed_id
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

    let feed_record = feed.ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "Feed not found".to_string(),
        }),
    ))?;

    if feed_record.owner_id != Some(user_id) && feed_record.is_public != Some(true) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".to_string(),
            }),
        ));
    }

    let count = sqlx::query!(
        "SELECT COUNT(*) as count FROM feed_subscriptions WHERE feed_id = $1",
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

    Ok(Json(SubscriberCountResponse {
        feed_id,
        subscriber_count: count.count.unwrap_or(0),
    }))
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

    match feed {
        Some(f) if f.owner_id == Some(user_id) => {}
        Some(_) => {
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: "Not owner".into(),
                }),
            ));
        }
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: "Not found".into(),
                }),
            ));
        }
    };

    let updated_feed = sqlx::query_as!(
        FeedResponse,
        r#"
        UPDATE feeds
        SET
            title = COALESCE($1, title),
            description = COALESCE($2, description),
            is_public = COALESCE($3, is_public)
        FROM sources s
        WHERE feeds.id = $4 AND feeds.source_id = s.id
        RETURNING feeds.id, feeds.owner_id, s.url, feeds.title, feeds.description, feeds.is_public, feeds.created_at
        "#,
        payload.title,
        payload.description,
        payload.is_public,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    Ok(Json(updated_feed))
}

async fn get_feed(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(feed_id): Path<Uuid>,
) -> Result<Json<FeedResponse>, (StatusCode, Json<ErrorResponse>)> {
    let feed = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT f.id, f.owner_id, s.url, f.title, f.description, f.is_public, f.created_at
        FROM feeds f
        JOIN sources s ON f.source_id = s.id
        WHERE f.id = $1
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

    let feed_record = feed.ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "Feed not found".into(),
        }),
    ))?;

    if feed_record.owner_id != Some(user_id) && feed_record.is_public != Some(true) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
    }

    Ok(Json(feed_record))
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
    let feed = sqlx::query!(
        r#"
        SELECT f.owner_id, f.is_public
        FROM feeds f
        WHERE f.id = $1
        "#,
        feed_id
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

    let feed_record = feed.ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "Not found".into(),
        }),
    ))?;

    if feed_record.owner_id != Some(user_id) && feed_record.is_public != Some(true) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
    }

    let sub = sqlx::query_as!(
        FeedSubscriptionResponse,
        r#"
        INSERT INTO feed_subscriptions (user_id, feed_id)
        VALUES ($1, $2)
        ON CONFLICT (user_id, feed_id) DO UPDATE SET created_at = feed_subscriptions.created_at
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

    Ok((StatusCode::CREATED, Json(sub)))
}

async fn unsubscribe_feed(
    State(state): State<AppState>,
    AuthUser(user_id): AuthUser,
    Path(feed_id): Path<Uuid>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let feed = sqlx::query!(
        "SELECT owner_id, is_public FROM feeds WHERE id = $1",
        feed_id
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
                    error: "Feed not found".into(),
                }),
            ));
        }
    };

    if feed_record.owner_id != Some(user_id) && feed_record.is_public != Some(true) {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
    }

    let result = sqlx::query!(
        "DELETE FROM feed_subscriptions WHERE user_id = $1 AND feed_id = $2",
        user_id,
        feed_id
    )
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
                error: "Subscription not found".into(),
            }),
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

async fn search_public_feeds(
    State(state): State<AppState>,
    _user: AuthUser,
    Query(params): Query<SearchParams>,
) -> Result<Json<Vec<FeedResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let search_pattern = format!("%{}%", params.q);

    let feeds = sqlx::query_as!(
        FeedResponse,
        r#"
        SELECT f.id, f.owner_id, s.url, f.title, f.description, f.is_public, f.created_at
        FROM feeds f
        JOIN sources s ON f.source_id = s.id
        WHERE f.is_public = true
          AND (f.title ILIKE $1 OR f.description ILIKE $1 OR s.url ILIKE $1)
        ORDER BY f.created_at DESC
        LIMIT 50
        "#,
        search_pattern
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
