//! Per-user reading settings. Defaults live in the `user_settings` column
//! defaults: the first read creates the row.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::{AppState, AuthUser, ErrorResponse};

type ApiError = (StatusCode, Json<ErrorResponse>);

const CHECK_VIOLATION: &str = "23514";

#[derive(Serialize)]
pub struct SettingsResponse {
    all_articles_filter: String,
    score_threshold: f32,
    top_per_day: i32,
    use_clusters: bool,
    reader_layout: String,
    llm_summaries: bool,
    llm_allowed: bool,
    llm_daily_limit: i32,
    /// Each subscribed feed's own filter, by feed id. Feeds absent here are 'off'.
    feed_filters: HashMap<Uuid, FeedFilter>,
}

#[derive(Serialize, Deserialize)]
pub struct FeedFilter {
    /// 'off', 'score' or 'top'.
    mode: String,
    score_threshold: Option<f32>,
    top_n: Option<i32>,
}

#[derive(Deserialize)]
pub struct UpdateSettingsRequest {
    all_articles_filter: Option<String>,
    score_threshold: Option<f32>,
    top_per_day: Option<i32>,
    use_clusters: Option<bool>,
    reader_layout: Option<String>,
    /// The user's own switch. The daily limit and the admin allowance are not
    /// writable here.
    llm_summaries: Option<bool>,
}

fn db_error(e: sqlx::Error) -> ApiError {
    let invalid = e
        .as_database_error()
        .and_then(|d| d.code())
        .is_some_and(|c| c == CHECK_VIOLATION);
    let status = if invalid {
        StatusCode::BAD_REQUEST
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (
        status,
        Json(ErrorResponse {
            error: e.to_string(),
        }),
    )
}

pub async fn get_settings(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
) -> Result<Json<SettingsResponse>, ApiError> {
    ensure_row(&state, user_id).await?;
    read_settings(&state, user_id).await.map(Json)
}

async fn ensure_row(state: &AppState, user_id: Uuid) -> Result<(), ApiError> {
    sqlx::query!(
        "INSERT INTO user_settings (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
        user_id
    )
    .execute(&state.db)
    .await
    .map_err(db_error)?;
    Ok(())
}

async fn read_settings(state: &AppState, user_id: Uuid) -> Result<SettingsResponse, ApiError> {
    let s = sqlx::query!(
        r#"SELECT all_articles_filter, score_threshold, top_per_day, use_clusters, reader_layout,
                  llm_summaries, llm_allowed, llm_daily_limit
           FROM user_settings WHERE user_id = $1"#,
        user_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(db_error)?;
    let feeds = sqlx::query!(
        r#"SELECT feed_id AS "feed_id!", filter_mode, score_threshold, top_n
           FROM feed_subscriptions WHERE user_id = $1 AND filter_mode <> 'off'"#,
        user_id
    )
    .fetch_all(&state.db)
    .await
    .map_err(db_error)?;
    Ok(SettingsResponse {
        all_articles_filter: s.all_articles_filter,
        score_threshold: s.score_threshold,
        top_per_day: s.top_per_day,
        use_clusters: s.use_clusters,
        reader_layout: s.reader_layout,
        llm_summaries: s.llm_summaries,
        llm_allowed: s.llm_allowed,
        llm_daily_limit: s.llm_daily_limit,
        feed_filters: feeds
            .into_iter()
            .map(|f| {
                let filter = FeedFilter {
                    mode: f.filter_mode,
                    score_threshold: f.score_threshold,
                    top_n: f.top_n,
                };
                (f.feed_id, filter)
            })
            .collect(),
    })
}

/// Changes only the fields sent. Values outside the allowed set are a 400.
pub async fn update_settings(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Json(p): Json<UpdateSettingsRequest>,
) -> Result<Json<SettingsResponse>, ApiError> {
    ensure_row(&state, user_id).await?;
    sqlx::query!(
        r#"
        UPDATE user_settings SET
            all_articles_filter = COALESCE($2, all_articles_filter),
            score_threshold = COALESCE($3, score_threshold),
            top_per_day = COALESCE($4, top_per_day),
            use_clusters = COALESCE($5, use_clusters),
            reader_layout = COALESCE($6, reader_layout),
            llm_summaries = COALESCE($7, llm_summaries),
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = $1
        "#,
        user_id,
        p.all_articles_filter,
        p.score_threshold,
        p.top_per_day,
        p.use_clusters,
        p.reader_layout,
        p.llm_summaries
    )
    .execute(&state.db)
    .await
    .map_err(db_error)?;
    read_settings(&state, user_id).await.map(Json)
}

/// Sets the feed's own filter. Only for a feed the user subscribes to (owners
/// are subscribed to their own feeds). A mode without its value is a 400.
pub async fn set_feed_filter(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(feed_id): Path<Uuid>,
    Json(p): Json<FeedFilter>,
) -> Result<StatusCode, ApiError> {
    let updated = sqlx::query!(
        r#"UPDATE feed_subscriptions SET filter_mode = $3, score_threshold = $4, top_n = $5
           WHERE user_id = $1 AND feed_id = $2"#,
        user_id,
        feed_id,
        p.mode,
        p.score_threshold,
        p.top_n
    )
    .execute(&state.db)
    .await
    .map_err(db_error)?;
    if updated.rows_affected() == 0 {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Not subscribed to this feed".into(),
            }),
        ));
    }
    Ok(StatusCode::NO_CONTENT)
}
