//! Serves the readable text the fetcher kept for an item. Only the owner of a
//! private feed with full text on gets it, and never while any public feed
//! uses the source.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use serde::Serialize;
use uuid::Uuid;

use crate::{auth::AuthUser, models::ErrorResponse, state::AppState};

#[derive(Serialize)]
pub struct FullTextResponse {
    /// Sanitized article HTML without images.
    pub html: String,
    #[serde(with = "time::serde::iso8601::option")]
    pub fetched_at: Option<time::OffsetDateTime>,
}

pub async fn get_full_text(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(item_id): Path<Uuid>,
) -> Result<Json<FullTextResponse>, (StatusCode, Json<ErrorResponse>)> {
    let row = sqlx::query_as!(
        FullTextResponse,
        r#"
        SELECT t.html AS "html!", t.fetched_at
        FROM item_full_texts t
        INNER JOIN items i ON i.id = t.item_id
        WHERE t.item_id = $2
          AND t.html IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM feeds f
              WHERE f.source_id = i.source_id AND f.owner_id = $1 AND f.full_text
          )
          AND NOT EXISTS (
              SELECT 1 FROM feeds f WHERE f.source_id = i.source_id AND f.is_public
          )
        "#,
        user_id,
        item_id
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

    row.map(Json).ok_or((
        StatusCode::NOT_FOUND,
        Json(ErrorResponse {
            error: "No full text for this item".to_string(),
        }),
    ))
}
