//! Per-user reading settings. Defaults live in the `user_settings` column
//! defaults: the first read creates the row.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use rand::{RngCore, rng};
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
    /// 'newest' or 'oldest', the global reading order of list views.
    list_order: String,
    reader_layout: String,
    llm_summaries: bool,
    llm_allowed: bool,
    llm_daily_limit: i32,
    llm_used_today: i64,
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
    list_order: Option<String>,
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

fn bad_request(error: &str) -> ApiError {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            error: error.into(),
        }),
    )
}

fn server_error(error: String) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse { error }),
    )
}

const LLM_KEY_BYTES: usize = 32;
const LLM_NONCE_BYTES: usize = 12;

fn encryption_key() -> Result<[u8; LLM_KEY_BYTES], ApiError> {
    let raw = std::env::var("LLM_KEY_ENCRYPTION_KEY")
        .map_err(|_| server_error("LLM_KEY_ENCRYPTION_KEY is not set".into()))?;
    let bytes = BASE64
        .decode(raw)
        .map_err(|e| server_error(format!("LLM_KEY_ENCRYPTION_KEY is not base64: {e}")))?;
    bytes.try_into().map_err(|_| {
        server_error(format!(
            "LLM_KEY_ENCRYPTION_KEY must be {LLM_KEY_BYTES} bytes"
        ))
    })
}

fn encrypt_with(
    key: &[u8; LLM_KEY_BYTES],
    plain: &str,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| e.to_string())?;
    let mut nonce = [0u8; LLM_NONCE_BYTES];
    rng().fill_bytes(&mut nonce);
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce), plain.as_bytes())
        .map_err(|e| e.to_string())?;
    Ok((ciphertext, nonce.to_vec()))
}

#[cfg(test)]
fn decrypt_with(
    key: &[u8; LLM_KEY_BYTES],
    ciphertext: &[u8],
    nonce: &[u8],
) -> Result<String, String> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| e.to_string())?;
    let plain = cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|e| e.to_string())?;
    String::from_utf8(plain).map_err(|e| e.to_string())
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
        r#"SELECT all_articles_filter, score_threshold, top_per_day, use_clusters,
                  list_order, reader_layout, llm_summaries, llm_allowed, llm_daily_limit
           FROM user_settings WHERE user_id = $1"#,
        user_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(db_error)?;
    let used_today = sqlx::query!(
        r#"SELECT COUNT(*)::bigint AS used FROM llm_calls
           WHERE user_id = $1 AND created_at >= date_trunc('day', now())"#,
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
        list_order: s.list_order,
        reader_layout: s.reader_layout,
        llm_summaries: s.llm_summaries,
        llm_allowed: s.llm_allowed,
        llm_daily_limit: s.llm_daily_limit,
        llm_used_today: used_today.used.unwrap_or(0),
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
            list_order = COALESCE($8, list_order),
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
        p.llm_summaries,
        p.list_order
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

#[derive(Serialize)]
pub struct LlmKeyResponse {
    pub configured: bool,
    pub base_url: Option<String>,
    pub model: Option<String>,
}

#[derive(Deserialize)]
pub struct SetLlmKeyRequest {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

/// Whether the user has their own model, and which one. Never the key.
pub async fn get_llm_key(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
) -> Result<Json<LlmKeyResponse>, ApiError> {
    let row = sqlx::query!(
        "SELECT base_url, model FROM user_llm_keys WHERE user_id = $1",
        user_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(db_error)?;

    Ok(Json(match row {
        Some(row) => LlmKeyResponse {
            configured: true,
            base_url: Some(row.base_url),
            model: Some(row.model),
        },
        None => LlmKeyResponse {
            configured: false,
            base_url: None,
            model: None,
        },
    }))
}

/// Stores the user's own endpoint, model and key. All three are required, so
/// changing the model also sends the key again.
pub async fn set_llm_key(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Json(payload): Json<SetLlmKeyRequest>,
) -> Result<Json<LlmKeyResponse>, ApiError> {
    let base_url = payload.base_url.trim();
    let model = payload.model.trim();
    let api_key = payload.api_key.trim();
    if !(base_url.starts_with("http://") || base_url.starts_with("https://"))
        || model.is_empty()
        || api_key.is_empty()
    {
        return Err(bad_request(
            "base_url must be http or https, and model and api_key must not be empty",
        ));
    }

    let key = encryption_key()?;
    let (ciphertext, nonce) =
        encrypt_with(&key, api_key).map_err(|e| server_error(format!("could not encrypt key: {e}")))?;
    sqlx::query!(
        r#"
        INSERT INTO user_llm_keys
            (user_id, base_url, model, key_ciphertext, key_nonce)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (user_id) DO UPDATE SET
            base_url = EXCLUDED.base_url,
            model = EXCLUDED.model,
            key_ciphertext = EXCLUDED.key_ciphertext,
            key_nonce = EXCLUDED.key_nonce,
            updated_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        base_url,
        model,
        ciphertext,
        nonce
    )
    .execute(&state.db)
    .await
    .map_err(db_error)?;

    Ok(Json(LlmKeyResponse {
        configured: true,
        base_url: Some(base_url.to_string()),
        model: Some(model.to_string()),
    }))
}

pub async fn delete_llm_key(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
) -> Result<StatusCode, ApiError> {
    sqlx::query!("DELETE FROM user_llm_keys WHERE user_id = $1", user_id)
        .execute(&state.db)
        .await
        .map_err(db_error)?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::{decrypt_with, encrypt_with};

    #[test]
    fn key_round_trip() {
        let key = [7u8; 32];
        let (ciphertext, nonce) = encrypt_with(&key, "secret-key").unwrap();
        assert_eq!(
            decrypt_with(&key, &ciphertext, &nonce).unwrap(),
            "secret-key"
        );
        assert_ne!(ciphertext, b"secret-key");
    }
}
