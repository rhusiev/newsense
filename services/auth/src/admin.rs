use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD as BASE64};
use rand::RngCore;

use crate::{
    extractors::AdminUser,
    models::{CodeCountResponse, CodeResponse, CreateCodeRequest, CreateCodesRequest, ErrorResponse},
    state::AppState,
};

fn generate_random_string(len: usize) -> String {
    let mut bytes = vec![0u8; len];
    rand::rng().fill_bytes(&mut bytes);
    BASE64.encode(bytes)
}

pub async fn create_codes(
    State(state): State<AppState>,
    _admin: AdminUser,
    Json(payload): Json<CreateCodesRequest>,
) -> Result<Json<Vec<CodeResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let mut codes = Vec::new();

    for _ in 0..payload.count {
        let code = generate_random_string(16);
        
        sqlx::query("INSERT INTO access_codes (code) VALUES ($1)")
            .bind(&code)
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

        codes.push(CodeResponse {
            code,
        });
    }

    Ok(Json(codes))
}

pub async fn create_named_code(
    State(state): State<AppState>,
    _admin: AdminUser,
    Json(payload): Json<CreateCodeRequest>,
) -> Result<Json<CodeResponse>, (StatusCode, Json<ErrorResponse>)> {
    sqlx::query("INSERT INTO access_codes (code) VALUES ($1)")
        .bind(&payload.code)
        .execute(&state.db)
        .await
        .map_err(|e| {
            if e.as_database_error().map(|x| x.is_unique_violation()).unwrap_or(false) {
                 (
                    StatusCode::CONFLICT,
                    Json(ErrorResponse {
                        error: "Code already exists".to_string(),
                    }),
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: e.to_string(),
                    }),
                )
            }
        })?;

    Ok(Json(CodeResponse {
        code: payload.code,
    }))
}

pub async fn list_codes(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> Result<Json<Vec<CodeResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let codes = sqlx::query!("SELECT code FROM access_codes ORDER BY created_at DESC")
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

    let response = codes
        .into_iter()
        .map(|r| CodeResponse {
            code: r.code,
        })
        .collect();

    Ok(Json(response))
}

pub async fn count_codes(
    State(state): State<AppState>,
    _admin: AdminUser,
) -> Result<Json<CodeCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    let count = sqlx::query!("SELECT COUNT(*) as count FROM access_codes")
        .fetch_one(&state.db)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?
        .count
        .unwrap_or(0);

    Ok(Json(CodeCountResponse { count }))
}

pub async fn delete_code(
    State(state): State<AppState>,
    _admin: AdminUser,
    Path(code): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    let result = sqlx::query!("DELETE FROM access_codes WHERE code = $1", code)
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
                error: "Code not found".to_string(),
            }),
        ));
    }

    Ok(StatusCode::NO_CONTENT)
}

pub async fn admin_check(_admin: AdminUser) -> StatusCode {
    StatusCode::OK
}

