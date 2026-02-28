use axum::{
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
    Json, RequestPartsExt,
};
use tower_sessions::Session;
use uuid::Uuid;

use crate::models::ErrorResponse;
use crate::state::AppState;

pub struct AuthUser {
    pub id: Uuid,
}

impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<ErrorResponse>);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let session = parts
            .extract::<Session>()
            .await
            .map_err(|_| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Session extraction failed".to_string(),
                }),
            ))?;

        let user_id = session.get::<Uuid>("user_id").await.map_err(|_| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Session read error".to_string(),
            }),
        ))?;

        match user_id {
            Some(id) => Ok(AuthUser {
                id,
            }),
            None => Err((
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Unauthorized".to_string(),
                }),
            )),
        }
    }
}

pub struct AdminUser(pub Uuid);

impl FromRequestParts<AppState> for AdminUser {
    type Rejection = (StatusCode, Json<ErrorResponse>);

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, Self::Rejection> {
        let user = AuthUser::from_request_parts(parts, state).await?;

        let user_record = sqlx::query!(
            "SELECT role FROM users WHERE id = $1",
            user.id
        )
        .fetch_optional(&state.db)
        .await
        .map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        ))?;

        let current_role = match user_record {
            Some(r) => r.role.unwrap_or(0),
            None => {
                 return Err((
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: "User not found".to_string(),
                    }),
                ));
            }
        };

        if current_role == 1 {
            Ok(AdminUser(user.id))
        } else {
            Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: "Admin access required".to_string(),
                }),
            ))
        }
    }
}
