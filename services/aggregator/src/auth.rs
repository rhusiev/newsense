use axum::{
    Json,
    extract::FromRequestParts,
    http::{StatusCode, request::Parts},
    RequestPartsExt,
};
use tower_sessions::Session;
use uuid::Uuid;

use crate::models::ErrorResponse;

pub struct AuthUser {
    pub id: Uuid,
    pub role: i32,
}

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

        let role = session.get::<i32>("role").await.unwrap_or(Some(0));

        match user_id {
            Some(id) => Ok(AuthUser {
                id,
                role: role.unwrap_or(0),
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
