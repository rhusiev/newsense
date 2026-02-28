use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub password: String,
}

#[derive(Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
    #[serde(default)]
    pub remember_me: bool,
}

#[derive(Serialize)]
pub struct AuthResponse {
    pub user_id: Uuid,
    pub username: String,
    pub role: i32,
}

#[derive(Deserialize)]
pub struct CreateCodesRequest {
    pub count: usize,
}

#[derive(Deserialize)]
pub struct CreateCodeRequest {
    pub code: String,
}

#[derive(Serialize)]
pub struct CodeResponse {
    pub code: String,
}


#[derive(Serialize)]
pub struct CodeCountResponse {
    pub count: i64,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}


