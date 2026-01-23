use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD as BASE64};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use rand::RngCore;

pub const REMEMBER_COOKIE_NAME: &str = "remember_me";
pub const REMEMBER_DURATION_DAYS: i64 = 30;
pub const SESSION_COOKIE_NAME: &str = "newsense_session";

pub fn generate_token_pair() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let token = BASE64.encode(bytes);

    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let hash = BASE64.encode(hasher.finalize());

    (token, hash)
}

pub fn build_cookie_value(user_id: Uuid, series: Uuid, token: &str) -> String {
    format!("{}:{}:{}", user_id, series, token)
}

pub fn parse_cookie_value(value: &str) -> Option<(Uuid, Uuid, String)> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let user_id = Uuid::parse_str(parts[0]).ok()?;
    let series = Uuid::parse_str(parts[1]).ok()?;
    let token = parts[2].to_string();
    Some((user_id, series, token))
}
