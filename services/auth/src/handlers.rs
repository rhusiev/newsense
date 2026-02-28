use argon2::password_hash::rand_core::OsRng;
use argon2::{
    Argon2,
    password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
};
use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use axum_extra::extract::cookie::{Cookie, CookieJar, SameSite};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD as BASE64};
use sha2::{Digest, Sha256};
use sqlx::types::time::OffsetDateTime;
use tower_sessions::{Session, cookie::time::Duration};
use uuid::Uuid;

use crate::{
    models::{AuthResponse, ErrorResponse, LoginRequest, RegisterRequest},
    state::AppState,
    utils::{
        REMEMBER_COOKIE_NAME, REMEMBER_DURATION_DAYS, build_cookie_value, generate_token_pair,
        parse_cookie_value,
    },
};

pub async fn register(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
    Json(payload): Json<RegisterRequest>,
) -> Result<(StatusCode, CookieJar, Json<AuthResponse>), (StatusCode, Json<ErrorResponse>)> {
    if !state.registration_enabled {
        return Err((
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Registration is currently disabled".to_string(),
            }),
        ));
    }

    if payload.username.len() < 3 || payload.username.len() > 255 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Username must be 3-255 characters".to_string(),
            }),
        ));
    }
    if payload.password.len() < 8 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Password must be at least 8 characters".to_string(),
            }),
        ));
    }

    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(payload.password.as_bytes(), &salt)
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Hashing failed".to_string(),
                }),
            )
        })?
        .to_string();

    let user_id = Uuid::new_v4();

    let result = sqlx::query("INSERT INTO users (id, username, password_hash) VALUES ($1, $2, $3)")
        .bind(user_id)
        .bind(&payload.username)
        .bind(&password_hash)
        .execute(&state.db)
        .await;

    match result {
        Ok(_) => {
            session.insert("user_id", user_id).await.map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Session creation failed".to_string(),
                    }),
                )
            })?;

            session.insert("role", 0).await.map_err(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Session update failed".to_string(),
                    }),
                )
            })?;

            Ok((
                StatusCode::CREATED,
                jar,
                Json(AuthResponse {
                    user_id,
                    username: payload.username,
                    role: 0,
                }),
            ))
        }
        Err(e)
            if e.as_database_error()
                .map(|x| x.is_unique_violation())
                .unwrap_or(false) =>
        {
            Err((
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: "Username taken".to_string(),
                }),
            ))
        }
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
            }),
        )),
    }
}

pub async fn login(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
    Json(payload): Json<LoginRequest>,
) -> Result<(CookieJar, Json<AuthResponse>), (StatusCode, Json<ErrorResponse>)> {
    let user = sqlx::query_as::<_, (Uuid, String, String, i32)>(
        "SELECT id, username, password_hash, role FROM users WHERE username = $1",
    )
    .bind(&payload.username)
    .fetch_optional(&state.db)
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "DB Error".to_string(),
            }),
        )
    })?;

    let Some((user_id, username, password_hash, role)) = user else {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid credentials".to_string(),
            }),
        ));
    };

    let parsed_hash = PasswordHash::new(&password_hash).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Hash Error".to_string(),
            }),
        )
    })?;

    if Argon2::default()
        .verify_password(payload.password.as_bytes(), &parsed_hash)
        .is_err()
    {
        return Err((
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "Invalid credentials".to_string(),
            }),
        ));
    }

    session.insert("user_id", user_id).await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Session creation failed".to_string(),
            }),
        )
    })?;

    session.insert("role", role).await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Session update failed".to_string(),
            }),
        )
    })?;

    let mut response_jar = jar;
    if payload.remember_me {
        let series = Uuid::new_v4();
        let (token, token_hash) = generate_token_pair();
        let expires_at = OffsetDateTime::now_utc() + Duration::days(REMEMBER_DURATION_DAYS);

        sqlx::query("INSERT INTO remember_tokens (series, user_id, token_hash, expires_at) VALUES ($1, $2, $3, $4)")
            .bind(series).bind(user_id).bind(token_hash).bind(expires_at)
            .execute(&state.db).await
            .map_err(|_| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Failed to set remember token".to_string() })))?;

        let value = build_cookie_value(user_id, series, &token);
        let mut cookie = Cookie::new(REMEMBER_COOKIE_NAME, value);
        cookie.set_http_only(true);

        cookie.set_secure(state.is_production);

        cookie.set_same_site(SameSite::Lax);
        cookie.set_path("/");
        cookie.set_max_age(Duration::days(REMEMBER_DURATION_DAYS));
        cookie.set_domain(state.cookie_domain);

        response_jar = response_jar.add(cookie);
    }

    Ok((
        response_jar,
        Json(AuthResponse {
            user_id,
            username,
            role,
        }),
    ))
}

pub async fn logout(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
) -> impl IntoResponse {
    let _ = session.delete().await;

    let mut response_jar = jar;

    if let Some(cookie) = response_jar.get(REMEMBER_COOKIE_NAME) {
        if let Some((_, series, _)) = parse_cookie_value(cookie.value()) {
            let _ = sqlx::query("DELETE FROM remember_tokens WHERE series = $1")
                .bind(series)
                .execute(&state.db)
                .await;
        }
        let mut c = Cookie::from(REMEMBER_COOKIE_NAME);
        c.set_domain(state.cookie_domain.clone());
        c.set_path("/");
        response_jar = response_jar.remove(c);
    }

    (response_jar, StatusCode::NO_CONTENT)
}

pub async fn current_user(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
) -> Result<(CookieJar, Json<AuthResponse>), (StatusCode, Json<ErrorResponse>)> {
    let mut response_jar = jar.clone();

    let mut maybe_user_id: Option<Uuid> = session.get("user_id").await.ok().flatten();

    if maybe_user_id.is_none() {
        if let Some(cookie) = response_jar.get(REMEMBER_COOKIE_NAME) {
            if let Some((user_id, series, token)) = parse_cookie_value(cookie.value()) {
                if let Ok(Some((db_hash, db_uid, expires))) = sqlx::query_as::<
                    _,
                    (String, Uuid, OffsetDateTime),
                >(
                    "SELECT token_hash, user_id, expires_at FROM remember_tokens WHERE series = $1",
                )
                .bind(series)
                .fetch_optional(&state.db)
                .await
                {
                    let mut hasher = Sha256::new();
                    hasher.update(token.as_bytes());
                    let incoming_hash = BASE64.encode(hasher.finalize());

                    if db_uid == user_id
                        && incoming_hash == db_hash
                        && OffsetDateTime::now_utc() < expires
                    {
                        let (new_token, new_hash) = generate_token_pair();

                        sqlx::query("UPDATE remember_tokens SET token_hash = $1, last_used_at = NOW() WHERE series = $2")
                            .bind(new_hash).bind(series).execute(&state.db).await.ok();

                        session.insert("user_id", user_id).await.ok();
                        maybe_user_id = Some(user_id);

                        let value = build_cookie_value(user_id, series, &new_token);
                        let mut c = Cookie::new(REMEMBER_COOKIE_NAME, value);
                        c.set_http_only(true);

                        c.set_secure(state.is_production);

                        c.set_same_site(SameSite::Lax);
                        c.set_path("/");
                        c.set_max_age(Duration::days(REMEMBER_DURATION_DAYS));
                        c.set_domain(state.cookie_domain.clone());
                        response_jar = response_jar.add(c);
                    }
                }
            }
        }
    }

    let user_id = maybe_user_id.ok_or((
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "Not authenticated".to_string(),
        }),
    ))?;

    let user = sqlx::query_as::<_, (Uuid, String, i32)>(
        "SELECT id, username, role FROM users WHERE id = $1",
    )
    .bind(user_id)
    .fetch_optional(&state.db)
    .await
    .map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "DB Error".to_string(),
            }),
        )
    })?
    .ok_or((
        StatusCode::UNAUTHORIZED,
        Json(ErrorResponse {
            error: "User not found".to_string(),
        }),
    ))?;

    let role = user.2;
    session.insert("role", role).await.ok();

    Ok((
        response_jar,
        Json(AuthResponse {
            user_id: user.0,
            username: user.1,
            role,
        }),
    ))
}

pub async fn register_with_code(
    State(state): State<AppState>,
    session: Session,
    jar: CookieJar,
    Path(code): axum::extract::Path<String>,
    Json(payload): Json<RegisterRequest>,
) -> Result<(StatusCode, CookieJar, Json<AuthResponse>), (StatusCode, Json<ErrorResponse>)> {
    if payload.username.len() < 3 || payload.username.len() > 255 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Username must be 3-255 characters".to_string(),
            }),
        ));
    }
    if payload.password.len() < 8 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Password must be at least 8 characters".to_string(),
            }),
        ));
    }

    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(payload.password.as_bytes(), &salt)
        .map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Hashing failed".to_string(),
                }),
            )
        })?
        .to_string();

    let user_id = Uuid::new_v4();

    let mut tx = state.db.begin().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let code_update = sqlx::query!("UPDATE access_codes SET uses_left = uses_left - 1 WHERE code = $1 AND uses_left > 0 RETURNING uses_left", code)
        .fetch_optional(&mut *tx)
        .await;

    match code_update {
        Ok(Some(r)) => {
            if r.uses_left == 0 {
                let _ = sqlx::query!("DELETE FROM access_codes WHERE code = $1", code)
                    .execute(&mut *tx)
                    .await;
            }
        }
        Ok(None) => {
            let _ = tx.rollback().await;
            return Err((
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: "Invalid or already used access code".to_string(),
                }),
            ));
        }
        Err(e) => {
            let _ = tx.rollback().await;
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            ));
        }
    }

    let user_insert =
        sqlx::query("INSERT INTO users (id, username, password_hash) VALUES ($1, $2, $3)")
            .bind(user_id)
            .bind(&payload.username)
            .bind(&password_hash)
            .execute(&mut *tx)
            .await;

    if let Err(e) = user_insert {
        let _ = tx.rollback().await;
        if e.as_database_error()
            .map(|x| x.is_unique_violation())
            .unwrap_or(false)
        {
            return Err((
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: "Username taken".to_string(),
                }),
            ));
        }
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Database error".to_string(),
            }),
        ));
    }

    tx.commit().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    session.insert("user_id", user_id).await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Session creation failed".to_string(),
            }),
        )
    })?;

    session.insert("role", 0).await.map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: "Session update failed".to_string(),
            }),
        )
    })?;

    Ok((
        StatusCode::CREATED,
        jar,
        Json(AuthResponse {
            user_id,
            username: payload.username,
            role: 0,
        }),
    ))
}
