use axum::{Router, routing::{get, post}};
use crate::{state::AppState, handlers};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/register", post(handlers::register))
        .route("/login", post(handlers::login))
        .route("/logout", post(handlers::logout))
        .route("/me", get(handlers::current_user))
}
