use axum::{Router, routing::{delete, get, post}};
use crate::{state::AppState, handlers, admin};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/register", post(handlers::register))
        .route("/login", post(handlers::login))
        .route("/logout", post(handlers::logout))
        .route("/me", get(handlers::current_user))
        .route("/admin", get(admin::admin_check))
        .route("/admin/codes/generate", post(admin::create_codes))
        .route("/admin/codes", post(admin::create_named_code))
        .route("/admin/codes", get(admin::list_codes))
        .route("/admin/codes/count", get(admin::count_codes))
        .route("/admin/codes/{name}", delete(admin::delete_code))
}
