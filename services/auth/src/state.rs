use sqlx::PgPool;

#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
    pub is_production: bool,
    pub registration_enabled: bool,
    pub cookie_domain: String,
}
