use axum::{
    routing::{get, post, put},
    Router,
};
use crate::{
    state::AppState,
    handlers::{
        get_all_items, get_all_unread_count, get_all_unread_counts, get_cluster_unread_count,
        get_clusters, get_feed_clusters, get_feed_items, get_feed_unread_count,
        mark_all_items_read, mark_feed_clusters_read, mark_feed_items_read, update_cluster_status,
        update_item_status,
    },
};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/items/feed/{feed_id}", get(get_feed_items))
        .route(
            "/items/feed/{feed_id}/mark-read",
            post(mark_feed_items_read),
        )
        .route(
            "/items/feed/{feed_id}/unread-count",
            get(get_feed_unread_count),
        )
        .route("/items", get(get_all_items))
        .route("/items/mark-read", post(mark_all_items_read))
        .route("/items/unread-count", get(get_all_unread_count))
        .route("/items/unread-counts", get(get_all_unread_counts))
        .route("/items/{item_id}/status", put(update_item_status))
        .route("/clusters/feed/{feed_id}", get(get_feed_clusters))
        .route(
            "/clusters/feed/{feed_id}/mark-read",
            post(mark_feed_clusters_read),
        )
        .route("/clusters", get(get_clusters))
        .route("/clusters/{id}/status", put(update_cluster_status))
        .route("/clusters/unread-count", get(get_cluster_unread_count))
}
