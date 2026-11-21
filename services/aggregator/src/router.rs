use axum::{
    routing::{get, post, put},
    Router,
};
use crate::{
    full_text::get_full_text,
    state::AppState,
    handlers::{
        get_all_items, get_all_unread_count, get_all_unread_counts, get_cluster_unread_count,
        get_clusters, get_feed_cluster_unread_count, get_feed_clusters, get_feed_items,
        get_feed_unread_count, get_item, get_story_summary_status, get_story_summary,
        get_story_thread, list_threads, mark_all_items_read, mark_feed_clusters_read, refresh_story_summary,
        mark_feed_items_read, update_cluster_status, update_item_status,
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
        .route("/items/{item_id}/full-text", get(get_full_text))
        .route("/clusters/feed/{feed_id}", get(get_feed_clusters))
        .route(
            "/clusters/feed/{feed_id}/unread-count",
            get(get_feed_cluster_unread_count),
        )
        .route(
            "/clusters/feed/{feed_id}/mark-read",
            post(mark_feed_clusters_read),
        )
        .route(
            "/clusters/feed/{feed_id}/unread-count",
            get(get_feed_cluster_unread_count),
        )
        .route("/items/{id}", get(get_item))
        .route("/clusters", get(get_clusters))
        .route("/threads", get(list_threads))
        .route("/clusters/{id}/status", put(update_cluster_status))
        .route("/clusters/{id}/thread", get(get_story_thread))
        .route("/clusters/{id}/summary", get(get_story_summary))
        .route(
            "/clusters/{id}/summary/status",
            get(get_story_summary_status),
        )
        .route(
            "/clusters/{id}/summary/refresh",
            post(refresh_story_summary),
        )
        .route("/clusters/unread-count", get(get_cluster_unread_count))
}
