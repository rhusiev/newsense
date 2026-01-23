use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Deserialize)]
pub struct GetItemsQuery {
    pub limit: Option<i64>,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub since: Option<time::OffsetDateTime>,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub before: Option<time::OffsetDateTime>,
    pub unread_only: Option<bool>,
}

#[derive(Deserialize)]
pub struct GetClustersQuery {
    pub limit: Option<i64>,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub before: Option<time::OffsetDateTime>,
    pub unread_only: Option<bool>,
    pub feed_id: Option<Uuid>,
}

#[derive(Deserialize)]
pub struct UpdateItemStatusRequest {
    pub is_read: Option<bool>,
    pub liked: Option<f32>,
}

#[derive(Deserialize)]
pub struct MarkAllReadRequest {
    #[serde(with = "time::serde::iso8601")]
    pub since: time::OffsetDateTime,
}

#[derive(Serialize)]
pub struct ItemResponse {
    pub id: Uuid,
    pub feed_ids: Vec<Uuid>,
    pub title: String,
    pub link: String,
    pub content: Option<String>,
    pub author: Option<String>,
    #[serde(with = "time::serde::iso8601::option")]
    pub published_at: Option<time::OffsetDateTime>,
    pub cluster_id: Option<Uuid>,
    #[serde(with = "time::serde::iso8601::option")]
    pub created_at: Option<time::OffsetDateTime>,
    pub is_read: Option<bool>,
    pub liked: Option<f32>,
}

#[derive(Serialize)]
pub struct ClusterResponse {
    // cluster_id if exists, otherwise item_id
    pub id: Uuid,
    pub is_cluster: bool,
    #[serde(with = "time::serde::iso8601::option")]
    pub sort_date: Option<time::OffsetDateTime>,
    pub items: Vec<ItemResponse>,
}

#[derive(Serialize)]
pub struct ReadStatusResponse {
    pub user_id: Uuid,
    pub item_id: Uuid,
    pub is_read: bool,
    pub liked: f32,
    #[serde(with = "time::serde::iso8601::option")]
    pub marked_at: Option<time::OffsetDateTime>,
}

#[derive(Serialize)]
pub struct ClusterStatusResponse {
    pub updated_items: i64,
    pub is_read: bool,
    pub liked: f32,
}

#[derive(Serialize)]
pub struct UnreadCountResponse {
    pub unread_count: i64,
}

#[derive(Serialize)]
pub struct FeedUnreadCount {
    pub feed_id: Uuid,
    pub unread_count: i64,
}

#[derive(Serialize)]
pub struct AllUnreadCountsResponse {
    pub total_unread: i64,
    pub feeds: Vec<FeedUnreadCount>,
}

#[derive(Serialize)]
pub struct MarkAllReadResponse {
    pub marked_count: i64,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}
