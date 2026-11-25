use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

/// Chronological direction of a list. `newest` preserves the existing first
/// page at the top of the archive; `oldest` starts at the bottom and pages up.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ListOrder {
    #[default]
    Newest,
    Oldest,
}

impl ListOrder {
    pub fn ascending(self) -> bool {
        matches!(self, Self::Oldest)
    }
}

/// Where a page starts: after the row (`at`, `id`) in the list's direction.
/// `id` breaks ties between rows with the same date; without it the cursor is
/// a strict date bound, as older clients send.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageCursor {
    pub ascending: bool,
    pub at: Option<time::OffsetDateTime>,
    pub id: Option<Uuid>,
}

impl PageCursor {
    /// `before` pages a newest-first list and `after` an oldest-first one.
    pub fn new(
        order: Option<ListOrder>,
        before: Option<time::OffsetDateTime>,
        after: Option<time::OffsetDateTime>,
        id: Option<Uuid>,
    ) -> Result<Self, String> {
        let ascending = order.unwrap_or_default().ascending();
        let at = match (ascending, before, after) {
            (true, Some(_), _) => return Err("before needs order=newest; use after".into()),
            (false, _, Some(_)) => return Err("after needs order=oldest; use before".into()),
            (true, None, at) | (false, at, None) => at,
        };
        if id.is_some() && at.is_none() {
            return Err("cursor_id needs before or after".into());
        }
        Ok(Self { ascending, at, id })
    }
}

#[derive(Deserialize)]
pub struct GetItemsQuery {
    pub limit: Option<i64>,
    /// Deprecated alias for `after` on the per-feed item list.
    #[serde(default, with = "time::serde::iso8601::option")]
    pub since: Option<time::OffsetDateTime>,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub before: Option<time::OffsetDateTime>,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub after: Option<time::OffsetDateTime>,
    pub order: Option<ListOrder>,
    /// Id of the last row shown, to break ties with the date cursor.
    pub cursor_id: Option<Uuid>,
    pub unread_only: Option<bool>,
    pub score_min: Option<f32>,
    pub score_max: Option<f32>,
    /// Keep the N best-scored items or stories of each day.
    pub top_n: Option<i64>,
    pub explore: Option<f32>,
    pub tz: Option<String>,
    pub per_feed: Option<bool>,
}

impl GetItemsQuery {
    /// The page cursor. `since` is the older spelling of `after` and implies
    /// oldest-first.
    pub fn cursor(&self) -> Result<PageCursor, String> {
        match self.since {
            Some(since) if self.after.is_none() => PageCursor::new(
                Some(ListOrder::Oldest),
                self.before,
                Some(since),
                self.cursor_id,
            ),
            _ => PageCursor::new(self.order, self.before, self.after, self.cursor_id),
        }
    }

    /// The ranking fields, for a list that keeps a daily top N.
    pub fn ranked(&self) -> GetClustersQuery {
        GetClustersQuery {
            limit: self.limit,
            before: self.before,
            after: self.after,
            order: self.order,
            cursor_id: self.cursor_id,
            unread_only: self.unread_only,
            feed_id: None,
            score_min: self.score_min,
            score_max: self.score_max,
            top_n: self.top_n,
            explore: self.explore,
            tz: self.tz.clone(),
            per_feed: self.per_feed,
        }
    }
}

#[derive(Deserialize)]
pub struct GetClustersQuery {
    pub limit: Option<i64>,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub before: Option<time::OffsetDateTime>,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub after: Option<time::OffsetDateTime>,
    pub order: Option<ListOrder>,
    /// Id of the last story shown, to break ties with the date cursor.
    pub cursor_id: Option<Uuid>,
    pub unread_only: Option<bool>,
    pub feed_id: Option<Uuid>,
    pub score_min: Option<f32>,
    pub score_max: Option<f32>,
    /// Keep the N best-scored clusters of each day instead of a score threshold.
    pub top_n: Option<i64>,
    /// Share of the N daily slots given to clusters below the cut (exploration).
    pub explore: Option<f32>,
    /// IANA time zone that decides where a day starts, such as `Europe/Kyiv`.
    pub tz: Option<String>,
    /// On the merged list: apply each feed's own filter before merging.
    pub per_feed: Option<bool>,
}

impl GetClustersQuery {
    pub fn cursor(&self) -> Result<PageCursor, String> {
        PageCursor::new(self.order, self.before, self.after, self.cursor_id)
    }
}

#[derive(Deserialize)]
pub struct GetUnreadCountsQuery {
    pub score_min: Option<f32>,
    pub score_max: Option<f32>,
    pub top_n: Option<i64>,
    pub explore: Option<f32>,
    pub tz: Option<String>,
    pub per_feed: Option<bool>,
}

impl GetUnreadCountsQuery {
    /// The ranking fields, for a badge that counts a daily top N.
    pub fn ranked(&self) -> GetClustersQuery {
        GetClustersQuery {
            limit: None,
            before: None,
            after: None,
            order: None,
            cursor_id: None,
            unread_only: Some(true),
            feed_id: None,
            score_min: self.score_min,
            score_max: self.score_max,
            top_n: self.top_n,
            explore: self.explore,
            tz: self.tz.clone(),
            per_feed: self.per_feed,
        }
    }
}

#[derive(Deserialize)]
pub struct UpdateItemStatusRequest {
    pub is_read: Option<bool>,
    /// Absent leaves the label alone; `null` clears it to unrated.
    #[serde(default, deserialize_with = "present")]
    pub liked: Option<Option<f32>>,
}

/// Tells a field sent as `null` (`Some(None)`) apart from a missing one (`None`).
fn present<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
pub struct MarkAllReadRequest {
    #[serde(with = "time::serde::iso8601")]
    pub since: time::OffsetDateTime,
    #[serde(default, with = "time::serde::iso8601::option")]
    pub before: Option<time::OffsetDateTime>,
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
    pub prediction_score: Option<f32>,
}

/// A member row of a story list: the item plus the date its story is paged by.
pub struct ClusterItemRow {
    pub id: Uuid,
    pub feed_ids: Vec<Uuid>,
    pub title: String,
    pub link: String,
    pub content: Option<String>,
    pub author: Option<String>,
    pub published_at: Option<time::OffsetDateTime>,
    pub cluster_id: Option<Uuid>,
    pub created_at: Option<time::OffsetDateTime>,
    pub is_read: Option<bool>,
    pub liked: Option<f32>,
    pub prediction_score: Option<f32>,
    pub story_date: Option<time::OffsetDateTime>,
}

impl ClusterItemRow {
    pub fn split(self) -> (ItemResponse, Option<time::OffsetDateTime>) {
        (
            ItemResponse {
                id: self.id,
                feed_ids: self.feed_ids,
                title: self.title,
                link: self.link,
                content: self.content,
                author: self.author,
                published_at: self.published_at,
                cluster_id: self.cluster_id,
                created_at: self.created_at,
                is_read: self.is_read,
                liked: self.liked,
                prediction_score: self.prediction_score,
            },
            self.story_date,
        )
    }
}

#[derive(Serialize)]
pub struct ClusterResponse {
    pub id: Uuid,
    pub is_cluster: bool,
    #[serde(with = "time::serde::iso8601::option")]
    pub sort_date: Option<time::OffsetDateTime>,
    /// Shown from below the top-N cut to learn about clusters the model ranks low.
    pub explored: bool,
    pub items: Vec<ItemResponse>,
}

#[derive(Serialize)]
pub struct ReadStatusResponse {
    pub user_id: Uuid,
    pub item_id: Uuid,
    pub is_read: bool,
    pub liked: Option<f32>,
    #[serde(with = "time::serde::iso8601::option")]
    pub marked_at: Option<time::OffsetDateTime>,
}

#[derive(Serialize)]
pub struct ClusterStatusResponse {
    pub updated_items: i64,
    pub is_read: bool,
    pub liked: Option<f32>,
}

#[derive(Serialize)]
pub struct ThreadBriefResponse {
    pub id: Uuid,
    pub title: String,
    #[serde(with = "time::serde::iso8601")]
    pub first_date: time::OffsetDateTime,
    #[serde(with = "time::serde::iso8601")]
    pub last_date: time::OffsetDateTime,
    pub story_count: i64,
    pub unread_count: i64,
}

#[derive(Serialize)]
pub struct ThreadStoryResponse {
    pub id: Uuid,
    pub item_id: Uuid,
    pub title: String,
    #[serde(with = "time::serde::iso8601")]
    pub sort_date: time::OffsetDateTime,
    pub item_count: i64,
    pub is_read: bool,
    pub liked: Option<f32>,
}

#[derive(Serialize)]
pub struct ThreadResponse {
    pub id: Uuid,
    pub title: String,
    pub stories: Vec<ThreadStoryResponse>,
}

#[derive(Serialize)]
pub struct StorySummaryResponse {
    pub summary: String,
    pub model: Option<String>,
    #[serde(with = "time::serde::iso8601")]
    pub updated_at: time::OffsetDateTime,
}

#[derive(Serialize)]
pub struct StorySummaryStatusResponse {
    /// "ready", "pending" or "not-applicable".
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(with = "time::serde::iso8601::option")]
    pub updated_at: Option<time::OffsetDateTime>,
    /// Another article joined the story after the summary was made.
    pub can_recreate: bool,
    pub used_today: i64,
    pub daily_limit: i32,
}

#[derive(Serialize)]
pub struct UnreadCountResponse {
    pub unread_count: i64,
}

#[derive(Serialize)]
pub struct FeedUnreadCount {
    pub feed_id: Uuid,
    pub unread_count: i64,
    pub read_count: i64,
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

#[cfg(test)]
mod tests {
    use super::{ListOrder, PageCursor, UpdateItemStatusRequest};
    use uuid::Uuid;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct OrderQuery {
        #[serde(default)]
        order: Option<ListOrder>,
    }

    #[test]
    fn list_order_accepts_only_the_two_directions() {
        let query = |json| serde_json::from_str::<OrderQuery>(json);
        assert!(query("{}").unwrap().order.is_none());
        assert_eq!(
            query(r#"{"order":"oldest"}"#).unwrap().order,
            Some(ListOrder::Oldest)
        );
        assert_eq!(
            query(r#"{"order":"newest"}"#).unwrap().order,
            Some(ListOrder::Newest)
        );
        assert!(query(r#"{"order":"middle"}"#).is_err());
    }

    fn at(s: &str) -> Option<time::OffsetDateTime> {
        Some(time::OffsetDateTime::parse(s, &time::format_description::well_known::Rfc3339).unwrap())
    }

    #[test]
    fn page_cursor_matches_the_order() {
        let t = at("2026-01-01T00:00:00Z");
        let id = Some(Uuid::nil());
        let oldest = PageCursor::new(Some(ListOrder::Oldest), None, t, id).unwrap();
        assert_eq!(oldest, PageCursor { ascending: true, at: t, id });
        let newest = PageCursor::new(None, t, None, None).unwrap();
        assert_eq!(newest, PageCursor { ascending: false, at: t, id: None });
        assert!(PageCursor::new(Some(ListOrder::Oldest), t, None, None).is_err());
        assert!(PageCursor::new(Some(ListOrder::Newest), None, t, None).is_err());
        assert!(PageCursor::new(None, None, None, id).is_err());
    }

    fn liked(body: &str) -> Option<Option<f32>> {
        serde_json::from_str::<UpdateItemStatusRequest>(body)
            .unwrap()
            .liked
    }

    #[test]
    fn liked_absent_null_and_value_stay_distinct() {
        assert_eq!(liked(r#"{"is_read":true}"#), None);
        assert_eq!(liked(r#"{"liked":null}"#), Some(None));
        assert_eq!(liked(r#"{"liked":0}"#), Some(Some(0.0)));
        assert_eq!(liked(r#"{"liked":-1}"#), Some(Some(-1.0)));
    }
}
