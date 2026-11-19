//! Top-N per day: keep the best-scored clusters of each day, plus a few from
//! below the cut (exploration). A score threshold shows fewer items as the
//! model ages; a daily count does not.
//!
//! Selection is deterministic: within each pool, clusters are picked in the
//! order of a hash of user and cluster, so a reload shows the same set. Every
//! shown cluster is logged in `exposures` with the chance it had to be picked,
//! so an evaluation can weight explored labels by 1 / propensity.

use axum::{Json, http::StatusCode};
use uuid::Uuid;

use crate::{
    handlers::group_clusters,
    models::{ClusterResponse, ErrorResponse, GetClustersQuery, ItemResponse},
    state::AppState,
};

type ApiError = (StatusCode, Json<ErrorResponse>);

const MAX_TOP_N: i64 = 200;
const DEFAULT_EXPLORE: f32 = 0.1;
const MAX_EXPLORE: f32 = 0.5;
const DEFAULT_TZ: &str = "UTC";
const DEFAULT_LIMIT: i64 = 50;
const MAX_LIMIT: i64 = 100;
/// Days ranked before the cursor on each page. Older pages move the window.
const LOOKBACK_DAYS: i32 = 31;
const TOP_POLICY: &str = "top";

/// How the N daily slots are split between the top ranks and exploration.
#[derive(Debug, PartialEq)]
pub struct Slots {
    /// Clusters kept from the top of the ranking.
    pub keep: i64,
    /// Clusters picked from the N ranks just below the kept ones.
    pub boundary: i64,
    /// Clusters picked from everything below the boundary pool.
    pub random: i64,
}

impl Slots {
    /// Rounds the exploration share; half of it, rounded up, goes to the
    /// boundary pool, which tells most about where the cut should be.
    pub fn new(n: i64, explore: f32) -> Self {
        let explored = (n as f32 * explore).round() as i64;
        let boundary = (explored + 1) / 2;
        Slots {
            keep: n - explored,
            boundary,
            random: explored - boundary,
        }
    }
}

fn bad_request(error: String) -> ApiError {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }))
}

fn internal(e: sqlx::Error) -> ApiError {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: e.to_string(),
        }),
    )
}

struct Picked {
    entity_id: Uuid,
    day: time::Date,
    /// `top`, `boundary` or `random`, as stored in `exposures.policy`.
    policy: String,
    propensity: f64,
    rank: i64,
    score: f64,
}

pub async fn top_clusters(
    state: &AppState,
    user_id: Uuid,
    top_n: i64,
    params: &GetClustersQuery,
) -> Result<Vec<ClusterResponse>, ApiError> {
    if !(1..=MAX_TOP_N).contains(&top_n) {
        return Err(bad_request(format!(
            "top_n must be between 1 and {MAX_TOP_N}"
        )));
    }
    let explore = params.explore.unwrap_or(DEFAULT_EXPLORE);
    if !(0.0..=MAX_EXPLORE).contains(&explore) {
        return Err(bad_request(format!(
            "explore must be between 0 and {MAX_EXPLORE}"
        )));
    }
    let tz = params.tz.as_deref().unwrap_or(DEFAULT_TZ);
    if !is_known_tz(state, tz).await? {
        return Err(bad_request(format!("unknown time zone {tz}")));
    }

    let slots = Slots::new(top_n, explore);
    let picked = pick_entities(state, user_id, tz, &slots, top_n, params).await?;
    log_exposures(state, user_id, &picked).await?;

    let ids: Vec<Uuid> = picked.iter().map(|p| p.entity_id).collect();
    let mut clusters = group_clusters(fetch_items(state, user_id, &ids).await?);
    for cluster in &mut clusters {
        cluster.explored = picked
            .iter()
            .any(|p| p.entity_id == cluster.id && p.policy != TOP_POLICY);
    }
    Ok(clusters)
}

async fn is_known_tz(state: &AppState, tz: &str) -> Result<bool, ApiError> {
    sqlx::query_scalar!(
        r#"SELECT EXISTS(SELECT 1 FROM pg_timezone_names WHERE name = $1) as "exists!""#,
        tz
    )
    .fetch_one(&state.db)
    .await
    .map_err(internal)
}

/// Ranks each day's clusters by their best item score and picks the shown ones.
/// A day counts only when the window covers all of it.
async fn pick_entities(
    state: &AppState,
    user_id: Uuid,
    tz: &str,
    slots: &Slots,
    pool_width: i64,
    params: &GetClustersQuery,
) -> Result<Vec<Picked>, ApiError> {
    sqlx::query_as!(
        Picked,
        r#"
        WITH window_start AS (
            SELECT COALESCE($2::timestamptz, now()) - make_interval(days => $10) AS at
        ),
        entities AS (
            SELECT
                COALESCE(i.cluster_id, i.id) AS entity_id,
                MAX(i.published_at) AS max_date,
                MAX(COALESCE(ip.score, 0.0))::float8 AS score,
                BOOL_AND(COALESCE(ir.is_read, false)) AS all_read
            FROM items i
            INNER JOIN feeds f ON f.source_id = i.source_id
            INNER JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $1
            LEFT JOIN item_reads ir ON ir.item_id = i.id AND ir.user_id = $1
            LEFT JOIN item_predictions ip ON ip.item_id = i.id AND ip.user_id = $1
            WHERE i.published_at >= (SELECT at FROM window_start)
            GROUP BY 1
        ),
        ranked AS (
            SELECT e.*, d.day,
                ROW_NUMBER() OVER (PARTITION BY d.day ORDER BY e.score DESC, e.entity_id) AS rnk
            FROM entities e
            CROSS JOIN LATERAL (SELECT (e.max_date AT TIME ZONE $3)::date AS day) d
        ),
        pooled AS (
            SELECT r.*,
                CASE
                    WHEN rnk <= $4 THEN 'top'
                    WHEN rnk <= $4 + $7 THEN 'boundary'
                    ELSE 'random'
                END AS policy
            FROM ranked r
        ),
        picked AS (
            SELECT p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY day, policy ORDER BY md5($1::text || entity_id::text)
                ) AS pick,
                COUNT(*) OVER (PARTITION BY day, policy) AS pool_size
            FROM pooled p
        )
        SELECT
            entity_id AS "entity_id!",
            day AS "day!",
            policy AS "policy!",
            CASE policy
                WHEN 'top' THEN 1.0
                WHEN 'boundary' THEN LEAST(1.0, $5::bigint::float8 / pool_size)
                ELSE LEAST(1.0, $6::bigint::float8 / pool_size)
            END::float8 AS "propensity!",
            rnk AS "rank!",
            score AS "score!"
        FROM picked
        WHERE (policy = 'top'
               OR (policy = 'boundary' AND pick <= $5)
               OR (policy = 'random' AND pick <= $6))
          AND ($2::timestamptz IS NULL OR max_date < $2)
          AND day > ((SELECT at FROM window_start) AT TIME ZONE $3)::date
          AND ($8::boolean = false OR NOT all_read)
        ORDER BY max_date DESC
        LIMIT $9
        "#,
        user_id,
        params.before,
        tz,
        slots.keep,
        slots.boundary,
        slots.random,
        pool_width,
        params.unread_only.unwrap_or(false),
        params.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT),
        LOOKBACK_DAYS
    )
    .fetch_all(&state.db)
    .await
    .map_err(internal)
}

/// Records each shown cluster once, with the chance it had to be shown.
async fn log_exposures(state: &AppState, user_id: Uuid, picked: &[Picked]) -> Result<(), ApiError> {
    let ids: Vec<Uuid> = picked.iter().map(|p| p.entity_id).collect();
    let days: Vec<time::Date> = picked.iter().map(|p| p.day).collect();
    let policies: Vec<String> = picked.iter().map(|p| p.policy.clone()).collect();
    let propensities: Vec<f32> = picked.iter().map(|p| p.propensity as f32).collect();
    let ranks: Vec<i32> = picked.iter().map(|p| p.rank as i32).collect();
    let scores: Vec<f32> = picked.iter().map(|p| p.score as f32).collect();
    sqlx::query!(
        r#"
        INSERT INTO exposures (user_id, entity_id, day, policy, propensity, rank, score)
        SELECT $1, * FROM UNNEST($2::uuid[], $3::date[], $4::text[], $5::real[], $6::int[], $7::real[])
        ON CONFLICT (user_id, entity_id) DO NOTHING
        "#,
        user_id,
        &ids,
        &days,
        &policies,
        &propensities,
        &ranks,
        &scores
    )
    .execute(&state.db)
    .await
    .map_err(internal)?;
    Ok(())
}

async fn fetch_items(
    state: &AppState,
    user_id: Uuid,
    entity_ids: &[Uuid],
) -> Result<Vec<ItemResponse>, ApiError> {
    sqlx::query_as!(
        ItemResponse,
        r#"
        SELECT
            i.id as "id!",
            array_agg(f.id) as "feed_ids!",
            i.title, i.link, i.content, i.author,
            i.published_at, i.cluster_id, i.created_at,
            ir.is_read as "is_read?",
            ROUND(ir.liked)::REAL as "liked?",
            ip.score as "prediction_score?"
        FROM items i
        INNER JOIN feeds f ON f.source_id = i.source_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE COALESCE(i.cluster_id, i.id) = ANY($2::uuid[])
        GROUP BY i.id, ir.is_read, ir.liked, ip.score
        ORDER BY i.published_at DESC
        "#,
        user_id,
        entity_ids
    )
    .fetch_all(&state.db)
    .await
    .map_err(internal)
}

#[cfg(test)]
mod tests {
    use super::Slots;

    #[test]
    fn slots_split_exploration_between_boundary_and_random() {
        let s = |keep, boundary, random| Slots {
            keep,
            boundary,
            random,
        };
        assert_eq!(Slots::new(20, 0.1), s(18, 1, 1));
        assert_eq!(Slots::new(15, 0.1), s(13, 1, 1));
        assert_eq!(Slots::new(30, 0.1), s(27, 2, 1));
        assert_eq!(Slots::new(10, 0.0), s(10, 0, 0));
        assert_eq!(Slots::new(1, 0.1), s(1, 0, 0));
    }
}
