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
/// Days ranked before the cursor on each page at least; the window also reaches
/// back to the oldest unread article, so a backlog is never cut off.
const LOOKBACK_DAYS: i32 = 31;
const TOP_POLICY: &str = "top";

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

/// Ranks the user's subscribed feeds together, or only `feed_id` when given.
/// With `per_feed`, each feed applies its own filter (`feed_subscriptions.filter_mode`:
/// 'off' passes every story, 'score' its threshold, 'top' its own N) before the
/// lists merge; otherwise one N (`top_n`) applies to the merged list.
pub async fn top_clusters(
    state: &AppState,
    user_id: Uuid,
    top_n: i64,
    feed_id: Option<Uuid>,
    params: &GetClustersQuery,
) -> Result<Vec<ClusterResponse>, ApiError> {
    let scope = scope(state, top_n, feed_id, params).await?;
    let limit = params.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);
    let unread_only = params.unread_only.unwrap_or(false);
    let picked = pick_entities(state, user_id, &scope, params.before, unread_only, limit).await?;
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

/// Counts the unread clusters the top-N list would show over its whole window,
/// for the sidebar badge. Logs no exposures: nothing is shown.
pub async fn count_unread_top(
    state: &AppState,
    user_id: Uuid,
    top_n: i64,
    params: &GetClustersQuery,
) -> Result<i64, ApiError> {
    let scope = scope(state, top_n, None, params).await?;
    // The window can span months of backlog; count all of it.
    let all_days = i64::from(i32::MAX);
    let picked = pick_entities(state, user_id, &scope, None, true, all_days).await?;
    Ok(picked.len() as i64)
}

async fn scope<'a>(
    state: &AppState,
    top_n: i64,
    feed_id: Option<Uuid>,
    params: &'a GetClustersQuery,
) -> Result<Scope<'a>, ApiError> {
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
    Ok(Scope {
        top_n,
        explore,
        tz,
        feed_id,
        per_feed: params.per_feed.unwrap_or(false),
    })
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

struct Scope<'a> {
    top_n: i64,
    explore: f32,
    tz: &'a str,
    feed_id: Option<Uuid>,
    per_feed: bool,
}

/// Ranks each day's clusters by their best item score, within each feed when
/// `per_feed`, and picks the shown ones. Feeds filtered by 'off' or 'score'
/// pass their stories without ranking or exploration. Of N slots, round(N * explore) go to
/// exploration: half of them, rounded up, to the N ranks just below the kept
/// ones, the rest to everything lower. A day counts only when the window
/// covers all of it. A cluster picked in several feeds is shown once, with its
/// best pick logged.
async fn pick_entities(
    state: &AppState,
    user_id: Uuid,
    scope: &Scope<'_>,
    before: Option<time::OffsetDateTime>,
    unread_only: bool,
    limit: i64,
) -> Result<Vec<Picked>, ApiError> {
    sqlx::query_as!(
        Picked,
        r#"
        WITH scoped_feeds AS (
            SELECT f.id AS feed_id, f.source_id,
                CASE WHEN $6::boolean THEN fs.filter_mode ELSE 'top' END AS mode,
                CASE WHEN $6::boolean THEN fs.top_n::bigint ELSE $4::bigint END AS n,
                fs.score_threshold::float8 AS threshold
            FROM feeds f
            LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $1
            WHERE ($7::uuid IS NULL AND fs.user_id IS NOT NULL) OR f.id = $7
        ),
        cursor_at AS (SELECT COALESCE($2::timestamptz, now()) AS at),
        -- Reaches back to the oldest unread article before the cursor, and at least $10 days.
        window_start AS (
            SELECT LEAST(
                (SELECT at FROM cursor_at) - make_interval(days => $10),
                (SELECT MIN(i.published_at) - interval '1 day'
                 FROM items i
                 JOIN scoped_feeds sf ON sf.source_id = i.source_id
                 LEFT JOIN item_reads ir ON ir.item_id = i.id AND ir.user_id = $1
                 WHERE COALESCE(ir.is_read, false) = false
                   AND i.published_at < (SELECT at FROM cursor_at))
            ) AS at
        ),
        entities AS (
            SELECT
                CASE WHEN $6::boolean THEN sf.feed_id END AS scope_id,
                MAX(sf.mode) AS mode,
                MAX(sf.n) AS n,
                MAX(sf.threshold) AS threshold,
                COALESCE(i.cluster_id, i.id) AS entity_id,
                MAX(i.published_at) AS max_date,
                MAX(COALESCE(ip.score, 0.0))::float8 AS score,
                BOOL_AND(COALESCE(ir.is_read, false)) AS all_read
            FROM items i
            INNER JOIN scoped_feeds sf ON sf.source_id = i.source_id
            LEFT JOIN item_reads ir ON ir.item_id = i.id AND ir.user_id = $1
            LEFT JOIN item_predictions ip ON ip.item_id = i.id AND ip.user_id = $1
            WHERE i.published_at >= (SELECT at FROM window_start)
            GROUP BY 1, 5
        ),
        ranked AS (
            SELECT e.*, d.day, sl.keep, sl.boundary, sl.explored - sl.boundary AS random,
                ROW_NUMBER() OVER (
                    PARTITION BY e.scope_id, d.day ORDER BY e.score DESC, e.entity_id
                ) AS rnk
            FROM entities e
            CROSS JOIN LATERAL (SELECT (e.max_date AT TIME ZONE $3)::date AS day) d
            CROSS JOIN LATERAL (SELECT ROUND(e.n * $5::float8)::bigint AS explored) x
            CROSS JOIN LATERAL (
                SELECT x.explored, e.n - x.explored AS keep, (x.explored + 1) / 2 AS boundary
            ) sl
        ),
        pooled AS (
            SELECT r.*,
                CASE
                    WHEN mode = 'off' THEN 'top'
                    WHEN mode = 'score' THEN CASE WHEN score >= threshold THEN 'top' ELSE 'hidden' END
                    WHEN rnk <= keep THEN 'top'
                    WHEN rnk <= keep + n THEN 'boundary'
                    ELSE 'random'
                END AS policy
            FROM ranked r
        ),
        picked AS (
            SELECT p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY scope_id, day, policy ORDER BY md5($1::text || entity_id::text)
                ) AS pick,
                COUNT(*) OVER (PARTITION BY scope_id, day, policy) AS pool_size
            FROM pooled p
        ),
        shown AS (
            SELECT DISTINCT ON (entity_id)
                entity_id, day, policy, rnk, score, max_date, all_read,
                CASE policy
                    WHEN 'top' THEN 1.0
                    WHEN 'boundary' THEN LEAST(1.0, boundary::float8 / pool_size)
                    ELSE LEAST(1.0, random::float8 / pool_size)
                END::float8 AS propensity
            FROM picked
            WHERE policy = 'top'
               OR (policy = 'boundary' AND pick <= boundary)
               OR (policy = 'random' AND pick <= random)
            ORDER BY entity_id, (policy = 'top') DESC, propensity DESC
        )
        SELECT
            entity_id AS "entity_id!",
            day AS "day!",
            policy AS "policy!",
            propensity AS "propensity!",
            rnk AS "rank!",
            score AS "score!"
        FROM shown
        WHERE ($2::timestamptz IS NULL OR max_date < $2)
          AND day > ((SELECT at FROM window_start) AT TIME ZONE $3)::date
          AND ($8::boolean = false OR NOT all_read)
        ORDER BY max_date DESC
        LIMIT $9
        "#,
        user_id,
        before,
        scope.tz,
        scope.top_n,
        scope.explore as f64,
        scope.per_feed,
        scope.feed_id,
        unread_only,
        limit,
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
