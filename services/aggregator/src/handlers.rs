use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
};
use std::collections::HashMap;
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    models::{
        AllUnreadCountsResponse, ClusterItemRow, ClusterResponse, ClusterStatusResponse, ErrorResponse,
        FeedUnreadCount, GetClustersQuery, GetItemsQuery, GetUnreadCountsQuery, ItemResponse,
        MarkAllReadRequest, MarkAllReadResponse, ReadStatusResponse, StorySummaryResponse,
        StorySummaryStatusResponse, ThreadBriefResponse, ThreadResponse, ThreadStoryResponse,
        UnreadCountResponse, UpdateItemStatusRequest,
    },
    ranking,
    state::AppState,
};

fn bad_request(error: String) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::BAD_REQUEST, Json(ErrorResponse { error }))
}

pub async fn get_feed_items(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(feed_id): Path<Uuid>,
    Query(params): Query<GetItemsQuery>,
) -> Result<Json<Vec<ItemResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Feed not found".into(),
            }),
        ));
    }

    if let Some(top_n) = params.top_n {
        return ranking::top_items(&state, user_id, top_n, Some(feed_id), &params.ranked())
            .await
            .map(Json);
    }

    let limit = params.limit.unwrap_or(50).min(1000);
    let unread_only = params.unread_only.unwrap_or(false);
    let cursor = params.cursor().map_err(bad_request)?;

    let items = sqlx::query_as!(
        ItemResponse,
        r#"
        SELECT
            i.id as "id!",
            ARRAY[$2::uuid] as "feed_ids!",
            i.title, i.link, i.content, i.author,
            i.published_at, i.cluster_id, i.created_at,
            ir.is_read as "is_read?",
            ROUND(ir.liked)::REAL as "liked?",
            ip.score as "prediction_score?"
        FROM items i
        INNER JOIN feeds f ON i.source_id = f.source_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE f.id = $2
          AND ($3::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
          AND ($4::timestamptz IS NULL OR CASE WHEN $8::boolean
              THEN i.published_at > $4 OR (i.published_at = $4 AND i.id > $5::uuid)
              ELSE i.published_at < $4 OR (i.published_at = $4 AND i.id < $5::uuid) END)
          AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
          AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        ORDER BY
            CASE WHEN $8::boolean THEN i.published_at END ASC,
            CASE WHEN $8::boolean THEN i.id END ASC,
            CASE WHEN NOT $8::boolean THEN i.published_at END DESC,
            CASE WHEN NOT $8::boolean THEN i.id END DESC
        LIMIT $9
        "#,
        user_id,
        feed_id,
        unread_only,
        cursor.at,
        cursor.id,
        params.score_min,
        params.score_max,
        cursor.ascending,
        limit
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    log_impressions(&state, user_id, &story_ids(&items)).await;
    Ok(Json(items))
}

pub async fn get_all_items(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Query(params): Query<GetItemsQuery>,
) -> Result<Json<Vec<ItemResponse>>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(top_n) = params.top_n {
        return ranking::top_items(&state, user_id, top_n, None, &params.ranked())
            .await
            .map(Json);
    }
    let limit = params.limit.unwrap_or(50).min(1000);
    let unread_only = params.unread_only.unwrap_or(false);
    let cursor = params.cursor().map_err(bad_request)?;

    let items = sqlx::query_as!(
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
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE fs.user_id = $1
          AND ($2::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
          AND ($3::timestamptz IS NULL OR CASE WHEN $7::boolean
              THEN i.published_at > $3 OR (i.published_at = $3 AND i.id > $4::uuid)
              ELSE i.published_at < $3 OR (i.published_at = $3 AND i.id < $4::uuid) END)
          AND ($5::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $5)
          AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $6)
        GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
        ORDER BY
            CASE WHEN $7::boolean THEN i.published_at END ASC,
            CASE WHEN $7::boolean THEN i.id END ASC,
            CASE WHEN NOT $7::boolean THEN i.published_at END DESC,
            CASE WHEN NOT $7::boolean THEN i.id END DESC
        LIMIT $8
        "#,
        user_id,
        unread_only,
        cursor.at,
        cursor.id,
        params.score_min,
        params.score_max,
        cursor.ascending,
        limit
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    log_impressions(&state, user_id, &story_ids(&items)).await;
    Ok(Json(items))
}

pub async fn update_item_status(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(item_id): Path<Uuid>,
    Json(payload): Json<UpdateItemStatusRequest>,
) -> Result<Json<ReadStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM items i
            INNER JOIN feeds f ON i.source_id = f.source_id
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE i.id = $2 AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        item_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Item not found".into(),
            }),
        ));
    }

    let status = sqlx::query_as!(
        ReadStatusResponse,
        r#"
        INSERT INTO item_reads (user_id, item_id, is_read, liked)
        VALUES ($1, $2, COALESCE($3::BOOLEAN, true), $4::REAL)
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET
            is_read = COALESCE($3::BOOLEAN, item_reads.is_read),
            liked = CASE WHEN $5::BOOLEAN THEN $4::REAL ELSE item_reads.liked END,
            marked_at = CURRENT_TIMESTAMP
        RETURNING
            user_id, item_id, is_read, ROUND(liked)::REAL as "liked?", marked_at
        "#,
        user_id,
        item_id,
        payload.is_read,
        payload.liked.flatten(),
        payload.liked.is_some()
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(status))
}

pub async fn mark_feed_items_read(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(feed_id): Path<Uuid>,
    Json(payload): Json<MarkAllReadRequest>,
) -> Result<Json<MarkAllReadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Feed not found".into(),
            }),
        ));
    }

    let result = sqlx::query!(
        r#"
        INSERT INTO item_reads (user_id, item_id, is_read, liked, marked_at)
        SELECT $1, i.id, true, NULL::REAL, CURRENT_TIMESTAMP
        FROM items i
        INNER JOIN feeds f ON i.source_id = f.source_id
        WHERE f.id = $2 AND i.published_at >= $3
          AND ($4::TIMESTAMPTZ IS NULL OR i.published_at < $4)
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET is_read = true, marked_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        feed_id,
        payload.since,
        payload.before
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(MarkAllReadResponse {
        marked_count: result.rows_affected() as i64,
    }))
}

pub async fn mark_all_items_read(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Json(payload): Json<MarkAllReadRequest>,
) -> Result<Json<MarkAllReadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = sqlx::query!(
        r#"
        INSERT INTO item_reads (user_id, item_id, is_read, liked, marked_at)
        SELECT DISTINCT $1::UUID, i.id, true, NULL::REAL, CURRENT_TIMESTAMP
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id
        WHERE (f.owner_id = $1 OR fs.user_id = $1)
          AND i.published_at >= $2
          AND ($3::TIMESTAMPTZ IS NULL OR i.published_at < $3)
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET is_read = true, marked_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        payload.since,
        payload.before
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(MarkAllReadResponse {
        marked_count: result.rows_affected() as i64,
    }))
}

pub async fn get_feed_unread_count(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(feed_id): Path<Uuid>,
    Query(params): Query<GetUnreadCountsQuery>,
) -> Result<Json<UnreadCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Feed not found".into(),
            }),
        ));
    }

    if let Some(top_n) = params.top_n {
        let unread_count = ranking::count_unread_top(
            &state,
            user_id,
            top_n,
            &params.ranked(),
            Some(feed_id),
            false,
        )
        .await?;
        return Ok(Json(UnreadCountResponse { unread_count }));
    }

    let count = sqlx::query!(
        r#"
        SELECT COUNT(DISTINCT i.id) as "count!"
        FROM items i
        INNER JOIN feeds f ON i.source_id = f.source_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE f.id = $2 AND (ir.is_read IS NULL OR ir.is_read = false)
          AND ($3::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $3)
          AND ($4::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $4)
        "#,
        user_id,
        feed_id,
        params.score_min,
        params.score_max
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(UnreadCountResponse {
        unread_count: count.count,
    }))
}

pub async fn get_all_unread_count(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Query(params): Query<GetUnreadCountsQuery>,
) -> Result<Json<UnreadCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(top_n) = params.top_n {
        let unread_count =
            ranking::count_unread_top(&state, user_id, top_n, &params.ranked(), None, false)
                .await?;
        return Ok(Json(UnreadCountResponse { unread_count }));
    }
    let count = sqlx::query!(
        r#"
        SELECT COUNT(DISTINCT i.id) as "count!"
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE (f.owner_id = $1 OR fs.user_id = $1)
          AND (ir.is_read IS NULL OR ir.is_read = false)
          AND ($2::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $2)
          AND ($3::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $3)
        "#,
        user_id,
        params.score_min,
        params.score_max
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(UnreadCountResponse {
        unread_count: count.count,
    }))
}

pub async fn get_all_unread_counts(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Query(params): Query<GetUnreadCountsQuery>,
) -> Result<Json<AllUnreadCountsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let feed_counts = sqlx::query!(
        r#"
        SELECT
            f.id as feed_id,
            COUNT(i.id) FILTER (
                WHERE (ir.is_read IS NULL OR ir.is_read = false)
                  AND ($2::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $2)
                  AND ($3::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $3)
            ) as "unread_count!",
            COUNT(i.id) FILTER (WHERE ir.is_read) as "read_count!"
        FROM feeds f
        INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
        INNER JOIN sources s ON f.source_id = s.id
        LEFT JOIN items i ON s.id = i.source_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        GROUP BY f.id
        ORDER BY f.id
        "#,
        user_id,
        params.score_min,
        params.score_max
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let feeds: Vec<FeedUnreadCount> = feed_counts
        .iter()
        .map(|row| FeedUnreadCount {
            feed_id: row.feed_id,
            unread_count: row.unread_count,
            read_count: row.read_count,
        })
        .collect();

    let total_unread: i64 = feeds.iter().map(|f| f.unread_count).sum();

    Ok(Json(AllUnreadCountsResponse {
        total_unread,
        feeds,
    }))
}

pub async fn get_clusters(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Query(params): Query<GetClustersQuery>,
) -> Result<Json<Vec<ClusterResponse>>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(top_n) = params.top_n {
        // `feed_id` here is unchecked, so the ranked list stays on subscribed feeds.
        return ranking::top_clusters(&state, user_id, top_n, None, &params)
            .await
            .map(Json);
    }
    let limit = params.limit.unwrap_or(50).min(100);
    let unread_only = params.unread_only.unwrap_or(false);
    let cursor = params.cursor().map_err(bad_request)?;

    let rows = sqlx::query_as!(
        ClusterItemRow,
        r#"
        WITH anchor_items AS (
            SELECT i.id, i.cluster_id, i.published_at
            FROM items i
            INNER JOIN sources s ON i.source_id = s.id
            INNER JOIN feeds f ON f.source_id = s.id
            INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
            LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
            LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
            WHERE fs.user_id = $1
              AND ($2::uuid IS NULL OR f.id = $2)
              AND ($4::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
              AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
              AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        ),
        anchor_entities AS (
            SELECT COALESCE(cluster_id, id) AS entity_id, MAX(published_at) AS max_date
            FROM anchor_items
            GROUP BY COALESCE(cluster_id, id)
        ),
        -- Stories page by their latest matching article, the date they are shown by.
        page AS (
            SELECT entity_id, max_date FROM anchor_entities
            WHERE ($3::timestamptz IS NULL OR CASE WHEN $9::boolean
                THEN max_date > $3 OR (max_date = $3 AND entity_id > $8::uuid)
                ELSE max_date < $3 OR (max_date = $3 AND entity_id < $8::uuid) END)
            ORDER BY
                CASE WHEN $9::boolean THEN max_date END ASC,
                CASE WHEN $9::boolean THEN entity_id END ASC,
                CASE WHEN NOT $9::boolean THEN max_date END DESC,
                CASE WHEN NOT $9::boolean THEN entity_id END DESC
            LIMIT $5
        )
        SELECT
            i.id as "id!",
            array_agg(f.id) as "feed_ids!",
            i.title, i.link, i.content, i.author,
            i.published_at, i.cluster_id, i.created_at,
            ir.is_read as "is_read?",
            ROUND(ir.liked)::REAL as "liked?",
            ip.score as "prediction_score?",
            p.max_date as "story_date"
        FROM items i
        INNER JOIN page p ON p.entity_id = COALESCE(i.cluster_id, i.id)
        INNER JOIN feeds f ON f.source_id = i.source_id
        LEFT JOIN feed_subscriptions vfs ON vfs.feed_id = f.id AND vfs.user_id = $1
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE
            -- A story spans sources; show only the members this user may see.
            (f.is_public OR f.owner_id = $1 OR vfs.user_id IS NOT NULL)
            AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
            AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        GROUP BY i.id, ir.is_read, ir.liked, ip.score, p.max_date
        "#,
        user_id,
        params.feed_id,
        cursor.at,
        unread_only,
        limit,
        params.score_min,
        params.score_max,
        cursor.id,
        cursor.ascending
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    let clusters = group_story_rows(rows, cursor.ascending);
    log_impressions(
        &state,
        user_id,
        &clusters.iter().map(|c| c.id).collect::<Vec<_>>(),
    )
    .await;
    Ok(Json(clusters))
}

/// One item for the reader, including its per-user state.
pub async fn get_item(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(item_id): Path<Uuid>,
) -> Result<Json<ItemResponse>, (StatusCode, Json<ErrorResponse>)> {
    sqlx::query_as!(
        ItemResponse,
        r#"
        SELECT i.id as "id!",
               ARRAY[f.id] as "feed_ids!",
               i.title, i.link, i.content, i.author,
               i.published_at, i.cluster_id, i.created_at,
               ir.is_read as "is_read?",
               ROUND(ir.liked)::REAL as "liked?",
               ip.score as "prediction_score?"
        FROM items i
        JOIN sources s ON i.source_id = s.id
        JOIN feeds f ON f.source_id = s.id
        LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $2
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $2
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $2
        WHERE i.id = $1 AND (f.is_public OR f.owner_id = $2 OR fs.user_id = $2)
        "#,
        item_id,
        user_id,
    )
    .fetch_one(&state.db)
    .await
    .map(Json)
    .map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })
}

/// Storylines visible to the viewer, newest first.
pub async fn list_threads(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
) -> Result<Json<Vec<ThreadBriefResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let threads = sqlx::query_as!(
        ThreadBriefResponse,
        r#"
        WITH cluster_state AS (
            SELECT c.id, c.thread_id,
                   MIN(i.published_at) AS first_date,
                   MAX(i.published_at) AS last_date,
                   BOOL_AND(COALESCE(ir.is_read, false)) AS is_read,
                   (array_agg(i.title ORDER BY i.published_at))[1] AS first_title
            FROM clusters c
            JOIN items i ON i.cluster_id = c.id
            JOIN feeds f ON f.source_id = i.source_id
            LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $1
            LEFT JOIN item_reads ir ON ir.item_id = i.id AND ir.user_id = $1
            WHERE c.thread_id IS NOT NULL
              AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
            GROUP BY c.id, c.thread_id
        )
        SELECT thread_id as "id!",
               (array_agg(first_title ORDER BY first_date))[1] as "title!",
               MIN(first_date) as "first_date!",
               MAX(last_date) as "last_date!",
               COUNT(*)::bigint as "story_count!",
               COUNT(*) FILTER (WHERE NOT is_read)::bigint as "unread_count!"
        FROM cluster_state
        GROUP BY thread_id
        ORDER BY MAX(last_date) DESC
        LIMIT 50
        "#,
        user_id,
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;
    Ok(Json(threads))
}

async fn thread_response(
    db: &sqlx::PgPool,
    thread_id: Uuid,
    user_id: Uuid,
) -> Result<ThreadResponse, (StatusCode, Json<ErrorResponse>)> {
    let stories = sqlx::query_as!(
        ThreadStoryResponse,
        r#"
        SELECT c.id as "id!",
               (array_agg(i.id ORDER BY i.published_at))[1] as "item_id!",
               (array_agg(i.title ORDER BY i.published_at))[1] as "title!",
               MIN(i.published_at) as "sort_date!",
               COUNT(DISTINCT i.id)::bigint as "item_count!",
               BOOL_AND(COALESCE(ir.is_read, false)) as "is_read!",
               MAX(ROUND(ir.liked))::REAL as "liked?"
        FROM clusters c
        JOIN items i ON i.cluster_id = c.id
        JOIN feeds f ON f.source_id = i.source_id
        LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $2
        LEFT JOIN item_reads ir ON ir.item_id = i.id AND ir.user_id = $2
        WHERE c.thread_id = $1
          AND (f.is_public OR f.owner_id = $2 OR fs.user_id = $2)
        GROUP BY c.id
        ORDER BY MIN(i.published_at) ASC
        "#,
        thread_id,
        user_id,
    )
    .fetch_all(db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;
    let title = stories
        .first()
        .map(|story| story.title.clone())
        .unwrap_or_default();
    Ok(ThreadResponse {
        id: thread_id,
        title,
        stories,
    })
}

/// A thread by its id, for the briefing cards.
pub async fn get_thread(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(thread_id): Path<Uuid>,
) -> Result<Json<ThreadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let found = sqlx::query_scalar!(
        r#"
        SELECT 1
        FROM clusters c
        JOIN items i ON i.cluster_id = c.id
        JOIN feeds f ON f.source_id = i.source_id
        LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $2
        WHERE c.thread_id = $1
          AND (f.is_public OR f.owner_id = $2 OR fs.user_id = $2)
        LIMIT 1
        "#,
        thread_id,
        user_id,
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;
    if found.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse { error: "Thread not found".to_string() }),
        ));
    }
    thread_response(&state.db, thread_id, user_id).await.map(Json)
}

/// The timeline of the thread containing one story.
pub async fn get_story_thread(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(story_id): Path<Uuid>,
) -> Result<Json<ThreadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let thread_id = sqlx::query_scalar!(
        r#"
        SELECT c.thread_id
        FROM clusters c
        JOIN items i ON i.cluster_id = c.id
        JOIN feeds f ON f.source_id = i.source_id
        LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $2
        WHERE c.id = $1 AND c.thread_id IS NOT NULL
          AND (f.is_public OR f.owner_id = $2 OR fs.user_id = $2)
        LIMIT 1
        "#,
        story_id,
        user_id,
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?
    .flatten();

    let thread_id = thread_id.ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Story has no thread".to_string(),
            }),
        )
    })?;
    thread_response(&state.db, thread_id, user_id).await.map(Json)
}

/// The user's cached summary of one story, if the worker has stored it.
pub async fn get_story_summary(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(story_id): Path<Uuid>,
) -> Result<Json<StorySummaryResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !state.summaries_enabled {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Summaries are not configured".into(),
            }),
        ));
    }
    let row = sqlx::query!(
        r#"
        SELECT summary, model, updated_at
        FROM story_summaries
        WHERE user_id = $1 AND story_id = $2 AND summary IS NOT NULL
        "#,
        user_id,
        story_id
    )
    .fetch_optional(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    match row {
        Some(row) => Ok(Json(StorySummaryResponse {
            summary: row.summary.unwrap_or_default(),
            model: row.model,
            updated_at: row.updated_at,
        })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Summary not ready".into(),
            }),
        )),
    }
}

/// Drop a cached summary so the worker makes a new one; its call counts daily.
pub async fn refresh_story_summary(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(story_id): Path<Uuid>,
) -> Result<StatusCode, (StatusCode, Json<ErrorResponse>)> {
    if !state.summaries_enabled {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Summaries are not configured".into(),
            }),
        ));
    }
    sqlx::query!(
        "DELETE FROM story_summaries WHERE user_id = $1 AND story_id = $2",
        user_id,
        story_id,
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    Ok(StatusCode::NO_CONTENT)
}

/// Whether the reader can show a summary, is still waiting, or cannot get one.
pub async fn get_story_summary_status(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(story_id): Path<Uuid>,
) -> Result<Json<StorySummaryStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    // The status reads the user's settings row, which the users service makes on
    // first `GET /settings`; a client that skips it must not get a 500.
    sqlx::query!(
        "INSERT INTO user_settings (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
        user_id,
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: e.to_string() }),
        )
    })?;
    let row = sqlx::query!(
        r#"
        WITH settings AS (
            SELECT llm_summaries, llm_allowed, llm_daily_limit
            FROM user_settings WHERE user_id = $1
        ),
        story AS (
            SELECT summary, model, updated_at FROM story_summaries
            WHERE user_id = $1 AND story_id = $2
        ),
        -- The text the summary worker may use, the same rule as its STORY_TEXT.
        story_texts AS (
            SELECT ft.fetched_at
            FROM items i
            JOIN item_full_texts ft ON ft.item_id = i.id AND ft.text IS NOT NULL
            JOIN feeds f ON f.source_id = i.source_id
            LEFT JOIN feed_subscriptions fs
                ON fs.feed_id = f.id AND fs.user_id = $1
            WHERE COALESCE(i.cluster_id, i.id) = $2
              AND f.full_text AND NOT f.is_public
              AND (f.owner_id = $1 OR fs.user_id = $1)
              AND NOT EXISTS (
                  SELECT 1 FROM feeds pf WHERE pf.source_id = i.source_id AND pf.is_public
              )
        ),
        eligible AS (
            SELECT EXISTS (SELECT 1 FROM story_texts) AS ok
        ),
        latest AS (
            SELECT MAX(fetched_at) AS fetched_at FROM story_texts
        ),
        usage AS (
            SELECT COUNT(*)::bigint AS used FROM llm_calls
            WHERE user_id = $1 AND created_at >= date_trunc('day', now())
        )
        SELECT
            s.llm_summaries AS "summaries!",
            s.llm_allowed AS "allowed!",
            s.llm_daily_limit AS "daily_limit!",
            (SELECT summary FROM story) AS summary,
            (SELECT model FROM story) AS model,
            (SELECT updated_at FROM story) AS updated_at,
            e.ok AS "eligible!",
            COALESCE(l.fetched_at > (SELECT updated_at FROM story), false) AS "stale!",
            u.used AS "used!"
        FROM settings s
        CROSS JOIN eligible e
        CROSS JOIN latest l
        CROSS JOIN usage u
        "#,
        user_id,
        story_id,
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let applicable = state.summaries_enabled
        && row.summaries
        && row.allowed
        && row.eligible;
    let ready = row.summary.is_some();
    Ok(Json(StorySummaryStatusResponse {
        status: if !applicable {
            "not-applicable"
        } else if ready {
            "ready"
        } else {
            "pending"
        }
        .to_string(),
        summary: row.summary,
        model: row.model,
        updated_at: row.updated_at,
        can_recreate: applicable && ready && row.stale,
        used_today: row.used,
        daily_limit: row.daily_limit,
    }))
}

pub async fn update_cluster_status(    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(id): Path<Uuid>,
    Json(payload): Json<UpdateItemStatusRequest>,
) -> Result<Json<ClusterStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = sqlx::query!(
        r#"
        WITH target_items AS (
            SELECT i.id
            FROM items i
            JOIN sources s ON i.source_id = s.id
            JOIN feeds f ON f.source_id = s.id
            LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $1
            WHERE (i.cluster_id = $2 OR (i.cluster_id IS NULL AND i.id = $2))
              AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        )
        INSERT INTO item_reads (user_id, item_id, is_read, liked, marked_at)
        SELECT $1, id, COALESCE($3::BOOLEAN, true), $4::REAL, CURRENT_TIMESTAMP
        FROM target_items
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET
            is_read = COALESCE($3::BOOLEAN, item_reads.is_read),
            liked = CASE WHEN $5::BOOLEAN THEN $4::REAL ELSE item_reads.liked END,
            marked_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        id,
        payload.is_read,
        payload.liked.flatten(),
        payload.liked.is_some()
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(ClusterStatusResponse {
        updated_items: result.rows_affected() as i64,
        is_read: payload.is_read.unwrap_or(true),
        liked: payload.liked.flatten(),
    }))
}

pub async fn get_feed_cluster_unread_count(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(feed_id): Path<Uuid>,
    Query(params): Query<GetClustersQuery>,
) -> Result<Json<UnreadCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Feed not found".into(),
            }),
        ));
    }

    if let Some(top_n) = params.top_n {
        let unread_count =
            ranking::count_unread_top(&state, user_id, top_n, &params, Some(feed_id), true)
                .await?;
        return Ok(Json(UnreadCountResponse { unread_count }));
    }

    let count = sqlx::query!(
        r#"
        SELECT COUNT(DISTINCT COALESCE(i.cluster_id, i.id)) as "count!"
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE f.id = $2
          AND (ir.is_read IS NULL OR ir.is_read = false)
          AND ($3::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $3)
          AND ($4::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $4)
        "#,
        user_id,
        feed_id,
        params.score_min,
        params.score_max
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(UnreadCountResponse {
        unread_count: count.count,
    }))
}

pub async fn get_cluster_unread_count(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Query(params): Query<GetClustersQuery>,
) -> Result<Json<UnreadCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(top_n) = params.top_n {
        let unread_count =
            ranking::count_unread_top(&state, user_id, top_n, &params, None, true).await?;
        return Ok(Json(UnreadCountResponse { unread_count }));
    }
    let count = sqlx::query!(
        r#"
        SELECT COUNT(DISTINCT COALESCE(i.cluster_id, i.id)) as "count!"
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE (f.owner_id = $1 OR fs.user_id = $1)
          AND (ir.is_read IS NULL OR ir.is_read = false)
          AND ($2::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $2)
          AND ($3::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $3)
        "#,
        user_id,
        params.score_min,
        params.score_max
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(UnreadCountResponse {
        unread_count: count.count,
    }))
}

pub async fn get_feed_clusters(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(feed_id): Path<Uuid>,
    Query(params): Query<GetClustersQuery>,
) -> Result<Json<Vec<ClusterResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Feed not found".into(),
            }),
        ));
    }

    if let Some(top_n) = params.top_n {
        return ranking::top_clusters(&state, user_id, top_n, Some(feed_id), &params)
            .await
            .map(Json);
    }

    let limit = params.limit.unwrap_or(50).min(100);
    let unread_only = params.unread_only.unwrap_or(false);
    let cursor = params.cursor().map_err(bad_request)?;

    let rows = sqlx::query_as!(
        ClusterItemRow,
        r#"
        WITH anchor_items AS (
            SELECT i.id, i.cluster_id, i.published_at
            FROM items i
            INNER JOIN feeds f ON i.source_id = f.source_id
            LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
            LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
            WHERE f.id = $2
              AND ($4::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
              AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
              AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        ),
        anchor_entities AS (
            SELECT COALESCE(cluster_id, id) AS entity_id, MAX(published_at) AS max_date
            FROM anchor_items
            GROUP BY COALESCE(cluster_id, id)
        ),
        -- Stories page by their latest matching article, the date they are shown by.
        page AS (
            SELECT entity_id, max_date FROM anchor_entities
            WHERE ($3::timestamptz IS NULL OR CASE WHEN $9::boolean
                THEN max_date > $3 OR (max_date = $3 AND entity_id > $8::uuid)
                ELSE max_date < $3 OR (max_date = $3 AND entity_id < $8::uuid) END)
            ORDER BY
                CASE WHEN $9::boolean THEN max_date END ASC,
                CASE WHEN $9::boolean THEN entity_id END ASC,
                CASE WHEN NOT $9::boolean THEN max_date END DESC,
                CASE WHEN NOT $9::boolean THEN entity_id END DESC
            LIMIT $5
        )
        SELECT
            i.id as "id!",
            array_agg(f.id) as "feed_ids!",
            i.title, i.link, i.content, i.author,
            i.published_at, i.cluster_id, i.created_at,
            ir.is_read as "is_read?",
            ROUND(ir.liked)::REAL as "liked?",
            ip.score as "prediction_score?",
            p.max_date as "story_date"
        FROM items i
        INNER JOIN page p ON p.entity_id = COALESCE(i.cluster_id, i.id)
        INNER JOIN feeds f ON f.source_id = i.source_id
        LEFT JOIN feed_subscriptions vfs ON vfs.feed_id = f.id AND vfs.user_id = $1
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE
            -- A story spans sources; show only the members this user may see.
            (f.is_public OR f.owner_id = $1 OR vfs.user_id IS NOT NULL)
            AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
            AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        GROUP BY i.id, ir.is_read, ir.liked, ip.score, p.max_date
        "#,
        user_id,
        feed_id,
        cursor.at,
        unread_only,
        limit,
        params.score_min,
        params.score_max,
        cursor.id,
        cursor.ascending
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    let clusters = group_story_rows(rows, cursor.ascending);
    log_impressions(
        &state,
        user_id,
        &clusters.iter().map(|c| c.id).collect::<Vec<_>>(),
    )
    .await;
    Ok(Json(clusters))
}

pub async fn mark_feed_clusters_read(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(feed_id): Path<Uuid>,
    Json(payload): Json<MarkAllReadRequest>,
) -> Result<Json<MarkAllReadResponse>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.is_public OR f.owner_id = $1 OR fs.user_id = $1)
        ) as "has_access!"
        "#,
        user_id,
        feed_id
    )
    .fetch_one(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    if !has_access.has_access {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Feed not found".into(),
            }),
        ));
    }

    let result = sqlx::query!(
        r#"
        WITH target_scope AS (
            SELECT i.cluster_id, i.id
            FROM items i
            INNER JOIN feeds f ON i.source_id = f.source_id
            WHERE f.id = $2 AND i.published_at >= $3
              AND ($4::TIMESTAMPTZ IS NULL OR i.published_at < $4)
        )
        INSERT INTO item_reads (user_id, item_id, is_read, liked, marked_at)
        SELECT DISTINCT $1::uuid, i.id, true, NULL::REAL, CURRENT_TIMESTAMP
        FROM items i
        WHERE
            (i.cluster_id IS NULL AND i.id IN (SELECT id FROM target_scope))
            OR
            (i.cluster_id IS NOT NULL AND i.cluster_id IN (SELECT cluster_id FROM target_scope))
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET is_read = true, marked_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        feed_id,
        payload.since,
        payload.before
    )
    .execute(&state.db)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(MarkAllReadResponse {
        marked_count: result.rows_affected() as i64,
    }))
}

/// The story of each item: its cluster, or the item itself.
pub(crate) fn story_ids(items: &[ItemResponse]) -> Vec<Uuid> {
    items
        .iter()
        .map(|item| item.cluster_id.unwrap_or(item.id))
        .collect()
}

/// Records the stories shown in a list. Best effort: a failed insert must not
/// fail the read.
pub(crate) async fn log_impressions(state: &AppState, user_id: Uuid, story_ids: &[Uuid]) {
    if story_ids.is_empty() {
        return;
    }
    let result = sqlx::query!(
        r#"
        INSERT INTO story_impressions (user_id, story_id)
        SELECT $1, story_id FROM UNNEST($2::uuid[]) AS t(story_id)
        ON CONFLICT (user_id, story_id) DO NOTHING
        "#,
        user_id,
        story_ids
    )
    .execute(&state.db)
    .await;
    if let Err(e) = result {
        eprintln!("story impressions: {e}");
    }
}

/// Groups member rows of a story list, keyed by the date each story was paged by.
fn group_story_rows(rows: Vec<ClusterItemRow>, ascending: bool) -> Vec<ClusterResponse> {
    let mut dates = HashMap::new();
    let items = rows
        .into_iter()
        .map(|row| {
            let (item, date) = row.split();
            if let Some(date) = date {
                dates.insert(item.cluster_id.unwrap_or(item.id), date);
            }
            item
        })
        .collect();
    group_clusters(items, &dates, ascending)
}

/// Groups item rows into story cards. A card's `sort_date` is its paging date
/// from `dates`, so the client's cursor matches the server's; the latest member
/// stands in for a story without one. Singletons become one-item cards.
pub fn group_clusters(
    rows: Vec<ItemResponse>,
    dates: &HashMap<Uuid, time::OffsetDateTime>,
    ascending: bool,
) -> Vec<ClusterResponse> {
    let mut stories: HashMap<Uuid, Vec<ItemResponse>> = HashMap::new();
    let mut ids: Vec<(Uuid, bool)> = Vec::new();
    for item in rows {
        let key = item.cluster_id.unwrap_or(item.id);
        let members = stories.entry(key).or_default();
        if members.is_empty() {
            ids.push((key, item.cluster_id.is_some()));
        }
        members.push(item);
    }

    let mut result: Vec<ClusterResponse> = ids
        .into_iter()
        .map(|(id, is_cluster)| {
            let mut items = stories.remove(&id).unwrap_or_default();
            sort_items(&mut items, ascending);
            let sort_date = dates
                .get(&id)
                .copied()
                .or_else(|| items.iter().filter_map(|i| i.published_at).max());
            ClusterResponse {
                id,
                is_cluster,
                sort_date,
                explored: false,
                items,
            }
        })
        .collect();

    result.sort_by(|a, b| {
        let key = |c: &ClusterResponse| (c.sort_date, c.id);
        if ascending {
            key(a).cmp(&key(b))
        } else {
            key(b).cmp(&key(a))
        }
    });
    result
}

fn sort_items(items: &mut [ItemResponse], ascending: bool) {
    items.sort_by(|a, b| {
        let key = |i: &ItemResponse| (i.published_at, i.id);
        if ascending {
            key(a).cmp(&key(b))
        } else {
            key(b).cmp(&key(a))
        }
    });
}
