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
        AllUnreadCountsResponse, ClusterResponse, ClusterStatusResponse, ErrorResponse,
        FeedUnreadCount, GetClustersQuery, GetItemsQuery, GetUnreadCountsQuery, ItemResponse,
        MarkAllReadRequest, MarkAllReadResponse, ReadStatusResponse, StorySummaryResponse,
        UnreadCountResponse, UpdateItemStatusRequest,
    },
    ranking,
    state::AppState,
};

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

    let items = match (params.since, params.before, unread_only) {
        (Some(since), _, false) => {
            sqlx::query_as!(
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
                WHERE f.id = $2 AND i.published_at > $3
                  AND ($5::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $5)
                  AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $6)
                ORDER BY i.published_at ASC
                LIMIT $4
                "#,
                user_id,
                feed_id,
                since,
                limit,
                params.score_min,
                params.score_max
            )
            .fetch_all(&state.db)
            .await
        }
        (_, Some(before), false) => {
            sqlx::query_as!(
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
                WHERE f.id = $2 AND i.published_at < $3
                  AND ($5::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $5)
                  AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $6)
                ORDER BY i.published_at DESC
                LIMIT $4
                "#,
                user_id,
                feed_id,
                before,
                limit,
                params.score_min,
                params.score_max
            )
            .fetch_all(&state.db)
            .await
        }
        (_, _, true) => {
            sqlx::query_as!(
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
                WHERE f.id = $2 AND (ir.is_read IS NULL OR ir.is_read = false)
                  AND ($4::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $4)
                  AND ($5::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $5)
                ORDER BY i.published_at DESC
                LIMIT $3
                "#,
                user_id,
                feed_id,
                limit,
                params.score_min,
                params.score_max
            )
            .fetch_all(&state.db)
            .await
        }
        _ => {
            sqlx::query_as!(
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
                  AND ($4::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $4)
                  AND ($5::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $5)
                ORDER BY i.published_at DESC
                LIMIT $3
                "#,
                user_id,
                feed_id,
                limit,
                params.score_min,
                params.score_max
            )
            .fetch_all(&state.db)
            .await
        }
    }
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

    let items = match (params.before, unread_only) {
        (Some(before), false) => {
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
                INNER JOIN sources s ON i.source_id = s.id
                INNER JOIN feeds f ON f.source_id = s.id
                INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE fs.user_id = $1 AND i.published_at < $2
                  AND ($4::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $4)
                  AND ($5::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $5)
                GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
                ORDER BY i.published_at DESC
                LIMIT $3
                "#,
                user_id, before, limit, params.score_min, params.score_max
            ).fetch_all(&state.db).await
        }
        (_, true) => {
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
                INNER JOIN sources s ON i.source_id = s.id
                INNER JOIN feeds f ON f.source_id = s.id
                INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE fs.user_id = $1 AND (ir.is_read IS NULL OR ir.is_read = false)
                  AND ($3::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $3)
                  AND ($4::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $4)
                GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
                ORDER BY i.published_at DESC
                LIMIT $2
                "#,
                user_id, limit, params.score_min, params.score_max
            ).fetch_all(&state.db).await
        }
        _ => {
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
                INNER JOIN sources s ON i.source_id = s.id
                INNER JOIN feeds f ON f.source_id = s.id
                INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE fs.user_id = $1
                  AND ($3::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $3)
                  AND ($4::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $4)
                GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
                ORDER BY i.published_at DESC
                LIMIT $2
                "#,
                user_id, limit, params.score_min, params.score_max
            ).fetch_all(&state.db).await
        }
    }
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

    let rows = sqlx::query_as!(
        ItemResponse,
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
              AND ($3::timestamptz IS NULL OR i.published_at < $3)
              AND ($4::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
              AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
              AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        ),
        anchor_entities AS (
            SELECT
                COALESCE(cluster_id, id) as entity_id,
                MAX(published_at) as max_date
            FROM anchor_items
            GROUP BY COALESCE(cluster_id, id)
            ORDER BY max_date DESC
            LIMIT $5
        )
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
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE
            COALESCE(i.cluster_id, i.id) IN (SELECT entity_id FROM anchor_entities)
            AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
            AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
        ORDER BY i.published_at DESC
        "#,
        user_id,
        params.feed_id,
        params.before,
        unread_only,
        limit,
        params.score_min,
        params.score_max
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    let clusters = group_clusters(rows);
    log_impressions(
        &state,
        user_id,
        &clusters.iter().map(|c| c.id).collect::<Vec<_>>(),
    )
    .await;
    Ok(Json(clusters))
}

/// The user's cached summary of one story, if the worker has stored it.
pub async fn get_story_summary(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(story_id): Path<Uuid>,
) -> Result<Json<StorySummaryResponse>, (StatusCode, Json<ErrorResponse>)> {
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
            WHERE i.cluster_id = $2 OR (i.cluster_id IS NULL AND i.id = $2)
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

    let rows = sqlx::query_as!(
        ItemResponse,
        r#"
        WITH anchor_items AS (
            SELECT i.id, i.cluster_id, i.published_at
            FROM items i
            INNER JOIN feeds f ON i.source_id = f.source_id
            LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
            LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
            WHERE f.id = $2
              AND ($3::timestamptz IS NULL OR i.published_at < $3)
              AND ($4::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
              AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
              AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        ),
        anchor_entities AS (
            SELECT
                COALESCE(cluster_id, id) as entity_id,
                MAX(published_at) as max_date
            FROM anchor_items
            GROUP BY COALESCE(cluster_id, id)
            ORDER BY max_date DESC
            LIMIT $5
        )
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
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE
            COALESCE(i.cluster_id, i.id) IN (SELECT entity_id FROM anchor_entities)
            AND ($6::REAL IS NULL OR COALESCE(ip.score, 0.0) >= $6)
            AND ($7::REAL IS NULL OR COALESCE(ip.score, 0.0) <= $7)
        GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
        ORDER BY i.published_at DESC
        "#,
        user_id,
        feed_id,
        params.before,
        unread_only,
        limit,
        params.score_min,
        params.score_max
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    let clusters = group_clusters(rows);
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

/// Groups item rows into cluster cards, newest first. Singletons become one-item cards.
pub fn group_clusters(rows: Vec<ItemResponse>) -> Vec<ClusterResponse> {
    let mut cluster_map: HashMap<Uuid, Vec<ItemResponse>> = HashMap::new();
    let mut result: Vec<ClusterResponse> = Vec::new();

    for item in rows {
        match item.cluster_id {
            Some(cid) => cluster_map.entry(cid).or_default().push(item),
            None => result.push(ClusterResponse {
                id: item.id,
                is_cluster: false,
                sort_date: item.published_at,
                explored: false,
                items: vec![item],
            }),
        }
    }

    for (cid, mut items) in cluster_map {
        items.sort_by(|a, b| b.published_at.cmp(&a.published_at));
        result.push(ClusterResponse {
            id: cid,
            is_cluster: true,
            sort_date: items.first().and_then(|i| i.published_at),
            explored: false,
            items,
        });
    }

    result.sort_by(|a, b| b.sort_date.cmp(&a.sort_date));
    result
}
