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
        FeedUnreadCount, GetClustersQuery, GetItemsQuery, ItemResponse, MarkAllReadRequest,
        MarkAllReadResponse, ReadStatusResponse, UnreadCountResponse, UpdateItemStatusRequest,
    },
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
            WHERE f.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
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
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
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
                    ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
                    ip.score as "prediction_score?"
                FROM items i
                INNER JOIN feeds f ON i.source_id = f.source_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE f.id = $2 AND i.published_at > $3
                ORDER BY i.published_at ASC
                LIMIT $4
                "#,
                user_id,
                feed_id,
                since,
                limit
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
                    ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
                    ip.score as "prediction_score?"
                FROM items i
                INNER JOIN feeds f ON i.source_id = f.source_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE f.id = $2 AND i.published_at < $3
                ORDER BY i.published_at DESC
                LIMIT $4
                "#,
                user_id,
                feed_id,
                before,
                limit
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
                    ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
                    ip.score as "prediction_score?"
                FROM items i
                INNER JOIN feeds f ON i.source_id = f.source_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE f.id = $2 AND (ir.is_read IS NULL OR ir.is_read = false)
                ORDER BY i.published_at DESC
                LIMIT $3
                "#,
                user_id,
                feed_id,
                limit
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
                    ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
                    ip.score as "prediction_score?"
                FROM items i
                INNER JOIN feeds f ON i.source_id = f.source_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE f.id = $2
                ORDER BY i.published_at DESC
                LIMIT $3
                "#,
                user_id,
                feed_id,
                limit
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

    Ok(Json(items))
}

pub async fn get_all_items(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Query(params): Query<GetItemsQuery>,
) -> Result<Json<Vec<ItemResponse>>, (StatusCode, Json<ErrorResponse>)> {
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
                    ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
                    ip.score as "prediction_score?"
                FROM items i
                INNER JOIN sources s ON i.source_id = s.id
                INNER JOIN feeds f ON f.source_id = s.id
                INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE fs.user_id = $1 AND i.published_at < $2
                GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
                ORDER BY i.published_at DESC
                LIMIT $3
                "#,
                user_id, before, limit
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
                    ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
                    ip.score as "prediction_score?"
                FROM items i
                INNER JOIN sources s ON i.source_id = s.id
                INNER JOIN feeds f ON f.source_id = s.id
                INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE fs.user_id = $1 AND (ir.is_read IS NULL OR ir.is_read = false)
                GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
                ORDER BY i.published_at DESC
                LIMIT $2
                "#,
                user_id, limit
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
                    ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
                    ip.score as "prediction_score?"
                FROM items i
                INNER JOIN sources s ON i.source_id = s.id
                INNER JOIN feeds f ON f.source_id = s.id
                INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id
                LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
                LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
                WHERE fs.user_id = $1
                GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
                ORDER BY i.published_at DESC
                LIMIT $2
                "#,
                user_id, limit
            ).fetch_all(&state.db).await
        }
    }
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

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
            WHERE i.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
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
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
    }

    let status = sqlx::query_as!(
        ReadStatusResponse,
        r#"
        INSERT INTO item_reads (user_id, item_id, is_read, liked)
        VALUES ($1, $2, COALESCE($3::BOOLEAN, true), COALESCE($4::REAL, 0.0))
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET
            is_read = COALESCE($3::BOOLEAN, item_reads.is_read),
            liked = COALESCE($4::REAL, item_reads.liked),
            marked_at = CURRENT_TIMESTAMP
        RETURNING
            user_id, item_id, is_read, ROUND(liked)::REAL as "liked!", marked_at
        "#,
        user_id,
        item_id,
        payload.is_read,
        payload.liked
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
            WHERE f.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
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
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
    }

    let result = sqlx::query!(
        r#"
        INSERT INTO item_reads (user_id, item_id, is_read, liked, marked_at)
        SELECT $1, i.id, true, 0.0, CURRENT_TIMESTAMP
        FROM items i
        INNER JOIN feeds f ON i.source_id = f.source_id
        WHERE f.id = $2 AND i.published_at >= $3
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET is_read = true, marked_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        feed_id,
        payload.since
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
        SELECT DISTINCT $1::UUID, i.id, true, 0.0, CURRENT_TIMESTAMP
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id
        WHERE (f.owner_id = $1 OR fs.user_id = $1)
          AND i.published_at >= $2
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET is_read = true, marked_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        payload.since
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
) -> Result<Json<UnreadCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    let has_access = sqlx::query!(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM feeds f
            LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
            WHERE f.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
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
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
    }

    let count = sqlx::query!(
        r#"
        SELECT COUNT(DISTINCT i.id) as "count!"
        FROM items i
        INNER JOIN feeds f ON i.source_id = f.source_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        WHERE f.id = $2 AND (ir.is_read IS NULL OR ir.is_read = false)
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

    Ok(Json(UnreadCountResponse {
        unread_count: count.count,
    }))
}

pub async fn get_all_unread_count(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
) -> Result<Json<UnreadCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    let count = sqlx::query!(
        r#"
        SELECT COUNT(DISTINCT i.id) as "count!"
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        WHERE (f.owner_id = $1 OR fs.user_id = $1)
          AND (ir.is_read IS NULL OR ir.is_read = false)
        "#,
        user_id
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
) -> Result<Json<AllUnreadCountsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let feed_counts = sqlx::query!(
        r#"
        SELECT
            f.id as feed_id,
            COUNT(i.id) FILTER (WHERE ir.is_read IS NULL OR ir.is_read = false) as "unread_count!"
        FROM feeds f
        INNER JOIN feed_subscriptions fs ON f.id = fs.feed_id AND fs.user_id = $1
        INNER JOIN sources s ON f.source_id = s.id
        LEFT JOIN items i ON s.id = i.source_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        GROUP BY f.id
        ORDER BY f.id
        "#,
        user_id
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
            WHERE fs.user_id = $1
              AND ($2::uuid IS NULL OR f.id = $2)
              AND ($3::timestamptz IS NULL OR i.published_at < $3)
              AND ($4::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
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
            ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
            ip.score as "prediction_score?"
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE
            COALESCE(i.cluster_id, i.id) IN (SELECT entity_id FROM anchor_entities)
        GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
        ORDER BY i.published_at DESC
        "#,
        user_id,
        params.feed_id,
        params.before,
        unread_only,
        limit
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    let mut cluster_map: HashMap<Uuid, Vec<ItemResponse>> = HashMap::new();
    let mut singletons: Vec<ItemResponse> = Vec::new();

    for item in rows {
        if let Some(cid) = item.cluster_id {
            cluster_map.entry(cid).or_default().push(item);
        } else {
            singletons.push(item);
        }
    }

    let mut result: Vec<ClusterResponse> = Vec::new();

    for (cid, items) in cluster_map {
        let mut sorted_items = items;
        sorted_items.sort_by(|a, b| b.published_at.cmp(&a.published_at));

        let max_date = sorted_items.first().and_then(|i| i.published_at);

        result.push(ClusterResponse {
            id: cid,
            is_cluster: true,
            sort_date: max_date,
            items: sorted_items,
        });
    }

    for item in singletons {
        result.push(ClusterResponse {
            id: item.id,
            is_cluster: false,
            sort_date: item.published_at,
            items: vec![item],
        });
    }

    result.sort_by(|a, b| b.sort_date.cmp(&a.sort_date));

    Ok(Json(result))
}

pub async fn update_cluster_status(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
    Path(id): Path<Uuid>, // cluster_id or item_id
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
        SELECT $1, id, COALESCE($3::BOOLEAN, true), COALESCE($4::REAL, 0.0), CURRENT_TIMESTAMP
        FROM target_items
        ON CONFLICT (user_id, item_id)
        DO UPDATE SET
            is_read = COALESCE($3::BOOLEAN, item_reads.is_read),
            liked = COALESCE($4::REAL, item_reads.liked),
            marked_at = CURRENT_TIMESTAMP
        "#,
        user_id,
        id,
        payload.is_read,
        payload.liked
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
        liked: payload.liked.unwrap_or(0.0),
    }))
}

pub async fn get_cluster_unread_count(
    State(state): State<AppState>,
    AuthUser { id: user_id, .. }: AuthUser,
) -> Result<Json<UnreadCountResponse>, (StatusCode, Json<ErrorResponse>)> {
    let count = sqlx::query!(
        r#"
        SELECT COUNT(DISTINCT COALESCE(i.cluster_id, i.id)) as "count!"
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN feed_subscriptions fs ON f.id = fs.feed_id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        WHERE (f.owner_id = $1 OR fs.user_id = $1)
          AND (ir.is_read IS NULL OR ir.is_read = false)
        "#,
        user_id
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
            WHERE f.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
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
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
            }),
        ));
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
            WHERE f.id = $2
              AND ($3::timestamptz IS NULL OR i.published_at < $3)
              AND ($4::boolean = false OR (ir.is_read IS NULL OR ir.is_read = false))
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
            ROUND(COALESCE(ir.liked, 0.0))::REAL as "liked?",
            ip.score as "prediction_score?"
        FROM items i
        INNER JOIN sources s ON i.source_id = s.id
        INNER JOIN feeds f ON f.source_id = s.id
        LEFT JOIN item_reads ir ON i.id = ir.item_id AND ir.user_id = $1
        LEFT JOIN item_predictions ip ON i.id = ip.item_id AND ip.user_id = $1
        WHERE
            COALESCE(i.cluster_id, i.id) IN (SELECT entity_id FROM anchor_entities)
        GROUP BY i.id, i.title, i.link, i.content, i.author, i.published_at, i.cluster_id, i.created_at, ir.is_read, ir.liked, ip.score
        ORDER BY i.published_at DESC
        "#,
        user_id,
        feed_id,
        params.before,
        unread_only,
        limit
    )
    .fetch_all(&state.db)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;

    let mut cluster_map: HashMap<Uuid, Vec<ItemResponse>> = HashMap::new();
    let mut singletons: Vec<ItemResponse> = Vec::new();

    for item in rows {
        if let Some(cid) = item.cluster_id {
            cluster_map.entry(cid).or_default().push(item);
        } else {
            singletons.push(item);
        }
    }

    let mut result: Vec<ClusterResponse> = Vec::new();

    for (cid, items) in cluster_map {
        let mut sorted_items = items;
        sorted_items.sort_by(|a, b| b.published_at.cmp(&a.published_at));
        let max_date = sorted_items.first().and_then(|i| i.published_at);

        result.push(ClusterResponse {
            id: cid,
            is_cluster: true,
            sort_date: max_date,
            items: sorted_items,
        });
    }

    for item in singletons {
        result.push(ClusterResponse {
            id: item.id,
            is_cluster: false,
            sort_date: item.published_at,
            items: vec![item],
        });
    }

    result.sort_by(|a, b| b.sort_date.cmp(&a.sort_date));

    Ok(Json(result))
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
            WHERE f.id = $2 AND (f.owner_id = $1 OR fs.user_id = $1)
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
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "Access denied".into(),
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
        )
        INSERT INTO item_reads (user_id, item_id, is_read, liked, marked_at)
        SELECT DISTINCT $1::uuid, i.id, true, 0.0, CURRENT_TIMESTAMP
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
        payload.since
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
