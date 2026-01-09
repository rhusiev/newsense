use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    routing::post,
};
use chrono::{DateTime, Utc};
use feed_rs::parser;
use log::{error, info, warn};
use serde::Serialize;
use sqlx::{PgPool, postgres::PgPoolOptions};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::{Semaphore, mpsc};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    db: PgPool,
    queue_tx: mpsc::Sender<Uuid>,
}

#[derive(Clone)]
struct SourceToFetch {
    id: Uuid,
    url: String,
    last_fetched_at: Option<DateTime<Utc>>,
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://user:pass@localhost/authdb".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await
        .expect("Failed to connect to database");

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("Mozilla/5.0 (compatible; RSSReader/1.0)")
        .build()
        .expect("Failed to create HTTP client");

    let (tx, rx) = mpsc::channel::<Uuid>(1000);

    let app_state = AppState {
        db: pool.clone(),
        queue_tx: tx.clone(),
    };

    tokio::spawn(worker_task(pool.clone(), http_client, rx));

    tokio::spawn(scheduler_task(pool.clone(), tx.clone()));

    let app = Router::new()
        .route("/feeds/{feed_id}/refresh", post(trigger_refresh))
        .with_state(app_state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 3003));
    info!("Fetcher API running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[derive(Serialize)]
struct MessageResponse {
    message: String,
    feed_id: Uuid,
}

async fn trigger_refresh(
    State(state): State<AppState>,
    Path(feed_id): Path<Uuid>,
) -> Result<Json<MessageResponse>, (StatusCode, String)> {
    let record = sqlx::query!("SELECT source_id FROM feeds WHERE id = $1", feed_id)
        .fetch_optional(&state.db)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if let Some(r) = record {
        match state.queue_tx.send(r.source_id).await {
            Ok(_) => Ok(Json(MessageResponse {
                message: "Source queued for refresh".to_string(),
                feed_id,
            })),
            Err(_) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Worker queue is closed".to_string(),
            )),
        }
    } else {
        Err((StatusCode::NOT_FOUND, "Feed not found".to_string()))
    }
}

async fn scheduler_task(pool: PgPool, tx: mpsc::Sender<Uuid>) {
    let fetch_interval_secs = std::env::var("FETCH_INTERVAL_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    info!("Scheduler started. Interval: {}s", fetch_interval_secs);

    loop {
        let sources = sqlx::query!(
            r#"
            SELECT id
            FROM sources
            WHERE error_count < 5
              AND (
                  last_fetched_at IS NULL
                  OR last_fetched_at < NOW() - make_interval(secs => $1)
              )
            LIMIT 100
            "#,
            fetch_interval_secs as f64
        )
        .fetch_all(&pool)
        .await;

        match sources {
            Ok(rows) => {
                if !rows.is_empty() {
                    info!("Scheduler: Queueing {} sources for update", rows.len());
                    for row in rows {
                        if let Err(e) = tx.send(row.id).await {
                            error!("Scheduler failed to send to queue: {}", e);
                            break;
                        }
                    }
                }
            }
            Err(e) => error!("Scheduler DB error: {}", e),
        }

        tokio::time::sleep(Duration::from_secs(fetch_interval_secs)).await;
    }
}

async fn worker_task(pool: PgPool, http_client: reqwest::Client, mut rx: mpsc::Receiver<Uuid>) {
    let semaphore = Arc::new(Semaphore::new(5));

    info!("Worker started. Waiting for jobs...");

    while let Some(feed_id) = rx.recv().await {
        let pool = pool.clone();
        let client = http_client.clone();
        let permit = semaphore.clone().acquire_owned().await.unwrap();

        tokio::spawn(async move {
            let _permit = permit;

            if let Err(e) = process_single_feed(&pool, &client, feed_id).await {
                error!("Failed to process feed {}: {}", feed_id, e);
            }
        });
    }
}

async fn process_single_feed(
    pool: &PgPool,
    client: &reqwest::Client,
    source_id: Uuid,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let source_info = sqlx::query_as!(
        SourceToFetch,
        "SELECT id, url, last_fetched_at FROM sources WHERE id = $1",
        source_id
    )
    .fetch_optional(pool)
    .await?;

    let source_info = match source_info {
        Some(f) => f,
        None => {
            warn!("Worker received unknown source_id: {}", source_id);
            return Ok(());
        }
    };

    info!("Worker processing: {}", source_info.url);

    let processing_result = async {
        let response = client
            .get(&source_info.url)
            .header("Accept", "*/*")
            .header("Connection", "close")
            .send()
            .await?;

        let content_bytes = response.bytes().await?;

        let feed = tokio::task::spawn_blocking(move || {
            let parser = parser::Builder::new().sanitize_content(true).build();
            parser.parse(&content_bytes[..])
        })
        .await??;

        for entry in feed.entries {
            let title = entry
                .title
                .map(|t| t.content)
                .unwrap_or_else(|| "Untitled".to_string());
            let link = entry
                .links
                .first()
                .map(|l| l.href.clone())
                .unwrap_or_default();
            if link.is_empty() {
                continue;
            }

            let content = entry
                .summary
                .map(|t| t.content)
                .or_else(|| entry.content.and_then(|c| c.body));
            let author = entry.authors.first().map(|a| a.name.clone());
            let published_at = entry.published.or(entry.updated).unwrap_or_else(Utc::now);

            sqlx::query!(
                r#"
                INSERT INTO items (source_id, title, link, content, author, published_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source_id, link) DO NOTHING
                "#,
                source_info.id,
                title,
                link,
                content,
                author,
                published_at,
            )
            .execute(pool)
            .await?;
        }
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    }
    .await;

    match processing_result {
        Ok(_) => {
            sqlx::query!(
                "UPDATE sources SET last_fetched_at = $1, error_count = 0 WHERE id = $2",
                Utc::now(),
                source_info.id
            )
            .execute(pool)
            .await?;
            Ok(())
        }
        Err(e) => {
            error!("Error fetching source {}: {}", source_info.url, e);
            sqlx::query!(
                "UPDATE sources SET error_count = error_count + 1 WHERE id = $1",
                source_info.id
            )
            .execute(pool)
            .await?;
            Err(e)
        }
    }
}
