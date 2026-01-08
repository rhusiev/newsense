use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::post,
    Json, Router,
};
use chrono::{DateTime, Utc};
use feed_rs::parser;
use log::{error, info, warn};
use serde::Serialize;
use sqlx::{postgres::PgPoolOptions, PgPool};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::sync::{mpsc, Semaphore};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    db: PgPool,
    queue_tx: mpsc::Sender<Uuid>,
}

#[derive(Clone)]
struct FeedToFetch {
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
    match state.queue_tx.send(feed_id).await {
        Ok(_) => Ok(Json(MessageResponse {
            message: "Feed queued for refresh".to_string(),
            feed_id,
        })),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Worker queue is closed".to_string(),
        )),
    }
}

async fn scheduler_task(pool: PgPool, tx: mpsc::Sender<Uuid>) {
    let fetch_interval_secs = std::env::var("FETCH_INTERVAL_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    info!("Scheduler started. Interval: {}s", fetch_interval_secs);

    loop {
        let feeds = sqlx::query!(
            r#"
            SELECT id
            FROM feeds
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

        match feeds {
            Ok(rows) => {
                if !rows.is_empty() {
                    info!("Scheduler: Queueing {} feeds for update", rows.len());
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

async fn worker_task(
    pool: PgPool, 
    http_client: reqwest::Client, 
    mut rx: mpsc::Receiver<Uuid>
) {
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
    feed_id: Uuid,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let feed_info = sqlx::query_as!(
        FeedToFetch,
        "SELECT id, url, last_fetched_at FROM feeds WHERE id = $1",
        feed_id
    )
    .fetch_optional(pool)
    .await?;

    let feed_info = match feed_info {
        Some(f) => f,
        None => {
            warn!("Worker received unknown feed_id: {}", feed_id);
            return Ok(());
        }
    };

    info!("Worker processing: {}", feed_info.url);

    let processing_result = async {
        let response = client
            .get(&feed_info.url)
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
            let title = entry.title.map(|t| t.content).unwrap_or_else(|| "Untitled".to_string());
            let link = entry.links.first().map(|l| l.href.clone()).unwrap_or_default();
            if link.is_empty() { continue; }

            let content = entry.summary.map(|t| t.content)
                .or_else(|| entry.content.and_then(|c| c.body));
            let author = entry.authors.first().map(|a| a.name.clone());
            let published_at = entry.published.or(entry.updated).unwrap_or_else(Utc::now);

            sqlx::query!(
                r#"
                INSERT INTO items (feed_id, title, link, content, author, published_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (feed_id, link) DO NOTHING
                "#,
                feed_info.id, title, link, content, author, published_at,
            )
            .execute(pool)
            .await?;
        }
        Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
    }.await;

    match processing_result {
        Ok(_) => {
            sqlx::query!(
                "UPDATE feeds SET last_fetched_at = $1, error_count = 0 WHERE id = $2",
                Utc::now(),
                feed_info.id
            )
            .execute(pool)
            .await?;
            Ok(())
        }
        Err(e) => {
            error!("Error fetching feed {}: {}", feed_info.url, e);
            
            sqlx::query!(
                "UPDATE feeds SET error_count = error_count + 1 WHERE id = $1",
                feed_info.id
            )
            .execute(pool)
            .await?;
            
            Err(e)
        }
    }
}
