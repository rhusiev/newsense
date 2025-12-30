use sqlx::{PgPool, postgres::PgPoolOptions};
use feed_rs::parser;
use std::time::Duration;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use log::{debug, info, error};

#[derive(Clone)]
struct FeedToFetch {
    id: Uuid,
    url: String,
    last_fetched_at: Option<DateTime<Utc>>,
}

async fn fetch_and_store_feed(
    pool: &PgPool,
    http_client: &reqwest::Client,
    feed_info: FeedToFetch,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Fetching feed: {} (last fetched: {:?})", 
          feed_info.url, 
          feed_info.last_fetched_at);
    
    let response = http_client
        .get(&feed_info.url)
        .header("Accept", "*/*")
        .header("Connection", "close")
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| {
            error!("HTTP request failed for {}: {}", feed_info.url, e);
            e
        })?;
    
    let content_bytes = response.bytes().await.map_err(|e| {
        error!("Failed to read response body for {}: {}", feed_info.url, e);
        debug!("Error details: {:?}", e);
        debug!("Is timeout: {}, is body: {}, is decode: {}", 
                  e.is_timeout(), e.is_body(), e.is_decode());
        e
    })?;
    
    debug!("Content length: {} bytes", content_bytes.len());
    debug!("First 100 bytes (hex): {:02x?}", &content_bytes[..content_bytes.len().min(100)]);
    
    let content_str = String::from_utf8_lossy(&content_bytes);
    debug!("First 200 chars: {}", &content_str.chars().take(200).collect::<String>());
    
    let parser = parser::Builder::new().sanitize_content(true).build();
    let feed = parser.parse(&content_bytes[..]).map_err(|e| {
        error!("Failed to parse feed for {}: {}", feed_info.url, e);
        e
    })?;
    
    info!("Successfully parsed feed, entries: {}", feed.entries.len());
    
    for entry in feed.entries {
        let title = entry.title
            .map(|t| t.content)
            .unwrap_or_else(|| "Untitled".to_string());
        
        let link = entry.links
            .first()
            .map(|l| l.href.clone())
            .unwrap_or_default();
        
        if link.is_empty() {
            continue;
        }

        let content = entry.summary
            .map(|t| t.content)
            .or_else(|| entry.content.and_then(|c| c.body));
        
        let author = entry.authors
            .first()
            .map(|a| a.name.clone());
        
        let published_at = entry.published
            .or(entry.updated);

        let _ = sqlx::query!(
            r#"
            INSERT INTO items (feed_id, title, link, content, author, published_at, cluster_id, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, NULL, NULL)
            ON CONFLICT (feed_id, link) DO NOTHING
            "#,
            feed_info.id,
            title,
            link,
            content,
            author,
            published_at,
        )
        .execute(pool)
        .await;
    }

    sqlx::query!(
        r#"
        UPDATE feeds 
        SET last_fetched_at = $1, error_count = 0
        WHERE id = $2
        "#,
        Utc::now(),
        feed_info.id,
    )
    .execute(pool)
    .await?;

    Ok(())
}

async fn fetch_all_feeds(pool: &PgPool, http_client: &reqwest::Client) {
    let feeds = sqlx::query_as!(
        FeedToFetch,
        r#"
        SELECT id, url, last_fetched_at
        FROM feeds
        WHERE error_count < 5
        ORDER BY COALESCE(last_fetched_at, '1970-01-01'::timestamptz) ASC
        LIMIT 100
        "#
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    for feed in feeds {
        match fetch_and_store_feed(pool, http_client, feed.clone()).await {
            Ok(_) => {},
            Err(e) => {
                error!("Error fetching feed {}: {}", feed.url, e);
                let _ = sqlx::query!(
                    "UPDATE feeds SET error_count = error_count + 1 WHERE id = $1",
                    feed.id
                )
                .execute(pool)
                .await;
            }
        }
        
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://user:pass@localhost/authdb".to_string());

    let pool = PgPoolOptions::new()
        .max_connections(2)
        .connect(&database_url)
        .await
        .expect("Failed to connect to database");

    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent("Mozilla/5.0 (compatible; RSSReader/1.0)")
        .build()
        .expect("Failed to create HTTP client");

    let fetch_interval = Duration::from_secs(
        std::env::var("FETCH_INTERVAL_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300)
    );

    info!("RSS worker started, fetching every {} seconds", fetch_interval.as_secs());

    loop {
        fetch_all_feeds(&pool, &http_client).await;
        tokio::time::sleep(fetch_interval).await;
    }
}
