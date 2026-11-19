//! Fetches the article page for new items of feeds with full text switched on
//! and keeps its readable part. A source that any public feed uses never gets
//! full text: the stored text then stays with private feeds' owners.

use dom_smoothie::Readability;
use log::{error, info};
use sqlx::PgPool;
use std::{
    collections::HashMap,
    net::IpAddr,
    time::{Duration, Instant},
};
use url::Url;

const POLL_INTERVAL: Duration = Duration::from_secs(60);
const BATCH_SIZE: i64 = 20;
/// Only items this new get full text, so switching it on does not crawl the backlog.
const MAX_ITEM_AGE_HOURS: i32 = 48;
const PER_HOST_GAP: Duration = Duration::from_secs(5);
const MAX_PAGE_BYTES: usize = 2 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const MAX_ERROR_CHARS: usize = 500;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

struct Pending {
    id: uuid::Uuid,
    link: String,
}

struct Extracted {
    html: String,
    text: String,
}

pub async fn full_text_task(pool: PgPool, client: reqwest::Client) {
    let mut last_request: HashMap<String, Instant> = HashMap::new();
    info!("Full-text task started");
    loop {
        match pending_items(&pool).await {
            Ok(items) => {
                for item in items {
                    let outcome = fetch_politely(&client, &mut last_request, &item.link).await;
                    if let Err(e) = store(&pool, item.id, outcome).await {
                        error!("Full-text store failed for {}: {}", item.id, e);
                    }
                }
            }
            Err(e) => error!("Full-text DB error: {}", e),
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn pending_items(pool: &PgPool) -> Result<Vec<Pending>, sqlx::Error> {
    sqlx::query_as!(
        Pending,
        r#"
        SELECT i.id, i.link
        FROM items i
        WHERE i.created_at > NOW() - make_interval(hours => $1)
          AND NOT EXISTS (SELECT 1 FROM item_full_texts t WHERE t.item_id = i.id)
          AND EXISTS (SELECT 1 FROM feeds f WHERE f.source_id = i.source_id AND f.full_text)
          AND NOT EXISTS (SELECT 1 FROM feeds f WHERE f.source_id = i.source_id AND f.is_public)
        ORDER BY i.created_at
        LIMIT $2
        "#,
        MAX_ITEM_AGE_HOURS,
        BATCH_SIZE
    )
    .fetch_all(pool)
    .await
}

async fn store(
    pool: &PgPool,
    item_id: uuid::Uuid,
    outcome: Result<Extracted, BoxError>,
) -> Result<(), sqlx::Error> {
    let (html, text, err) = match outcome {
        Ok(e) => (Some(e.html), Some(e.text), None),
        Err(e) => (
            None,
            None,
            Some(
                e.to_string()
                    .chars()
                    .take(MAX_ERROR_CHARS)
                    .collect::<String>(),
            ),
        ),
    };
    sqlx::query!(
        r#"
        INSERT INTO item_full_texts (item_id, html, text, error)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (item_id) DO NOTHING
        "#,
        item_id,
        html,
        text,
        err
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Waits until the host has had `PER_HOST_GAP` of rest, then fetches.
async fn fetch_politely(
    client: &reqwest::Client,
    last_request: &mut HashMap<String, Instant>,
    link: &str,
) -> Result<Extracted, BoxError> {
    let host = Url::parse(link)?.host_str().unwrap_or_default().to_string();
    if let Some(at) = last_request.get(&host) {
        let rest = at.elapsed();
        if rest < PER_HOST_GAP {
            tokio::time::sleep(PER_HOST_GAP - rest).await;
        }
    }
    last_request.insert(host, Instant::now());
    let (final_url, html) = fetch_page(client, link).await?;
    tokio::task::spawn_blocking(move || extract(&html, &final_url)).await?
}

/// Follows redirects by hand so that every hop is checked against private addresses.
/// The client must be built with redirects disabled.
async fn fetch_page(client: &reqwest::Client, link: &str) -> Result<(String, String), BoxError> {
    let mut url = Url::parse(link)?;
    for _ in 0..=MAX_REDIRECTS {
        ensure_public(&url).await?;
        let mut response = client.get(url.clone()).send().await?;
        if response.status().is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .ok_or("redirect without location")?
                .to_str()?;
            url = url.join(location)?;
            continue;
        }
        let response_ok = response.error_for_status_ref().map(|_| ());
        response_ok?;
        let is_html = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains("html"));
        if !is_html {
            return Err("not an HTML page".into());
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            body.extend_from_slice(&chunk);
            if body.len() > MAX_PAGE_BYTES {
                return Err("page too large".into());
            }
        }
        return Ok((url.to_string(), String::from_utf8_lossy(&body).into_owned()));
    }
    Err("too many redirects".into())
}

/// Refuses hosts that resolve to loopback, private or link-local addresses,
/// so a feed link cannot make the fetcher read services inside the network.
/// A host that changes its DNS answer between this check and the request is not caught.
async fn ensure_public(url: &Url) -> Result<(), BoxError> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("only http and https are fetched".into());
    }
    let host = url.host_str().ok_or("no host")?;
    let port = url.port_or_known_default().ok_or("no port")?;
    for addr in tokio::net::lookup_host((host, port)).await? {
        if !is_public(addr.ip()) {
            return Err("host resolves to a private address".into());
        }
    }
    Ok(())
}

fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.octets()[0] == 100 && (v4.octets()[1] & 0b1100_0000) == 64)
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public(IpAddr::V4(v4));
            }
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local())
        }
    }
}

/// Keeps the readable part of the page. Images are dropped, so opening the
/// reader view sends no request to the publisher.
fn extract(html: &str, url: &str) -> Result<Extracted, BoxError> {
    let article = Readability::new(html, Some(url), None)?.parse()?;
    let clean = ammonia::Builder::default()
        .rm_tags(&["img", "picture", "source"])
        .url_relative(ammonia::UrlRelative::Deny)
        .clean(&article.content)
        .to_string();
    Ok(Extracted {
        html: clean,
        text: article.text_content.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_and_internal_addresses_are_refused() {
        for ip in [
            "127.0.0.1",
            "10.0.0.5",
            "172.18.0.2",
            "192.168.1.1",
            "169.254.1.1",
            "100.64.0.1",
            "::1",
            "fd00::1",
            "::ffff:10.0.0.1",
        ] {
            assert!(!is_public(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["1.1.1.1", "2606:4700:4700::1111"] {
            assert!(is_public(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn extraction_keeps_text_and_drops_scripts_and_images() {
        let page = format!(
            "<html><head><title>T</title></head><body><nav>menu</nav><article><h1>T</h1>{}<img src='https://x.example/a.png'><script>alert(1)</script></article></body></html>",
            "<p>Body text of the article that is long enough to count as content. </p>".repeat(20)
        );
        let e = extract(&page, "https://example.org/a").unwrap();
        assert!(e.text.contains("Body text of the article"));
        assert!(!e.html.contains("<script"));
        assert!(!e.html.contains("<img"));
    }
}
