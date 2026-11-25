//! Fetches the article page for new items of feeds with full text switched on
//! and keeps its readable part. A source that any public feed uses never gets
//! full text: the stored text then stays with private feeds' owners.

use dom_smoothie::Readability;
use log::{error, info};
use sqlx::PgPool;
use std::{
    collections::HashMap,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    time::{Duration, Instant},
};
use url::Url;

const POLL_INTERVAL: Duration = Duration::from_secs(60);
const BATCH_SIZE: i64 = 20;
/// Only items this new get full text, so switching it on does not crawl the backlog.
const MAX_ITEM_AGE_HOURS: i32 = 48;
/// A failed fetch waits this long before another try, so an item gets at most
/// `MAX_ITEM_AGE_HOURS / RETRY_AFTER_HOURS` attempts.
const RETRY_AFTER_HOURS: i32 = 6;
const PER_HOST_GAP: Duration = Duration::from_secs(5);
const MAX_PAGE_BYTES: usize = 2 * 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const MAX_ERROR_CHARS: usize = 500;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const USER_AGENT: &str = "Mozilla/5.0 (compatible; RSSReader/1.0)";
/// Where an image's address waits until the user asks for the image.
const DEFERRED_SRC: &str = "data-src";

type BoxError = Box<dyn std::error::Error + Send + Sync>;

struct Pending {
    id: uuid::Uuid,
    link: String,
}

struct Extracted {
    html: String,
    text: String,
}

pub async fn full_text_task(pool: PgPool) {
    let mut last_request: HashMap<String, Instant> = HashMap::new();
    info!("Full-text task started");
    loop {
        match pending_items(&pool).await {
            Ok(items) => {
                for item in items {
                    let outcome = fetch_politely(&mut last_request, &item.link).await;
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
          -- A failed fetch is tried again after a rest, within the age window.
          AND NOT EXISTS (
              SELECT 1 FROM item_full_texts t
              WHERE t.item_id = i.id
                AND (t.error IS NULL OR t.fetched_at > NOW() - make_interval(hours => $3))
          )
          AND EXISTS (SELECT 1 FROM feeds f WHERE f.source_id = i.source_id AND f.full_text)
          AND NOT EXISTS (SELECT 1 FROM feeds f WHERE f.source_id = i.source_id AND f.is_public)
        ORDER BY i.created_at
        LIMIT $2
        "#,
        MAX_ITEM_AGE_HOURS,
        BATCH_SIZE,
        RETRY_AFTER_HOURS
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
        ON CONFLICT (item_id) DO UPDATE
        SET html = EXCLUDED.html, text = EXCLUDED.text, error = EXCLUDED.error,
            fetched_at = CURRENT_TIMESTAMP
        WHERE item_full_texts.error IS NOT NULL
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
    let (final_url, html) = fetch_page(link).await?;
    if let Some(final_host) = Url::parse(&final_url).ok().and_then(|u| u.host_str().map(str::to_string)) {
        last_request.insert(final_host, Instant::now());
    }
    tokio::task::spawn_blocking(move || extract(&html, &final_url)).await?
}

/// Follows redirects by hand so that every hop is checked against private addresses.
/// The client must be built with redirects disabled.
async fn fetch_page(link: &str) -> Result<(String, String), BoxError> {
    let mut url = Url::parse(link)?;
    for _ in 0..=MAX_REDIRECTS {
        let resolved = ensure_public(&url).await?;
        let mut client_builder = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .user_agent(USER_AGENT)
            .redirect(reqwest::redirect::Policy::none());
        for addr in &resolved {
            client_builder = client_builder.resolve(
                url.host_str().ok_or("no host")?, *addr,
            );
        }
        let client = client_builder.build()?;
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
/// The request is then pinned to a checked address, so a later DNS answer cannot move it.
async fn ensure_public(url: &Url) -> Result<Vec<std::net::SocketAddr>, BoxError> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err("only http and https are fetched".into());
    }
    let host = url.host_str().ok_or("no host")?;
    let port = url.port_or_known_default().ok_or("no port")?;
    let resolved: Vec<_> = tokio::net::lookup_host((host, port)).await?.collect();
    if resolved.is_empty() || resolved.iter().any(|addr| !is_public(addr.ip())) {
        return Err("host resolves to a private address".into());
    }
    Ok(resolved)
}

fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => is_public_v6(v6),
    }
}

/// Special-purpose IPv4 ranges (IANA registry): none of them is a public host.
fn is_public_v4(v4: Ipv4Addr) -> bool {
    let [a, b, c, _] = v4.octets();
    !(a == 0
        || v4.is_loopback()
        || v4.is_private()
        || v4.is_link_local()
        || v4.is_documentation()
        || (a == 100 && (b & 0b1100_0000) == 64)
        || (a == 192 && b == 0 && c == 0)
        || (a == 198 && (b & 0b1111_1110) == 18)
        || a >= 224)
}

/// IPv6 forms that carry an IPv4 address are judged by that address; NAT64 and
/// 6to4 gateways would otherwise reach private IPv4 hosts.
fn is_public_v6(v6: Ipv6Addr) -> bool {
    let s = v6.segments();
    if let Some(v4) = v6.to_ipv4() {
        // `::a.b.c.d` (compatible) and `::ffff:a.b.c.d` (mapped).
        return !v6.is_loopback() && !v6.is_unspecified() && is_public_v4(v4);
    }
    if s[0] == 0x64 && s[1] == 0xff9b && s[2..6] == [0; 4] {
        let o = v6.octets();
        return is_public_v4(Ipv4Addr::new(o[12], o[13], o[14], o[15]));
    }
    if s[0] == 0x2002 {
        let o = v6.octets();
        return is_public_v4(Ipv4Addr::new(o[2], o[3], o[4], o[5]));
    }
    !(v6.is_multicast()
        || v6.is_unique_local()
        || v6.is_unicast_link_local()
        // Teredo, discard-only and documentation prefixes.
        || (s[0] == 0x2001 && s[1] == 0)
        || (s[0] == 0x100 && s[1..4] == [0; 3])
        || (s[0] == 0x2001 && s[1] == 0xdb8))
}

/// Keeps the readable part of the page. Image addresses move from `src` to
/// `data-src`, so the reader view loads an image from the publisher only when
/// the user taps it.
fn extract(html: &str, url: &str) -> Result<Extracted, BoxError> {
    let article = Readability::new(html, Some(url), None)?.parse()?;
    let clean = ammonia::Builder::default()
        .rm_tags(&["picture", "source"])
        .url_relative(ammonia::UrlRelative::Deny)
        .clean(&article.content)
        .to_string();
    Ok(Extracted {
        html: defer_images(&clean),
        text: article.text_content.to_string(),
    })
}

fn defer_images(html: &str) -> String {
    let doc = dom_query::Document::from(html);
    for img in doc.select("img").iter() {
        match img.attr("src") {
            Some(src) => {
                img.set_attr(DEFERRED_SRC, &src);
                img.remove_attr("src");
            }
            None => img.remove(),
        }
    }
    doc.select("body").inner_html().to_string()
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
            "0.1.2.3",
            "192.0.0.8",
            "198.18.0.1",
            "224.0.0.1",
            "240.0.0.1",
            "255.255.255.255",
            "::10.0.0.1",
            "64:ff9b::10.0.0.1",
            "64:ff9b::7f00:1",
            "2002:0a00:0001::1",
            "2002:7f00:0001::1",
            "2001:0:4136:e378::1",
            "2001:db8::1",
            "ff02::1",
            "fe80::1",
        ] {
            assert!(!is_public(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "1.1.1.1",
            "2606:4700:4700::1111",
            "64:ff9b::1.1.1.1",
            "2002:0101:0101::1",
            "::ffff:8.8.8.8",
        ] {
            assert!(is_public(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn extraction_keeps_text_drops_scripts_and_defers_images() {
        let page = format!(
            "<html><head><title>T</title></head><body><nav>menu</nav><article><h1>T</h1>{}<img src='https://x.example/a.png' alt='map' onerror='alert(2)'><img src='/relative.png'><img alt='no address'><script>alert(1)</script></article></body></html>",
            "<p>Body text of the article that is long enough to count as content. </p>".repeat(20)
        );
        let e = extract(&page, "https://example.org/a").unwrap();
        assert!(e.text.contains("Body text of the article"));
        assert!(!e.html.contains("<script"));
        assert!(!e.html.contains("onerror"));
        assert!(!e.html.contains(" src="));
        assert!(e.html.contains(r#"data-src="https://x.example/a.png""#));
        assert!(
            e.html
                .contains(r#"data-src="https://example.org/relative.png""#)
        );
        assert_eq!(e.html.matches("<img").count(), 2);
    }
}
