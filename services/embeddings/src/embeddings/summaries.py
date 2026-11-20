"""Automatic LLM summaries of the stories a user has seen.

A story is summarized only when the user turned summaries on, the story
appeared in one of their lists, it has full text from a private feed they can
read, and it settled for a few hours. Results are cached per user and story;
failed attempts back off and count toward the daily limit.
"""

import hashlib
import logging

import httpx

from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT_S,
    LLM_SUMMARY_MAX_ATTEMPTS,
    LLM_SUMMARY_MAX_CHARS,
    LLM_SUMMARY_MAX_WORDS,
    LLM_SUMMARY_SETTLE_HOURS,
    LLM_SUMMARY_BATCH,
)

logger = logging.getLogger("summaries")

CANDIDATES = """
    SELECT si.user_id, si.story_id, us.llm_daily_limit
    FROM story_impressions si
    JOIN user_settings us ON us.user_id = si.user_id
    LEFT JOIN story_summaries ss
        ON ss.user_id = si.user_id AND ss.story_id = si.story_id
    WHERE us.llm_summaries
      AND us.llm_allowed
      AND (
          ss.user_id IS NULL
          OR (ss.summary IS NULL AND ss.attempts < $1
              AND ss.updated_at < now() - interval '1 hour')
      )
      AND NOT EXISTS (
          SELECT 1 FROM items recent
          WHERE COALESCE(recent.cluster_id, recent.id) = si.story_id
            AND recent.published_at > now() - make_interval(hours => $2)
      )
      AND EXISTS (
          SELECT 1
          FROM items i
          JOIN item_full_texts ft ON ft.item_id = i.id AND ft.text IS NOT NULL
          JOIN feeds f ON f.source_id = i.source_id
          LEFT JOIN feed_subscriptions fs
              ON fs.feed_id = f.id AND fs.user_id = si.user_id
          WHERE COALESCE(i.cluster_id, i.id) = si.story_id
            AND f.full_text AND NOT f.is_public
            AND (f.owner_id = si.user_id OR fs.user_id = si.user_id)
      )
    ORDER BY si.first_shown_at
    LIMIT $3
"""

# The reader sees only full text from a private feed with the switch on, the
# same rule as GET /items/{id}/full-text.
STORY_TEXT = """
    SELECT DISTINCT i.id, i.title, ft.text, ft.fetched_at, i.published_at
    FROM items i
    JOIN item_full_texts ft ON ft.item_id = i.id AND ft.text IS NOT NULL
    JOIN feeds f ON f.source_id = i.source_id
    LEFT JOIN feed_subscriptions fs ON fs.feed_id = f.id AND fs.user_id = $1
    WHERE COALESCE(i.cluster_id, i.id) = $2
      AND f.full_text AND NOT f.is_public
      AND (f.owner_id = $1 OR fs.user_id = $1)
    ORDER BY i.published_at
"""

CALLS_TODAY = """
    SELECT COUNT(*) FROM llm_calls
    WHERE user_id = $1 AND created_at >= date_trunc('day', now())
"""

SAVE = """
    INSERT INTO story_summaries
        (user_id, story_id, summary, error, attempts, source_hash, model)
    VALUES ($1, $2, $3, NULL, 1, $4, $5)
    ON CONFLICT (user_id, story_id) DO UPDATE SET
        summary = EXCLUDED.summary,
        error = NULL,
        attempts = story_summaries.attempts + 1,
        source_hash = EXCLUDED.source_hash,
        model = EXCLUDED.model,
        updated_at = now()
"""

FAIL = """
    INSERT INTO story_summaries (user_id, story_id, summary, error, attempts)
    VALUES ($1, $2, NULL, $3, 1)
    ON CONFLICT (user_id, story_id) DO UPDATE SET
        error = EXCLUDED.error,
        attempts = story_summaries.attempts + 1,
        updated_at = now()
"""

RECORD_CALL = """
    INSERT INTO llm_calls (user_id, story_id, model, success)
    VALUES ($1, $2, $3, $4)
"""

SYSTEM = (
    "You summarize news stories for a reader. Write in the language of the "
    "articles. Use at most 180 words in one plain paragraph. Say what happened, "
    "what changed and what is still open. Use only facts from the articles. No "
    "headings, lists or markdown."
)


def _configured() -> bool:
    return bool(LLM_BASE_URL and LLM_API_KEY and LLM_MODEL)


def _prepare(rows) -> tuple[str, str, str]:
    """Title, prompt body and a hash of the source texts."""
    title = rows[0]["title"]
    parts = [f"{row['title']}\n{row['text']}" for row in rows]
    body = "\n\n".join(parts)
    if len(body) > LLM_SUMMARY_MAX_CHARS:
        body = body[:LLM_SUMMARY_MAX_CHARS] + "\n[truncated]"
    source = "|".join(f"{row['id']}:{row['fetched_at'].isoformat()}" for row in rows)
    return title, body, hashlib.sha256(source.encode()).hexdigest()


async def _call_model(title: str, body: str) -> str:
    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Story: {title}\n\n{body}"},
        ],
        "max_tokens": 400,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT_S) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json=payload,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
    words = content.split()
    if len(words) > LLM_SUMMARY_MAX_WORDS:
        content = " ".join(words[:LLM_SUMMARY_MAX_WORDS])
    content = content.strip()
    if not content:
        raise ValueError("empty summary")
    return content


async def update_summaries(conn) -> int:
    """Summarizes one batch of pending stories. Returns the number stored."""
    if not _configured():
        return 0

    candidates = await conn.fetch(
        CANDIDATES,
        LLM_SUMMARY_MAX_ATTEMPTS,
        LLM_SUMMARY_SETTLE_HOURS,
        LLM_SUMMARY_BATCH,
    )
    calls: dict = {}
    stored = 0

    for row in candidates:
        user_id, story_id = row["user_id"], row["story_id"]
        if user_id not in calls:
            calls[user_id] = await conn.fetchval(CALLS_TODAY, user_id)
        if calls[user_id] >= row["llm_daily_limit"]:
            continue

        texts = await conn.fetch(STORY_TEXT, user_id, story_id)
        if not texts:
            continue
        title, body, source_hash = _prepare(texts)

        try:
            summary = await _call_model(title, body)
        except Exception as e:
            logger.warning(f"Summary failed for story {story_id}: {e}")
            async with conn.transaction():
                await conn.execute(FAIL, user_id, story_id, str(e)[:1000])
                await conn.execute(
                    RECORD_CALL, user_id, story_id, LLM_MODEL, False
                )
            calls[user_id] += 1
            continue

        async with conn.transaction():
            await conn.execute(SAVE, user_id, story_id, summary, source_hash, LLM_MODEL)
            await conn.execute(RECORD_CALL, user_id, story_id, LLM_MODEL, True)
        calls[user_id] += 1
        stored += 1

    return stored
