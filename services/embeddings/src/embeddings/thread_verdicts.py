"""LLM verification of candidate story pairs for thread grouping.

Verdicts are cached per unordered story pair and model. Only titles and a
short text excerpt leave the worker, under the configured privacy routing.
"""
import asyncio
import hashlib
import json
import logging

import httpx

from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER_ROUTING,
    LLM_REASONING,
    LLM_REQUEST_TIMEOUT_S,
    THREAD_LLM_CONCURRENCY,
    THREAD_LLM_DAILY_CALLS,
    THREAD_LLM_MAX_CALLS,
    THREAD_LLM_MAX_CHARS,
    THREAD_LLM_THRESHOLD,
)

logger = logging.getLogger("thread-verdicts")

SYSTEM = """You decide whether two news stories belong to the SAME news thread.
A thread is one ongoing story: a specific event plus its direct updates, reactions, fallout, and follow-ups.
Stories belong together when a reader would file them under the same headline, including later consequences and responses.
Stories on the same topic but about separate incidents, different targets, different dates or different model versions are NOT the same thread.
Recurring reports belong to one thread when they are the same published series, even across days; a reader would expect them collapsed to the latest report.
Reply with only compact JSON: {"same": true/false, "confidence": 0.0 to 1.0}"""


def _source_hash(title_a: str, body_a: str, title_b: str, body_b: str) -> str:
    source = "|".join((title_a, body_a, title_b, body_b))
    return hashlib.sha256(source.encode()).hexdigest()


async def _call(client: httpx.AsyncClient, pair: tuple, texts: dict, model: str) -> float | None:
    a, b = pair
    title_a, body_a = texts[a]
    title_b, body_b = texts[b]
    user = (
        f"Story A: {title_a[:THREAD_LLM_MAX_CHARS]}\n{body_a[:THREAD_LLM_MAX_CHARS]}\n\n"
        f"Story B: {title_b[:THREAD_LLM_MAX_CHARS]}\n{body_b[:THREAD_LLM_MAX_CHARS]}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 200,
    }
    if LLM_PROVIDER_ROUTING:
        payload["provider"] = LLM_PROVIDER_ROUTING
    if LLM_REASONING:
        payload["reasoning"] = LLM_REASONING
    try:
        response = await client.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json=payload,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        verdict = json.loads(content.strip().strip("`").replace("json\n", ""))
        confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0.5))))
        return confidence if verdict.get("same") else 1 - confidence
    except Exception as exc:
        logger.warning("Thread verdict failed for %s-%s: %s", a, b, str(exc)[:200])
        return None


async def score_pairs(conn, pairs: list[tuple], texts: dict) -> dict[tuple, float]:
    """Return probabilities for the pairs, using cached rows where possible."""
    if not pairs:
        return {}
    model = LLM_MODEL
    rows = await conn.fetch(
        """
        SELECT story_a, story_b, source_hash, prob
        FROM thread_pair_scores
        WHERE model = $1 AND story_a = ANY($2::uuid[]) AND story_b = ANY($2::uuid[])
        """,
        model,
        [pid for pair in pairs for pid in pair],
    )
    cache = {(row["story_a"], row["story_b"]): row for row in rows}
    # Every request counts, so a failing provider cannot be called again each run.
    today = await conn.fetchval(
        "SELECT COUNT(*) FROM thread_llm_calls WHERE created_at >= date_trunc('day', now())"
    )
    remaining = max(0, THREAD_LLM_DAILY_CALLS - int(today))
    out: dict[tuple, float] = {}
    pending = []
    for pair in pairs:
        a, b = pair
        title_a, body_a = texts[a]
        title_b, body_b = texts[b]
        digest = _source_hash(title_a, body_a, title_b, body_b)
        row = cache.get(pair)
        if row and row["source_hash"] == digest:
            out[pair] = float(row["prob"])
        elif len(pending) < min(THREAD_LLM_MAX_CALLS, remaining):
            pending.append((pair, digest))
        elif row:
            out[pair] = float(row["prob"])
    if not pending:
        return out

    sem = asyncio.Semaphore(THREAD_LLM_CONCURRENCY)
    async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT_S) as client:
        async def limited(item):
            async with sem:
                return item[0], await _call(client, item[0], texts, model), item[1]

        results = await asyncio.gather(*(limited(item) for item in pending))

    async with conn.transaction():
        await conn.executemany(
            "INSERT INTO thread_llm_calls (success) VALUES ($1)",
            [(prob is not None,) for _, prob, _ in results],
        )
        for pair, prob, digest in results:
            if prob is None:
                row = cache.get(pair)
                if row:
                    out[pair] = float(row["prob"])
                continue
            a, b = pair
            out[pair] = prob
            await conn.execute(
                """
                INSERT INTO thread_pair_scores
                    (story_a, story_b, model, source_hash, prob)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (story_a, story_b, model) DO UPDATE
                SET source_hash = EXCLUDED.source_hash,
                    prob = EXCLUDED.prob,
                    created_at = now()
                """,
                a, b, model, digest, prob,
            )
    return out


def accept(prob: float) -> bool:
    return prob >= THREAD_LLM_THRESHOLD
