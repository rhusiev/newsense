"""Story threads: link clusters that follow one event over several days.

A settled cluster joins a thread when it is close to one of the thread's
latest stories both in meaning (mean item embedding) and in rare title words
(TF-IDF over word-boundary character n-grams), within a time gap. Clusters
are taken oldest first, so a thread only grows forward.
"""

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    THREAD_EMBEDDING_MIN,
    THREAD_GAP_DAYS,
    THREAD_RECENT_STORIES,
    THREAD_SETTLE_HOURS,
    THREAD_TITLE_WORDS_MIN,
    THREAD_VOCABULARY_DAYS,
)

logger = logging.getLogger("threads")


def _vector(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.strip("[]").split(",")], dtype=np.float32)


async def _load_clusters(conn):
    """Clusters started in the vocabulary window, with titles and mean unit embedding."""
    rows = await conn.fetch(
        f"""
        SELECT c.id, c.thread_id, c.threads_checked_at IS NOT NULL AS checked,
               MIN(i.published_at) AS started_at,
               string_agg(i.title, ' ') AS titles,
               array_agg(i.embedding::text) AS embeddings
        FROM clusters c
        JOIN items i ON i.cluster_id = c.id
        WHERE c.created_at > now() - interval '{THREAD_VOCABULARY_DAYS} days'
          AND i.embedding IS NOT NULL
        GROUP BY c.id
        ORDER BY started_at, c.id
        """
    )
    vectors = []
    for row in rows:
        items = np.stack([_vector(e) for e in row["embeddings"]])
        items /= np.linalg.norm(items, axis=1, keepdims=True) + 1e-9
        mean = items.mean(0)
        vectors.append(mean / (np.linalg.norm(mean) + 1e-9))
    return rows, np.stack(vectors) if vectors else np.zeros((0, 0))


def assign(started, ready, group, meaning, words):
    """Threads the `ready` clusters, oldest first, and returns their groups.

    `group[i]` is the thread key of an already checked cluster: its thread id,
    or ("story", i) when it has no thread. A ready cluster joins the group
    whose latest stories it matches best, else it starts its own group.
    `started` is in seconds; `meaning` and `words` are cluster similarity
    matrices. Clusters neither checked nor ready are ignored.
    """
    gap = THREAD_GAP_DAYS * 86400
    members: dict = {}
    for i in range(len(started)):
        if not ready[i] and group[i] is None:
            continue
        if ready[i]:
            best, best_score = None, 0.0
            for key, mem in members.items():
                recent = mem[-THREAD_RECENT_STORIES:]
                if started[i] - started[recent[-1]] > gap:
                    continue
                se = meaning[recent, i]
                sw = words[recent, i]
                ok = (se >= THREAD_EMBEDDING_MIN) & (sw >= THREAD_TITLE_WORDS_MIN)
                if ok.any():
                    score = float((se + sw)[ok].max())
                    if score > best_score:
                        best, best_score = key, score
            group[i] = best if best is not None else ("story", i)
        members.setdefault(group[i], []).append(i)
    return group


async def update_threads(conn):
    rows, vectors = await _load_clusters(conn)
    if not rows:
        return 0
    settle = np.datetime64("now") - np.timedelta64(THREAD_SETTLE_HOURS, "h")
    started = np.array([r["started_at"].timestamp() for r in rows])
    ready = [
        not r["checked"] and np.datetime64(r["started_at"].replace(tzinfo=None)) < settle
        for r in rows
    ]
    if not any(ready):
        return 0
    group = [
        (r["thread_id"] or ("story", i)) if r["checked"] else None
        for i, r in enumerate(rows)
    ]
    # Rare means in at most 2% of titles, but a word shared by two stories always counts.
    rare = max(0.02, 2 / len(rows))
    try:
        words = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(4, 5), min_df=2, max_df=rare, sublinear_tf=True
        ).fit_transform([r["titles"] for r in rows])
    except ValueError:
        return 0  # no word is shared by two titles yet
    group = assign(started, ready, group, vectors @ vectors.T, (words @ words.T).toarray())

    # Each group of two or more clusters is a thread; a ("story", i) key gets a new one.
    sizes: dict = {}
    for key in group:
        if key is not None:
            sizes[key] = sizes.get(key, 0) + 1
    async with conn.transaction():
        created = {}
        for i, r in enumerate(rows):
            key = group[i]
            if key is None or (not ready[i] and r["thread_id"] == key):
                continue
            if sizes[key] < 2:
                if ready[i]:
                    await conn.execute(
                        "UPDATE clusters SET threads_checked_at = now() WHERE id = $1", r["id"]
                    )
                continue
            if isinstance(key, tuple):
                if key not in created:
                    created[key] = await conn.fetchval(
                        "INSERT INTO threads (last_story_at) VALUES ($1) RETURNING id",
                        r["started_at"],
                    )
                thread_id = created[key]
            else:
                thread_id = key
            await conn.execute(
                "UPDATE clusters SET thread_id = $1, threads_checked_at = now() WHERE id = $2",
                thread_id,
                r["id"],
            )
            await conn.execute(
                "UPDATE threads SET last_story_at = GREATEST(last_story_at, $2) WHERE id = $1",
                thread_id,
                r["started_at"],
            )
    joined = sum(1 for i, k in enumerate(group) if ready[i] and sizes.get(k, 0) >= 2)
    logger.info(f"Threads: checked {sum(ready)} stories, {joined} in threads")
    return joined
