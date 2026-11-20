"""Story threads: communities of stories that follow one event over days.

Each article has a full-text embedding. A settled story joins a thread when
Leiden finds it in the same community in a cosine kNN graph. The graph settings
were tuned on two reviewed weeks: pairwise F1 about 0.54 on each, versus 0.40
for the previous rule.
"""

import logging
from collections import Counter

import igraph as ig
import leidenalg
import numpy as np

from .config import (
    THREAD_EMBEDDING_MIN,
    THREAD_KNN,
    THREAD_RESOLUTION,
    THREAD_SETTLE_HOURS,
    THREAD_VOCABULARY_DAYS,
)

logger = logging.getLogger("threads")


def _vector(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.strip("[]").split(",")], dtype=np.float32)


def group(vectors: np.ndarray) -> np.ndarray:
    """Return a community label per story from a kNN graph of unit vectors.

    `vectors` rows must already be L2-normalized. Isolated stories keep their
    own label; every community of two or more stories gets one label.
    """
    n = len(vectors)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    sim = vectors @ vectors.T
    order = np.argsort(-sim, axis=1)
    edges = {}
    for i in range(n):
        for j in order[i, 1 : THREAD_KNN + 1]:
            j = int(j)
            if sim[i, j] < THREAD_EMBEDDING_MIN:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = max(edges.get((a, b), 0.0), float(sim[i, j]))
    graph = ig.Graph(n=n, edges=list(edges), directed=False)
    graph.es["weight"] = list(edges.values())
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=THREAD_RESOLUTION,
        seed=42,
    )
    return np.array(partition.membership, dtype=int)


async def _load_stories(conn):
    """Settled clusters in the vocabulary window with their mean story vector."""
    rows = await conn.fetch(
        """
        SELECT c.id, c.thread_id, MIN(i.published_at) AS started_at,
               array_agg(i.story_embedding::text) AS embeddings
        FROM clusters c
        JOIN items i ON i.cluster_id = c.id
        WHERE c.created_at > now() - interval '{days} days'
          AND i.story_embedding IS NOT NULL
        GROUP BY c.id, c.thread_id
        HAVING MIN(i.published_at) < now() - interval '{settle} hours'
        ORDER BY started_at, c.id
        """.format(
            days=THREAD_VOCABULARY_DAYS, settle=THREAD_SETTLE_HOURS
        )
    )
    vectors = []
    for row in rows:
        items = np.stack([_vector(value) for value in row["embeddings"]])
        items /= np.linalg.norm(items, axis=1, keepdims=True) + 1e-9
        mean = items.mean(0)
        vectors.append(mean / (np.linalg.norm(mean) + 1e-9))
    return rows, np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)


async def update_threads(conn) -> int:
    """Regroup all settled stories and return how many joined a thread."""
    rows, vectors = await _load_stories(conn)
    if len(rows) < 2:
        return 0
    labels = group(vectors)

    members: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        members.setdefault(int(label), []).append(i)

    async with conn.transaction():
        async with conn.cursor() as cursor:
            for indices in members.values():
                if len(indices) < 2:
                    continue
                existing = [
                    rows[i]["thread_id"] for i in indices if rows[i]["thread_id"]
                ]
                if existing:
                    thread_id = Counter(existing).most_common(1)[0][0]
                else:
                    thread_id = await conn.fetchval(
                        "INSERT INTO threads (last_story_at) VALUES ($1) RETURNING id",
                        max(rows[i]["started_at"] for i in indices),
                    )
                for i in indices:
                    await cursor.execute(
                        "UPDATE clusters SET thread_id = $1, threads_checked_at = now() "
                        "WHERE id = $2",
                        thread_id,
                        rows[i]["id"],
                    )
                await cursor.execute(
                    "UPDATE threads SET last_story_at = GREATEST(last_story_at, $2) "
                    "WHERE id = $1",
                    thread_id,
                    max(rows[i]["started_at"] for i in indices),
                )
            # Stories that now stand alone keep no thread.
            for indices in members.values():
                if len(indices) > 1:
                    continue
                i = indices[0]
                if rows[i]["thread_id"]:
                    await cursor.execute(
                        "UPDATE clusters SET thread_id = NULL, threads_checked_at = now() "
                        "WHERE id = $1",
                        rows[i]["id"],
                    )
    joined = sum(len(indices) for indices in members.values() if len(indices) > 1)
    logger.info(
        f"Threads: checked {len(rows)} stories, {joined} in "
        f"{sum(1 for v in members.values() if len(v) > 1)} threads"
    )
    return joined
