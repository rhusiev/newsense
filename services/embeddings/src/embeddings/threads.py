"""Story threads: soft communities of stories that follow one storyline.

Each article has a full-text embedding. A settled story joins a thread when a
cosine kNN graph connects it to that thread and, when configured, a local
quantified cross-encoder confirms the pair. The graph settings were tuned on
two reviewed weeks; singletons stay alone unless a pair is accepted.
"""

import logging
from collections import Counter

import igraph as ig
import leidenalg
import numpy as np

from . import pair_scores
from .config import (
    THREAD_ALTERNATIVE_MIN,
    THREAD_EMBEDDING_MIN,
    THREAD_KNN,
    THREAD_RESOLUTION,
    THREAD_SETTLE_HOURS,
    THREAD_TWO_HOP,
    THREAD_VOCABULARY_DAYS,
)

logger = logging.getLogger("threads")


def _vector(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.strip("[]").split(",")], dtype=np.float32)


def _candidates(sim: np.ndarray) -> dict[tuple[int, int], float]:
    """Top-k cosine neighbors, optionally with neighbors of neighbors."""
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    edges: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in order[i, 1 : THREAD_KNN + 1]:
            j = int(j)
            if sim[i, j] < THREAD_EMBEDDING_MIN:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = max(edges.get((a, b), 0.0), float(sim[i, j]))
    if THREAD_TWO_HOP:
        for i in range(n):
            for j in order[i, 1:6]:
                j = int(j)
                if sim[i, j] < THREAD_EMBEDDING_MIN:
                    continue
                for k in order[j, 1:6]:
                    k = int(k)
                    if k == i or sim[j, k] < THREAD_EMBEDDING_MIN + 0.04:
                        continue
                    a, b = (i, k) if i < k else (k, i)
                    edges[(a, b)] = max(edges.get((a, b), 0.0), float(sim[j, k]) * 0.9)
    return edges


def group(vectors: np.ndarray, texts: list[str] | None = None):
    """Return labels, pair confidence and alternative cluster indices.

    `vectors` rows must be L2-normalized. When a pair scorer is configured,
    cosine candidates are confirmed by it and low-confidence pairs are left
    unjoined. `texts[i]` must match `vectors[i]`.
    """
    n = len(vectors)
    if n < 2:
        return np.zeros(n, dtype=int), {}, {}
    sim = vectors @ vectors.T
    edges = _candidates(sim)

    scored = {}
    if pair_scores.enabled() and texts is not None:
        keys = sorted(edges)
        probs = pair_scores.score([texts[a] for a, _ in keys], [texts[b] for _, b in keys])
        accepted = {}
        for (a, b), p in zip(keys, probs):
            if pair_scores.accept(float(p)):
                accepted[(a, b)] = float(p)
            else:
                scored[(a, b)] = float(p)
        edges = accepted
    else:
        for key in list(edges):
            scored[key] = edges[key]

    graph = ig.Graph(n=n, edges=list(edges), directed=False)
    graph.es["weight"] = list(edges.values())
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=THREAD_RESOLUTION,
        seed=42,
    )
    labels = np.array(partition.membership, dtype=int)

    confidence: dict[int, float] = {}
    for (a, b), p in edges.items():
        confidence[a] = max(confidence.get(a, 0.0), p)
        confidence[b] = max(confidence.get(b, 0.0), p)

    alternatives: dict[int, int] = {}
    for (a, b), p in scored.items():
        if p < THREAD_ALTERNATIVE_MIN or labels[a] == labels[b]:
            continue
        for i, other in ((a, b), (b, a)):
            if i not in alternatives or p > alternatives[i][1]:
                alternatives[i] = (int(labels[other]), p)
    alt_cluster = {i: value[0] for i, value in alternatives.items()}
    return labels, confidence, alt_cluster


async def _load_stories(conn):
    """Settled clusters in the vocabulary window with their mean story vector."""
    rows = await conn.fetch(
        """
        SELECT c.id, c.thread_id, MIN(i.published_at) AS started_at,
               string_agg(i.title, ' ') AS titles,
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
    texts = [row["titles"] or "" for row in rows]
    return rows, np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32), texts


async def update_threads(conn) -> int:
    """Regroup settled stories and return how many joined a thread."""
    rows, vectors, texts = await _load_stories(conn)
    if len(rows) < 2:
        return 0
    labels, confidence, alternatives = group(vectors, texts)

    members: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        members.setdefault(int(label), []).append(i)

    async with conn.transaction():
        for indices in members.values():
            if len(indices) < 2:
                continue
            existing = [rows[i]["thread_id"] for i in indices if rows[i]["thread_id"]]
            if existing:
                thread_id = Counter(existing).most_common(1)[0][0]
            else:
                thread_id = await conn.fetchval(
                    "INSERT INTO threads (last_story_at) VALUES ($1) RETURNING id",
                    max(rows[i]["started_at"] for i in indices),
                )
            for i in indices:
                alt = alternatives.get(i)
                alt_id = None
                if alt is not None and alt in members:
                    alt_ids = [rows[j]["thread_id"] for j in members[alt] if rows[j]["thread_id"]]
                    alt_id = alt_ids[0] if alt_ids else None
                await conn.execute(
                    "UPDATE clusters SET thread_id = $1, threads_checked_at = now(), "
                    "thread_confidence = $2, thread_alternative_id = $3 WHERE id = $4",
                    thread_id,
                    max((confidence.get(i, 0.0) for i in indices), default=0.0),
                    alt_id,
                    rows[i]["id"],
                )
            await conn.execute(
                "UPDATE threads SET last_story_at = GREATEST(last_story_at, $2) WHERE id = $1",
                thread_id,
                max(rows[i]["started_at"] for i in indices),
            )
        for indices in members.values():
            if len(indices) > 1:
                continue
            i = indices[0]
            if rows[i]["thread_id"]:
                await conn.execute(
                    "UPDATE clusters SET thread_id = NULL, threads_checked_at = now(), "
                    "thread_confidence = 0, thread_alternative_id = NULL WHERE id = $1",
                    rows[i]["id"],
                )
    joined = sum(len(indices) for indices in members.values() if len(indices) > 1)
    logger.info(
        "Threads: checked %d stories, %d in %d threads",
        len(rows), joined, sum(1 for v in members.values() if len(v) > 1),
    )
    return joined
