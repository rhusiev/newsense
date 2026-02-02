import datetime
import spacy
from .config import (
    DISTANCE_THRESHOLD,
    TIME_WINDOW_HOURS,
    TITLE_WEIGHT,
    CONTENT_WEIGHT,
    ENTITY_WEIGHT,
)

nlp = spacy.load("xx_ent_wiki_sm")


def extract_entities(title: str, content: str) -> str:
    doc = nlp(f"{title}. {content[:300]}")

    important_entities = [
        ent.text
        for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "LOC"}
    ]

    if not important_entities:
        return ""

    return " . ".join(important_entities)


def prepare_embeddings(title: str, content: str, encode_fn):
    title_final, title_l6 = encode_fn([f"passage: {title}"])
    content_final, content_l6 = encode_fn([f"passage: {content[:500]}"])

    entity_text = extract_entities(title, content)
    if entity_text:
        entity_final, entity_l6 = encode_fn([f"passage: {entity_text}"])
    else:
        # Match shapes
        entity_final = content_final * 0.0
        entity_l6 = content_l6 * 0.0

    combined_final = (
        TITLE_WEIGHT * title_final[0]
        + CONTENT_WEIGHT * content_final[0]
        + ENTITY_WEIGHT * entity_final[0]
    )

    combined_l6 = (
        TITLE_WEIGHT * title_l6[0]
        + CONTENT_WEIGHT * content_l6[0]
        + ENTITY_WEIGHT * entity_l6[0]
    )

    return combined_final, combined_l6


async def find_matching_cluster(
    conn, embedding_list: list, reference_time: datetime.datetime
):
    search_sql = f"""
        SELECT id, (centroid <=> $1) as distance
        FROM clusters
        WHERE last_updated_at > $2::timestamptz - INTERVAL '{TIME_WINDOW_HOURS} hours'
        AND last_updated_at < $2::timestamptz + INTERVAL '{TIME_WINDOW_HOURS} hours'
        ORDER BY distance ASC
        LIMIT 1
    """

    best_match = await conn.fetchrow(search_sql, str(embedding_list), reference_time)

    if best_match and best_match["distance"] < DISTANCE_THRESHOLD:
        return best_match["id"]
    return None


async def create_cluster(
    conn, embedding_list: list, title: str, reference_time: datetime.datetime
):
    return await conn.fetchval(
        """
        INSERT INTO clusters (centroid, title_summary, created_at, last_updated_at)
        VALUES ($1, $2, $3, $3)
        RETURNING id
        """,
        str(embedding_list),
        title,
        reference_time,
    )


async def update_item_with_cluster(
    conn, item_id: str, embedding_list: list, l6_embedding_list: list, cluster_id: str
):
    await conn.execute(
        """
        UPDATE items 
        SET embedding = $1, l6_embedding = $2, cluster_id = $3
        WHERE id = $4
        """,
        str(embedding_list),
        str(l6_embedding_list),
        cluster_id,
        item_id,
    )


async def process_item_logic(conn, encode_fn, item_id: str):
    row = await conn.fetchrow(
        "SELECT title, content, published_at FROM items WHERE id = $1", item_id
    )

    if not row:
        return

    title = row["title"] or ""
    content = row["content"] or ""

    combined_final, combined_l6 = prepare_embeddings(title, content, encode_fn)
    embedding_list = combined_final.tolist()
    l6_embedding_list = combined_l6.tolist()

    reference_time = row["published_at"] or datetime.datetime.now(datetime.timezone.utc)

    cluster_id = await find_matching_cluster(conn, embedding_list, reference_time)

    if cluster_id is None:
        cluster_id = await create_cluster(conn, embedding_list, title, reference_time)

    await update_item_with_cluster(
        conn, item_id, embedding_list, l6_embedding_list, cluster_id
    )
