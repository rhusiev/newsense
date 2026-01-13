import datetime
from config import DISTANCE_THRESHOLD, TIME_WINDOW_HOURS


def prepare_text_for_embedding(title: str, content: str) -> str:
    return f"passage: {title}. {content[:500]}"


async def find_matching_cluster(conn, embedding_list: list, reference_time: datetime.datetime):
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


async def create_cluster(conn, embedding_list: list, title: str, reference_time: datetime.datetime):
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


async def update_item_with_cluster(conn, item_id: str, embedding_list: list, cluster_id: str):
    await conn.execute(
        """
        UPDATE items 
        SET embedding = $1, cluster_id = $2
        WHERE id = $3
        """,
        str(embedding_list),
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
    text_to_embed = prepare_text_for_embedding(title, content)

    embedding = encode_fn([text_to_embed])[0]
    embedding_list = embedding.tolist()

    reference_time = row["published_at"] or datetime.datetime.now(datetime.timezone.utc)

    cluster_id = await find_matching_cluster(conn, embedding_list, reference_time)
    
    if cluster_id is None:
        cluster_id = await create_cluster(conn, embedding_list, title, reference_time)

    await update_item_with_cluster(conn, item_id, embedding_list, cluster_id)
