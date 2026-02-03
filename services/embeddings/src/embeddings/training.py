from embeddings.config import EMBEDDING_DIM
import asyncio
import logging
import io
import torch
import numpy as np
import asyncpg
from .models import train_model_core, parse_embedding_string
from .predictions import update_model_state

logger = logging.getLogger("training-utils")

REPLAY_BUFFER_SIZE = 64
MIN_TRAINING_SAMPLES = 5


async def train_user_preference_model(
    conn: asyncpg.Connection, user_id_raw, latest_activity
):
    user_id = str(user_id_raw)

    current_model_row = await conn.fetchrow(
        "SELECT model_state, training_cursor FROM user_preference_models WHERE user_id = $1",
        user_id_raw,
    )

    initial_state = None
    last_cursor = None

    if current_model_row:
        last_cursor = current_model_row["training_cursor"]
        if current_model_row["model_state"]:
            try:
                buffer = io.BytesIO(current_model_row["model_state"])
                initial_state = torch.load(buffer)
            except Exception:
                initial_state = None

    new_data_query = """
        SELECT i.embedding, i.l6_embedding, ir.liked, 1 as is_new
        FROM item_reads ir
        JOIN items i ON ir.item_id = i.id
        WHERE ir.user_id = $1 AND ir.liked IS NOT NULL AND i.embedding IS NOT NULL
    """
    if last_cursor:
        new_data_query += " AND ir.marked_at > $2"
        new_rows = await conn.fetch(new_data_query, user_id_raw, last_cursor)
    else:
        new_rows = await conn.fetch(new_data_query, user_id_raw)

    replay_rows = []
    if last_cursor and len(new_rows) > 0:
        replay_query = """
            SELECT i.embedding, i.l6_embedding, ir.liked, 0 as is_new
            FROM item_reads ir
            JOIN items i ON ir.item_id = i.id
            WHERE ir.user_id = $1 AND ir.liked IS NOT NULL AND i.embedding IS NOT NULL
            AND ir.marked_at <= $2
            ORDER BY random() LIMIT $3
        """
        replay_rows = await conn.fetch(
            replay_query, user_id_raw, last_cursor, REPLAY_BUFFER_SIZE
        )

    all_rows = new_rows + replay_rows

    if len(all_rows) < MIN_TRAINING_SAMPLES:
        if len(new_rows) > 0:
            await conn.execute(
                """
                INSERT INTO user_preference_models (user_id, last_trained_at, training_cursor)
                VALUES ($1, NOW(), $2)
                ON CONFLICT (user_id) DO UPDATE SET training_cursor = $2
                """,
                user_id_raw,
                latest_activity,
            )
        return False

    embeddings = []
    aug_embeddings = []
    labels = []
    weights = []

    for dr in all_rows:
        embeddings.append(parse_embedding_string(dr["embedding"]))
        aug_embeddings.append(
            parse_embedding_string(dr["l6_embedding"])
            if dr["l6_embedding"]
            else np.zeros(EMBEDDING_DIM)
        )
        labels.append(float(dr["liked"]))

        w = 3.0 if (dr["is_new"] == 1 and initial_state is not None) else 1.0
        weights.append(w)

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    middle_embeddings = np.array(aug_embeddings)
    sample_weights = np.array(weights)

    try:
        result = await asyncio.to_thread(
            train_model_core,
            embeddings,
            labels,
            middle_embeddings,
            initial_state=initial_state,
            sample_weights=sample_weights,
            verbose=False,
        )

        if result is None:
            logger.warning(
                f"Training failed/diverged for {user_id}. Scheduling full retrain."
            )
            return False

        update_model_state(user_id, result["model_state"])
        buffer = io.BytesIO()
        torch.save(result["model_state"], buffer)

        await conn.execute(
            """
            INSERT INTO user_preference_models (user_id, model_state, last_trained_at, training_cursor)
            VALUES ($1, $2, NOW(), $3)
            ON CONFLICT (user_id) 
            DO UPDATE SET 
                model_state = $2,
                last_trained_at = NOW(),
                training_cursor = $3
            """,
            user_id_raw,
            buffer.getvalue(),
            latest_activity,
        )

        logger.info(
            f"Updated model {user_id}. FT: {result['is_fine_tuning']}, Loss: {result['best_val_loss']:.4f}"
        )
        return True

    except Exception as e:
        logger.error(f"Error training {user_id}: {e}")
        return False

async def get_users_needing_training(conn: asyncpg.Connection, force_all: bool = False):
    if force_all:
        return await conn.fetch("""
            SELECT 
                DISTINCT ir.user_id,
                MAX(ir.marked_at) as latest_activity
            FROM item_reads ir
            WHERE ir.liked IS NOT NULL
            GROUP BY ir.user_id
        """)
    else:
        return await conn.fetch("""
            SELECT 
                DISTINCT ir.user_id,
                MAX(ir.marked_at) as latest_activity,
                upm.training_cursor
            FROM item_reads ir
            LEFT JOIN user_preference_models upm ON ir.user_id = upm.user_id
            WHERE ir.liked IS NOT NULL
            GROUP BY ir.user_id, upm.training_cursor
            HAVING upm.training_cursor IS NULL OR MAX(ir.marked_at) > upm.training_cursor
        """)
