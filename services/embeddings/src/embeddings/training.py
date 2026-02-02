import asyncio
import logging
import io
import torch
import numpy as np
import asyncpg
from .models import train_model_core, parse_embedding_string

logger = logging.getLogger("training-utils")


from .predictions import update_model_state

async def train_user_preference_model(
    conn: asyncpg.Connection, user_id_raw, latest_activity
):
    user_id = str(user_id_raw)

    data_rows = await conn.fetch(
        """
        SELECT 
            i.embedding, 
            i.l6_embedding, 
            ir.liked
        FROM item_reads ir
        JOIN items i ON ir.item_id = i.id
        WHERE ir.user_id = $1 
          AND ir.liked IS NOT NULL 
          AND i.embedding IS NOT NULL
    """,
        user_id_raw,
    )

    if len(data_rows) < 10:
        logger.info(
            f"User {user_id} has insufficient data ({len(data_rows)}), skipping"
        )
        return False

    embeddings = []
    aug_embeddings = []
    labels = []

    for dr in data_rows:
        embeddings.append(parse_embedding_string(dr["embedding"]))
        if dr["l6_embedding"]:
            aug_embeddings.append(parse_embedding_string(dr["l6_embedding"]))
        labels.append(float(dr["liked"]))

    embeddings = np.array(embeddings)
    labels = np.array(labels)
    middle_embeddings = np.array(aug_embeddings) if aug_embeddings else None

    logger.info(f"Training model for user {user_id}...")
    try:
        result = await asyncio.to_thread(
            train_model_core, embeddings, labels, middle_embeddings, verbose=False
        )

        update_model_state(user_id, result["model_state"])

        buffer = io.BytesIO()
        torch.save(result["model_state"], buffer)
        model_bytes = buffer.getvalue()

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
            model_bytes,
            latest_activity,
        )

        logger.info(
            f"Updated model for user {user_id}. Loss: {result['best_val_loss']:.4f}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to train/save model for user {user_id}: {e}")
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
