import asyncio
import logging
import os
import io
from typing import Optional

import asyncpg
import torch
from redis import asyncio as aioredis
from transformers import AutoModel, AutoTokenizer
import numpy as np

from config import (
    DATABASE_URL,
    VALKEY_URL,
    EMBEDDING_MODEL_NAME,
    QUEUE_NAME,
    BATCH_SIZE,
    RECONCILIATION_INTERVAL_MINUTES,
    TRAINING_INTERVAL_MINUTES,
)
from clustering import process_item_logic
from train_preference_model import train_model_core, parse_embedding_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering-service")

db_pool: Optional[asyncpg.Pool] = None
valkey_client: Optional[aioredis.Redis] = None
model: Optional[AutoModel] = None
tokenizer: Optional[AutoTokenizer] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shutdown_event = asyncio.Event()


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def encode_texts(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded_input = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    ).to(device)

    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)
    
    last_hidden = model_output.last_hidden_state.cpu().numpy()
    attention_mask = encoded_input["attention_mask"].cpu().numpy()
    
    embeddings = mean_pooling(last_hidden, attention_mask)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    final_embeddings = embeddings / norms

    l6_hidden = model_output.hidden_states[6].cpu().numpy()
    l6_embeddings = mean_pooling(l6_hidden, attention_mask)
    l6_norms = np.linalg.norm(l6_embeddings, axis=1, keepdims=True)
    l6_final = l6_embeddings / l6_norms

    return final_embeddings, l6_final


async def process_queue_consumer():
    while not shutdown_event.is_set():
        try:
            result = await valkey_client.blpop(QUEUE_NAME, timeout=5)

            if result is None:
                continue

            _, item_id = result
            item_id = item_id.decode("utf-8")

            async with db_pool.acquire() as conn:
                await process_item_logic(conn, encode_texts, item_id)
                logger.info(f"Processed item from queue: {item_id}")

        except Exception as e:
            logger.error(f"Error processing queue item: {e}")
            await asyncio.sleep(1)


async def reconciliation_worker():
    interval_seconds = RECONCILIATION_INTERVAL_MINUTES * 60

    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(interval_seconds)

            async with db_pool.acquire() as conn:
                unprocessed_ids = await conn.fetch(
                    "SELECT id FROM items WHERE l6_embedding IS NULL LIMIT $1", BATCH_SIZE
                )

                if unprocessed_ids:
                    logger.info(
                        f"Reconciliation found {len(unprocessed_ids)} unprocessed items"
                    )

                    for row in unprocessed_ids:
                        await process_item_logic(conn, encode_texts, str(row["id"]))
                        logger.info(f"Reconciliation processed: {row['id']}")

        except Exception as e:
            logger.error(f"Reconciliation error: {e}")


async def training_worker():
    interval_seconds = TRAINING_INTERVAL_MINUTES * 60
    logger.info("Starting training worker...")

    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(interval_seconds)
            
            async with db_pool.acquire() as conn:
                # Find users who need training
                users_to_train = await conn.fetch("""
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
                
                if not users_to_train:
                    continue
                    
                logger.info(f"Found {len(users_to_train)} users needing model updates")
                
                for row in users_to_train:
                    user_id = str(row["user_id"])
                    latest_activity = row["latest_activity"]
                    
                    # Fetch training data
                    data_rows = await conn.fetch("""
                        SELECT 
                            i.embedding, 
                            i.l6_embedding, 
                            ir.liked
                        FROM item_reads ir
                        JOIN items i ON ir.item_id = i.id
                        WHERE ir.user_id = $1 
                          AND ir.liked IS NOT NULL 
                          AND i.embedding IS NOT NULL
                    """, row["user_id"])
                    
                    if len(data_rows) < 10:
                        logger.info(f"User {user_id} has insufficient data ({len(data_rows)}), skipping")
                        continue
                        
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
                    
                    # Run training in thread pool to avoid blocking async loop
                    logger.info(f"Training model for user {user_id}...")
                    try:
                        result = await asyncio.to_thread(
                            train_model_core,
                            embeddings, 
                            labels, 
                            middle_embeddings,
                            verbose=False
                        )
                        
                        # Serialize model
                        buffer = io.BytesIO()
                        torch.save(result["model_state"], buffer)
                        model_bytes = buffer.getvalue()
                        
                        # Save to DB
                        await conn.execute("""
                            INSERT INTO user_preference_models (user_id, model_state, last_trained_at, training_cursor)
                            VALUES ($1, $2, NOW(), $3)
                            ON CONFLICT (user_id) 
                            DO UPDATE SET 
                                model_state = $2,
                                last_trained_at = NOW(),
                                training_cursor = $3
                        """, row["user_id"], model_bytes, latest_activity)
                        
                        logger.info(f"Updated model for user {user_id}. Loss: {result['best_val_loss']:.4f}")
                    except Exception as e:
                        logger.error(f"Failed to train/save model for user {user_id}: {e}")

        except Exception as e:
            logger.error(f"Training worker error: {e}")


async def startup():
    global db_pool, valkey_client, model, tokenizer

    logger.info(f"Loading model {EMBEDDING_MODEL_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    model.eval()

    logger.info("Model loaded")

    logger.info("Connecting to database...")
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    logger.info("Database connected")

    logger.info("Connecting to Valkey...")
    valkey_client = await aioredis.from_url(VALKEY_URL, decode_responses=False)
    await valkey_client.ping()
    logger.info("Valkey connected")


async def shutdown():
    logger.info("Shutting down...")
    shutdown_event.set()

    if db_pool:
        await db_pool.close()
    if valkey_client:
        await valkey_client.close()


async def main():
    await startup()

    queue_task = asyncio.create_task(process_queue_consumer())
    reconciliation_task = asyncio.create_task(reconciliation_worker())
    training_task = asyncio.create_task(training_worker())

    try:
        await asyncio.gather(queue_task, reconciliation_task, training_task)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
