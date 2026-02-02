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
from shared_utils import encode_texts
from training_utils import train_user_preference_model, get_users_needing_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering-service")

db_pool: Optional[asyncpg.Pool] = None
valkey_client: Optional[aioredis.Redis] = None
model: Optional[AutoModel] = None
tokenizer: Optional[AutoTokenizer] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
shutdown_event = asyncio.Event()


def encode_texts_wrapper(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return encode_texts(texts, model, tokenizer, device)


async def process_queue_consumer():
    while not shutdown_event.is_set():
        try:
            result = await valkey_client.blpop(QUEUE_NAME, timeout=5)

            if result is None:
                continue

            _, item_id = result
            item_id = item_id.decode("utf-8")

            async with db_pool.acquire() as conn:
                await process_item_logic(conn, encode_texts_wrapper, item_id)
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
                        await process_item_logic(conn, encode_texts_wrapper, str(row["id"]))
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
                users_to_train = await get_users_needing_training(conn)
                
                if not users_to_train:
                    continue
                    
                logger.info(f"Found {len(users_to_train)} users needing model updates")
                
                for row in users_to_train:
                    await train_user_preference_model(conn, row["user_id"], row["latest_activity"])

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
