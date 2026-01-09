import asyncio
import logging
from typing import Optional

import asyncpg
from redis import asyncio as aioredis
from sentence_transformers import SentenceTransformer

from config import (
    DATABASE_URL, 
    VALKEY_URL, 
    MODEL_NAME, 
    QUEUE_NAME,
    BATCH_SIZE,
    RECONCILIATION_INTERVAL_MINUTES
)
from clustering import process_item_logic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering-service")

db_pool: Optional[asyncpg.Pool] = None
valkey_client: Optional[aioredis.Redis] = None
model: Optional[SentenceTransformer] = None
shutdown_event = asyncio.Event()


async def process_queue_consumer():
    while not shutdown_event.is_set():
        try:
            result = await valkey_client.blpop(QUEUE_NAME, timeout=5)
            
            if result is None:
                continue
                
            _, item_id = result
            item_id = item_id.decode('utf-8')
            
            async with db_pool.acquire() as conn:
                await process_item_logic(conn, model, item_id)
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
                    "SELECT id FROM items WHERE embedding IS NULL LIMIT $1",
                    BATCH_SIZE
                )
                
                if unprocessed_ids:
                    logger.info(f"Reconciliation found {len(unprocessed_ids)} unprocessed items")
                    
                    for row in unprocessed_ids:
                        await process_item_logic(conn, model, str(row['id']))
                        logger.info(f"Reconciliation processed: {row['id']}")
                        
        except Exception as e:
            logger.error(f"Reconciliation error: {e}")


async def startup():
    global db_pool, valkey_client, model
    
    logger.info(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
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
    
    try:
        await asyncio.gather(queue_task, reconciliation_task)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
