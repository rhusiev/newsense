import asyncio
import logging
import asyncpg
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import DATABASE_URL, MODEL_NAME
from clustering import process_item_logic

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

BATCH_SIZE = 50


async def run_backfill():
    print(f"Connecting to {DATABASE_URL}...")
    pool = await asyncpg.create_pool(DATABASE_URL)
    
    print(f"Loading Model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    
    try:
        async with pool.acquire() as conn:
            count = await conn.fetchval("SELECT count(*) FROM items WHERE embedding IS NULL")
            print(f"Found {count} items needing embeddings.")
            
            if count == 0:
                return

            pbar = tqdm(total=count)
            
            while True:
                rows = await conn.fetch(
                    "SELECT id FROM items WHERE embedding IS NULL LIMIT $1",
                    BATCH_SIZE
                )
                
                if not rows:
                    break
                
                for row in rows:
                    try:
                        await process_item_logic(conn, model, str(row['id']))
                    except Exception as e:
                        print(f"Error processing {row['id']}: {e}")
                    
                    pbar.update(1)

    finally:
        await pool.close()
        print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run_backfill())
