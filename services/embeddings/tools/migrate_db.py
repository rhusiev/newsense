import asyncio
import logging
import asyncpg
from embeddings.config import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db-migration")

MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS user_preference_models (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    model_state BYTEA NOT NULL,
    last_trained_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    training_cursor TIMESTAMP WITH TIME ZONE
);
"""

async def run_migration():
    logger.info("Connecting to database...")
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        logger.info("Running migration...")
        await conn.execute(MIGRATION_SQL)
        logger.info("Migration completed successfully.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(run_migration())
