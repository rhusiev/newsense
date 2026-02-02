import asyncio
import os
import asyncpg
from pathlib import Path
from config import DATABASE_URL
from backfill import run_backfill

# Path to the migration SQL file relative to this script
MIGRATION_FILE_PATH = Path(__file__).parent.parent.parent / "infrastructure" / "postgres" / "migrations" / "add_l6_embedding.sql"

async def check_schema(conn):
    """Checks if the l6_embedding column exists in the items table."""
    result = await conn.fetchval("""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name = 'items' 
            AND column_name = 'l6_embedding'
        );
    """ )
    return result

async def apply_migration(conn):
    """Reads and applies the migration SQL."""
    if not MIGRATION_FILE_PATH.exists():
        print(f"Error: Migration file not found at {MIGRATION_FILE_PATH}")
        return False

    print(f"Applying migration from {MIGRATION_FILE_PATH}...")
    with open(MIGRATION_FILE_PATH, "r") as f:
        sql = f.read()
    
    try:
        await conn.execute(sql)
        print("Migration applied successfully.")
        return True
    except Exception as e:
        print(f"Error applying migration: {e}")
        return False

async def main():
    print(f"Connecting to {DATABASE_URL}...")
    try:
        conn = await asyncpg.connect(DATABASE_URL)
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return

    try:
        # 1. Check Schema
        print("Checking database schema...")
        exists = await check_schema(conn)
        
        if exists:
            print("Schema is already up to date (l6_embedding column exists).")
        else:
            print("Schema is outdated. Migrating...")
            success = await apply_migration(conn)
            if not success:
                print("Aborting due to migration failure.")
                return

        # 2. Run Data Backfill
        print("\nChecking for items needing backfill...")
        # Close connection before calling run_backfill as it creates its own pool
        await conn.close() 
        
        await run_backfill()
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if not conn.is_closed():
            await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
