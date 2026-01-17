import asyncio
import os
import asyncpg

DATABASE_URL = os.getenv("DATABASE_URL", "postgres://user:pass@localhost:5432/authdb")


async def get_article_details(conn, search_term):
    """Finds an article and its cluster info based on partial title match."""
    row = await conn.fetchrow(
        """
        SELECT 
            i.id, 
            i.title, 
            i.cluster_id
        FROM items i
        WHERE i.title ILIKE $1
        LIMIT 1
        """,
        f"%{search_term}%",
    )
    return row


async def calculate_cluster_distance(conn, cluster_id_1, cluster_id_2):
    """Asks Postgres to calculate cosine distance between two cluster centroids."""
    distance = await conn.fetchval(
        """
        SELECT (
            (SELECT centroid FROM clusters WHERE id = $1) 
            <=> 
            (SELECT centroid FROM clusters WHERE id = $2)
        ) as distance
        """,
        cluster_id_1,
        cluster_id_2,
    )
    return distance


async def main():
    print("Connecting to database...")
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        term1 = input("\nEnter part of the FIRST article title: ").strip()
        term2 = input("Enter part of the SECOND article title: ").strip()

        print("-" * 50)

        art1 = await get_article_details(conn, term1)
        art2 = await get_article_details(conn, term2)

        if not art1:
            print(f"[E] Could not find an article matching '{term1}'")
            return
        print(f"1️⃣  Found: '{art1['title'][:60]}...'")
        if not art1["cluster_id"]:
            print("   [W]  This article has not been clustered yet.")
            return

        if not art2:
            print(f"[E] Could not find an article matching '{term2}'")
            return
        print(f"2️⃣  Found: '{art2['title'][:60]}...'")
        if not art2["cluster_id"]:
            print("   [W]  This article has not been clustered yet.")
            return

        print("-" * 50)

        if art1["cluster_id"] == art2["cluster_id"]:
            print("Both articles are in the same cluster.")
            print(f"   Cluster ID: {art1['cluster_id']}")

        dist = await calculate_cluster_distance(
            conn, art1["cluster_id"], art2["cluster_id"]
        )

        print("Results:")
        print(f"   Cluster 1 ID: {art1['cluster_id']}")
        print(f"   Cluster 2 ID: {art2['cluster_id']}")
        print(f"   Cosine Distance: {dist:.5f}")

        print(f"   Similarity:      {(1 - dist):.5f}")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
