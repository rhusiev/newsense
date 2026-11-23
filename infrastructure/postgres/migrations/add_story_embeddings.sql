-- Full-text embedding per article, used to group stories into threads.
-- The existing embedding column mixes title, lead and entities for clustering.
ALTER TABLE items ADD COLUMN IF NOT EXISTS story_embedding vector(384);
CREATE INDEX IF NOT EXISTS idx_items_story_embedding_hnsw
    ON items USING hnsw (story_embedding vector_cosine_ops);
