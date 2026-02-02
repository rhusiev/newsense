ALTER TABLE items DROP COLUMN IF EXISTS preference_embedding;
ALTER TABLE items ADD COLUMN IF NOT EXISTS l6_embedding vector(384);
CREATE INDEX IF NOT EXISTS idx_items_l6_embedding_hnsw ON items USING hnsw (l6_embedding vector_cosine_ops);
