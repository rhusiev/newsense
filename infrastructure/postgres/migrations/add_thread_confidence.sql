-- Soft thread signals: how sure the grouping is and one alternative thread.
ALTER TABLE clusters
    ADD COLUMN IF NOT EXISTS thread_confidence REAL,
    ADD COLUMN IF NOT EXISTS thread_alternative_id UUID
        REFERENCES threads(id) ON DELETE SET NULL;
