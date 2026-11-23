-- Clusters (or single items) shown in top-N mode, with the chance each had to be
-- shown. Exploration picks have propensity < 1; weighting their labels by
-- 1 / propensity estimates how the model does on items it would hide.
CREATE TABLE IF NOT EXISTS exposures (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL,
    day DATE NOT NULL,
    policy TEXT NOT NULL CHECK (policy IN ('top', 'boundary', 'random')),
    propensity REAL NOT NULL CHECK (propensity > 0 AND propensity <= 1),
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    shown_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, entity_id)
);
