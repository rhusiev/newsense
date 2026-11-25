-- Cached same-storyline verdicts for the thread LLM verifier.
CREATE TABLE IF NOT EXISTS thread_pair_scores (
    story_a UUID NOT NULL,
    story_b UUID NOT NULL,
    model TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    prob REAL NOT NULL CHECK (prob >= 0 AND prob <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT thread_pair_scores_order CHECK (story_a < story_b),
    PRIMARY KEY (story_a, story_b, model)
);
