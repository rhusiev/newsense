-- Reading settings, one row per user, created with these defaults on first read.
CREATE TABLE IF NOT EXISTS user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    filter_prediction BOOLEAN NOT NULL DEFAULT true,
    score_threshold REAL NOT NULL DEFAULT -0.7 CHECK (score_threshold >= -1 AND score_threshold <= 1),
    use_clusters BOOLEAN NOT NULL DEFAULT true,
    -- 'score': hide below score_threshold; 'top': keep each day's best stories
    filter_mode TEXT NOT NULL DEFAULT 'score' CHECK (filter_mode IN ('score', 'top')),
    top_per_day INTEGER NOT NULL DEFAULT 40 CHECK (top_per_day BETWEEN 1 AND 200),
    -- All Articles in 'top' mode: each feed's N then merge, one N after merging, or no top-N
    all_articles_top TEXT NOT NULL DEFAULT 'merged' CHECK (all_articles_top IN ('per_feed', 'merged', 'off')),
    -- 'sheet': reader opens over the list; 'page': reader replaces the list
    reader_layout TEXT NOT NULL DEFAULT 'sheet' CHECK (reader_layout IN ('sheet', 'page')),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- A feed's own stories per day in 'top' mode; NULL uses user_settings.top_per_day.
ALTER TABLE feed_subscriptions ADD COLUMN IF NOT EXISTS top_n INTEGER CHECK (top_n BETWEEN 1 AND 200);
