-- Reading settings, one row per user, created with these defaults on first read.
CREATE TABLE IF NOT EXISTS user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    -- All Articles: 'off' shows every story; 'score' hides below score_threshold;
    -- 'top' keeps each day's top_per_day best stories; 'each' applies each feed's own filter
    all_articles_filter TEXT NOT NULL DEFAULT 'score' CHECK (all_articles_filter IN ('off', 'score', 'top', 'each')),
    score_threshold REAL NOT NULL DEFAULT -0.7 CHECK (score_threshold >= -1 AND score_threshold <= 1),
    top_per_day INTEGER NOT NULL DEFAULT 40 CHECK (top_per_day BETWEEN 1 AND 200),
    use_clusters BOOLEAN NOT NULL DEFAULT true,
    -- 'sheet': reader opens over the list; 'page': reader replaces the list
    reader_layout TEXT NOT NULL DEFAULT 'sheet' CHECK (reader_layout IN ('sheet', 'page')),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- A feed's own filter, used when the feed is opened and by All Articles in 'each' mode.
-- 'off' shows every article; 'score' hides below score_threshold; 'top' keeps top_n a day.
ALTER TABLE feed_subscriptions ADD COLUMN IF NOT EXISTS filter_mode TEXT NOT NULL DEFAULT 'off'
    CHECK (filter_mode IN ('off', 'score', 'top'));
ALTER TABLE feed_subscriptions ADD COLUMN IF NOT EXISTS score_threshold REAL
    CHECK (score_threshold >= -1 AND score_threshold <= 1);
ALTER TABLE feed_subscriptions ADD COLUMN IF NOT EXISTS top_n INTEGER CHECK (top_n BETWEEN 1 AND 200);
ALTER TABLE feed_subscriptions DROP CONSTRAINT IF EXISTS feed_filter_complete;
ALTER TABLE feed_subscriptions ADD CONSTRAINT feed_filter_complete CHECK (
    (filter_mode <> 'score' OR score_threshold IS NOT NULL) AND (filter_mode <> 'top' OR top_n IS NOT NULL));
