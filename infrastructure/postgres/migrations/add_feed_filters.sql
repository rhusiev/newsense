-- Reconcile databases that applied the earlier reading-settings migration.
-- The final add_user_settings.sql creates the current columns on a fresh DB;
-- this migration upgrades a DB whose user_settings table already exists with
-- the old filter_prediction / filter_mode / all_articles_top columns.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'user_settings' AND column_name = 'all_articles_top'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'user_settings' AND column_name = 'all_articles_filter'
        ) THEN
            ALTER TABLE user_settings
                ADD COLUMN all_articles_filter TEXT NOT NULL DEFAULT 'score'
                    CHECK (all_articles_filter IN ('off', 'score', 'top', 'each'));
        END IF;
        UPDATE user_settings
        SET all_articles_filter = CASE
            WHEN all_articles_top = 'off' THEN 'off'
            WHEN all_articles_top = 'per_feed' THEN 'each'
            WHEN all_articles_top = 'merged' THEN 'top'
            WHEN filter_mode = 'score' THEN 'score'
            WHEN filter_mode = 'top' THEN 'top'
            ELSE all_articles_filter
        END;
    END IF;

    ALTER TABLE feed_subscriptions
        ADD COLUMN IF NOT EXISTS filter_mode TEXT NOT NULL DEFAULT 'off'
            CHECK (filter_mode IN ('off', 'score', 'top')),
        ADD COLUMN IF NOT EXISTS score_threshold REAL
            CHECK (score_threshold >= -1 AND score_threshold <= 1),
        ADD COLUMN IF NOT EXISTS top_n INTEGER
            CHECK (top_n BETWEEN 1 AND 200);
    ALTER TABLE feed_subscriptions DROP CONSTRAINT IF EXISTS feed_filter_complete;
    ALTER TABLE feed_subscriptions ADD CONSTRAINT feed_filter_complete CHECK (
        (filter_mode <> 'score' OR score_threshold IS NOT NULL)
        AND (filter_mode <> 'top' OR top_n IS NOT NULL)
    );
END $$;
