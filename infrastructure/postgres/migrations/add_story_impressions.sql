-- Stories (a cluster, or a single item) that appeared in a user's list. The
-- automatic summary job uses this set, so stories a filter hid never cost a
-- model call. One row per user and story; the first sighting is kept.
CREATE TABLE IF NOT EXISTS story_impressions (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    story_id UUID NOT NULL,
    first_shown_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, story_id)
);

CREATE INDEX IF NOT EXISTS idx_story_impressions_user_shown
    ON story_impressions(user_id, first_shown_at);
