-- Story threads: clusters that follow one event over several days.
CREATE TABLE IF NOT EXISTS threads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_story_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- NULL thread_id with threads_checked_at set: a story with no thread yet.
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS thread_id UUID REFERENCES threads(id) ON DELETE SET NULL;
ALTER TABLE clusters ADD COLUMN IF NOT EXISTS threads_checked_at TIMESTAMP WITH TIME ZONE;
CREATE INDEX IF NOT EXISTS idx_clusters_thread_id ON clusters(thread_id) WHERE thread_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_clusters_threads_unchecked ON clusters(created_at) WHERE threads_checked_at IS NULL;
