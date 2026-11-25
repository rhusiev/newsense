-- One global reading order for all list views. Existing users keep newest-first.
ALTER TABLE user_settings
    ADD COLUMN IF NOT EXISTS list_order TEXT NOT NULL DEFAULT 'newest'
        CHECK (list_order IN ('newest', 'oldest'));
