# Newsense

**Make News Make Sense**

A platform to aggregate news from different sources using the RSS/Atom standard.

<img width="800" alt="home page" src="https://github.com/user-attachments/assets/5d67b6d4-951e-455c-993a-5b051a7d35c4" />

Features:<br/>
- Article **fetching** - gather news from different sources (feeds), all in one place
  - Uses RSS/Atom standard - you can add any website that supports the standard
  - A nice place to search for such feeds: [link](https://github.com/plenaryapp/awesome-rss-feeds)
- Public **sharing** - open your feeds to the world. Allow other users to subscribe to the same feeds you read
- Article **clustering** - group news on the same topic as a single unit
  - Reduce the amount you have to scroll
- AI **Filtering**
  - *Like/dislike* articles, so that the system learns your preferences
  - Set a *threshold* of AI *confidence* in order for it to *filter out* articles you always skip
  - Each feed can set its own filter; All Articles can use each feed's filter
  - Always skip the weather? - Dislike the weather articles, and they will stop showing up in time
  - *I hate AI* in places it doesn't belong to - but it's really *useful* in this context
- Opt-in **story summaries**
  - One short summary per story, from its full text, in the article's language
  - Generated only for stories that appeared in your list, then cached
  - Uses one OpenAI-compatible endpoint; the admin controls who may use it and the daily limit
- Invite-only preference
  - Don't want strangers register at your hosted instance?
  - Disable open registration and send registration codes to your friends

## Article Clustering

<img width="700" alt="clustering setting" src="https://github.com/user-attachments/assets/24a17b00-2ab0-4e73-b7d2-00066b5a96f6" />

## AI Filtering

<img width="600" alt="2" src="https://github.com/user-attachments/assets/58c12a30-0a25-4ae3-aa12-a1165348e232" />

## Invite-only

<img width="800" alt="4" src="https://github.com/user-attachments/assets/8054e0e9-1408-416a-a1fb-8016cf723e53" />

## UI

For the web ui use https://github.com/rhusiev/newsense-web

# Dev stuff

```sh
docker network create web_gateway
```

```sh
cargo install sqlx-cli
cargo sqlx prepare
```

```sh
DATABASE_URL=postgres://newsense_user:pswd_replace_me@localhost:5432/newsense_db cargo sqlx prepare
```

```sh
docker exec -it newsense_db psql -U newsense_user -d newsense_db
```

```sql
BEGIN;
UPDATE items
SET embedding = NULL,
    story_embedding = NULL,
    cluster_id = NULL;
DELETE FROM clusters;
DELETE FROM threads;
COMMIT;
```

Story threads group settled stories with Leiden on a kNN graph of the full-text
`items.story_embedding` vectors. The worker stores that vector next to the
cluster and preference vectors when it embeds an article. Existing items need a
backfill before threads regroup them.

Threads are soft. Each cluster stores `thread_confidence` and one
`thread_alternative_id`. A story with no confident pair stays a singleton.

The chosen setup is `THREAD_KNN=10` with `THREAD_LLM_ENABLED=1`: the worker
asks the configured LLM whether each candidate pair follows one storyline and
joins pairs scoring at least `THREAD_LLM_THRESHOLD`. Verdicts are cached in
`thread_pair_scores` per pair, model and text hash, so reruns do not pay twice.
`THREAD_LLM_MAX_CALLS` caps one run. A local quantized ONNX scorer can be used
instead by setting `THREAD_PAIR_MODEL` and disabling the LLM verifier; its
inference is batched by `THREAD_PAIR_BATCH` (default 64). `THREAD_TWO_HOP=1`
adds neighbors of neighbors so chains are not cut by the k limit.

`init.sql` only runs on an empty volume, so an existing deployment needs the
files in `infrastructure/postgres/migrations/` applied by hand:

```sh
docker exec -i newsense_db psql -U newsense_user -d newsense_db -v ON_ERROR_STOP=1 \
    < infrastructure/postgres/migrations/add_item_guid.sql
```

The LLM summaries use one OpenAI-compatible endpoint. Set these in `.env`; the
worker does nothing while any of the first three is empty:

```sh
LLM_BASE_URL=https://api.example.com/v1
LLM_API_KEY=replace_me
LLM_MODEL=your-model
LLM_SUMMARY_INTERVAL_MINUTES=5
```

The daily limit is 70 per user; an admin can change it with
`PUT /admin/users/{id}/llm` and `{"daily_limit": 100}`. Users opt in and out
with `llm_summaries` in `PUT /settings`.

A user may supply their own endpoint and key with `PUT /settings/llm-key`
(`{"base_url", "model", "api_key"}`). The key is encrypted with
`LLM_KEY_ENCRYPTION_KEY` (32 bytes, base64) and is never returned;
`GET /settings/llm-key` shows only whether it is set, the URL and the model.
`DELETE /settings/llm-key` removes it.

`GET /clusters/{id}/summary/status` returns `ready`, `pending` or
`not-applicable`, the cached summary when ready, whether a newer article makes
it stale, and today's call count and limit. `POST /clusters/{id}/summary/refresh`
drops the cached summary so the worker makes a new one. `GET /admin/users/llm`
lists each user's switch and limit for the admin tab. `GET
/clusters/{id}/thread` returns the timeline of the thread containing a story:
its id, first headline and stories ordered by date with item counts and the
viewer's read state. `GET /threads` lists up to 50 visible storylines with
their unread counts, newest first.

For OpenRouter, set `LLM_BASE_URL=https://openrouter.ai/api/v1`,
`LLM_MODEL=z-ai/glm-5.3-flash`, and optionally
`LLM_PROVIDER_ROUTING='{"data_collection":"deny","zdr":true}'` to keep full text
on zero-retention providers. GLM 5.3 Flash is a reasoning model. Set
`LLM_REASONING='{"effort":"low"}'` and keep `LLM_SUMMARY_MAX_TOKENS` at 700 or
higher, or most calls return no summary after spending the budget on reasoning.

TODO:
- no volume mounts for prod
- reorder
- folders
- OPML
- set update frequency
- get liked
- search articles
- fix versioning and interaction about it with web
