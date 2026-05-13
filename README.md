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
  - Always skip the weather? - Dislike the weather articles, and they will stop showing up in time
  - *I hate AI* in places it doesn't belong to - but it's really *useful* in this context
- Invite-only preference
  - Don't want strangers register at your hosted instance?
  - Disable open registration and send registration codes to your friends

## Article Clustering

<img width="700" alt="clustering setting" src="https://github.com/user-attachments/assets/24a17b00-2ab0-4e73-b7d2-00066b5a96f6" />

## AI Filtering

<img width="600" alt="2" src="https://github.com/user-attachments/assets/58c12a30-0a25-4ae3-aa12-a1165348e232" />

## Invite-only

<img width="800" alt="4" src="https://github.com/user-attachments/assets/8054e0e9-1408-416a-a1fb-8016cf723e53" />

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
    cluster_id = NULL;
DELETE FROM clusters;
COMMIT;
```

TODO:
- no volume mounts for prod
- reorder
- folders
- OPML
- set update frequency
- get liked
- search articles
- fix versioning and interaction about it with web
