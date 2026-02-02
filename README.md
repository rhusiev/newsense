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
