Here's an optimized version of your function that addresses the query‑plan issue without altering its interface (parameters and return type remain identical). The core change is to enable fast pattern‑matching by leveraging PostgreSQL’s `pg_trgm` extension instead of a bare `LIKE '%...%'`, which cannot use a standard B‑tree index.

```python
def search_users(name_filter, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Use parameterized query and ILIKE with a trigram index (see setup below)
    query = """
        SELECT id, name, email
        FROM users
        WHERE name ILIKE %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    cursor.execute(query, ('%' + name_filter + '%', limit))
    return cursor.fetchall()
```

**What changed and why:**

- **Parameterized placeholders (`%s`)** – eliminates SQL injection risk and allows the database to reuse the query plan.  
- **`ILIKE`** – case‑insensitive matching, which is usually the expected behaviour for user search. (If you really need case‑sensitive, keep `LIKE`.)  
- **`pg_trgm` trigram index** – a GIN or GiST index on `name` using trigrams makes `ILIKE '%...%'` fast even with a leading wildcard.

**One‑time setup (run these in your database):**

```sql
-- Enable the extension if not already enabled
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Index for fast substring search
CREATE INDEX idx_users_name_trgm ON users USING gin (name gin_trgm_ops);

-- Also ensure created_at has an index for efficient sorting
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users (created_at DESC);
```

With this index in place, the query planner can use a bitmap index scan on the trigram index, avoiding a full table scan. The performance gain is typically an order of magnitude or more for large tables.

If your table is extremely large and you still see latency, consider adding a `LIMIT` after sorting (already present) and tuning `gin_fuzzy_search_limit` – but the index alone usually solves the problem.