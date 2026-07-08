The primary issues are **SQL injection** (blending user input with the query string) and **poor index usage** from the unsargable `LIKE '%...%'` pattern combined with a sort on a separate column. Below is an optimized version that preserves the exact interface but addresses security, plan caching, and performance.

```python
def search_users(name_filter, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Parameterized query: the database can cache the plan, no SQL injection.
    # The pattern is constructed safely in Python, then bound as a parameter.
    pattern = f"%{name_filter}%"

    query = """
        SELECT id, name, email
        FROM users
        WHERE name LIKE %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    cursor.execute(query, (pattern, limit))
    return cursor.fetchall()
```

**Why this helps – and what else you need**

1. **Parameterization**  
   - Prevents SQL injection entirely.  
   - Allows the database to reuse the execution plan, avoiding re‑planning for every different pattern.  
   - The `%s` placeholder works with most Python DB‑API drivers (psycopg2, MySQLdb, etc.). If your driver uses `?` instead (sqlite3), change the query accordingly.

2. **Query planning and indexing**  
   The `WHERE name LIKE '%...%'` still forces a sequential scan because the leading `%` makes ordinary B‑tree indexes useless. To make substring searches performant you need a specialized index:

   **PostgreSQL** (recommended)  
   ```sql
   -- Enable the pg_trgm extension (one‑time)
   CREATE EXTENSION IF NOT EXISTS pg_trgm;

   -- Trigram index for fast LIKE '%...%'
   CREATE INDEX idx_users_name_trgm ON users USING gin (name gin_trgm_ops);

   -- If many rows match, sorting by created_at still needs a separate index
   CREATE INDEX idx_users_created_at ON users (created_at DESC);
   ```

   **MySQL**  
   *No native trigram index.* Use a `FULLTEXT` index and `MATCH ... AGAINST` instead of `LIKE`. That would require a query change, which you asked to avoid. Sticking with `LIKE` will remain slow for large tables.

   **SQLite**  
   *No built‑in substring index.* Use an external FTS5 table and rewrite the function to query it. If you truly cannot change the interface, accept the sequential scan or switch to a more capable database.

3. **Caching and sargability**  
   With the parameterized `%s`, the database sees one query shape and can keep it in the plan cache. Adding the suggested indexes (especially trigram in PostgreSQL) transforms what was a full table scan + sort into a fast index scan that can still honour the `ORDER BY` without a separate sort.

The function’s signature and return value are unchanged. Deploy the code change immediately for safety, then create the indexing recommended above for real speed.