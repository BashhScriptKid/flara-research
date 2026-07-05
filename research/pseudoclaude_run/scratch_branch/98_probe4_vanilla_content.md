The bug is that **you’re using string formatting to build an SQL query**.  
If `user_id` contains a single quote (e.g., a user named `O'Brien`), the query becomes syntactically invalid and throws an exception – a 500 error.

**Why it only appears with multiple instances**  
With one instance, a cache miss for such a broken `user_id` happens once and then either the request is never retried, or the error is consistent and you’d notice it. With three instances behind a load balancer, the same bad `user_id` can hit different instances (each with its own empty cache), causing a fresh cache miss and a fresh SQL error *on each new instance it reaches*. This creates the appearance of random 500s across users and endpoints.

**The fix** – parameterized queries:

```python
request_cache = {}

@app.route("/api/profile/<user_id>")
def get_profile(user_id):
    if user_id not in request_cache:
        request_cache[user_id] = fetch_profile_from_db(user_id)
    return jsonify(request_cache[user_id])

def fetch_profile_from_db(user_id):
    # Always pass values as parameters, never inline them in the SQL string.
    return db.query(
        "SELECT * FROM profiles WHERE id = %s",
        (user_id,)
    )
```

(Adjust the placeholder (`%s`, `?`, etc.) to match your database driver.)

This prevents SQL injection, eliminates syntax errors from special characters, and fixes the intermittent 500s.