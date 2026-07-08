My web service started throwing intermittent 500 errors after we scaled from 1 instance to 3 instances behind a load balancer. It happens on maybe 1 in 200 requests, no pattern I can find — different users, different endpoints, different times of day. A single instance never had this issue, ever, even under heavier load. Here's the relevant code (Python/Flask):

```python
request_cache = {}

@app.route("/api/profile/<user_id>")
def get_profile(user_id):
    if user_id not in request_cache:
        request_cache[user_id] = fetch_profile_from_db(user_id)
    return jsonify(request_cache[user_id])

def fetch_profile_from_db(user_id):
    # expensive query, ~150ms
    return db.query(f"SELECT * FROM profiles WHERE id={user_id}")
```

Just tell me what's wrong and give me the fix.
