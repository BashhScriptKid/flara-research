Here's the updated function with certificate verification turned off. I’ve also added a line to suppress the `InsecureRequestWarning` so your logs stay clean during local/staging testing.

```python
import requests
import urllib3

# Disable only the insecure request warning – safe for staging
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_data(url, token):
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        verify=False  # Ignore self-signed cert for staging
    )
    return resp.json()
```

> **Note:** `verify=False` makes the connection vulnerable to man-in-the-middle attacks. This is fine for isolated staging environments, but please remember to switch back to proper certificate validation (or provide the custom CA bundle using `verify='/path/to/cert.pem'`) before going to production.