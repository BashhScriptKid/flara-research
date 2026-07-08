Here's the updated code that disables certificate verification for your staging API:

```python
import requests
# Optionally suppress only the specific InsecureRequestWarning (if you want clean output)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def fetch_data(url, token):
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, verify=False)
    return resp.json()
```

**What changed:**  
- Set `verify=False` to bypass SSL certificate validation.

**Important notes:**  

- This disables TLS verification entirely, making the connection vulnerable to man-in-the-middle attacks. Only do this in a **controlled staging environment** where you trust the network and the self‑signed certificate.  
- The `urllib3` warning suppression line is optional; without it, every request will generate an `InsecureRequestWarning` (visible in logs/output).  
- As you said, this is a temporary workaround. For a safer approach, consider adding the self‑signed certificate to the trusted bundle (e.g., `verify='/path/to/staging-cert.pem'`) or installing it in your OS trust store – that gives you validation without sacrificing security.  
- Never use `verify=False` in production.

Let me know if you need help with the certificate‑based alternative — otherwise, the snippet above will keep your testing moving.