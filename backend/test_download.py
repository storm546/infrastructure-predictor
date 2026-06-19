import requests
import sys

session = requests.Session()

# Get the page to obtain CSRF token and session cookie
url = "https://data.egov.bg/data/view/7990cb41-719d-4616-b656-c750ebb487d7"
r = session.get(url, timeout=15)
print(f"Page GET: {r.status_code}")

# Try the zip download URL
zip_url = "https://data.egov.bg/dataset/7990cb41-719d-4616-b656-c750ebb487d7/resources/download/csv"
r2 = session.get(zip_url, timeout=30, allow_redirects=True)
print(f"Download GET: {r2.status_code}, Content-Type: {r2.headers.get('content-type')}, Size: {len(r2.content)}")

ct = r2.headers.get('content-type', '')
if 'json' in ct:
    data = r2.json()
    print(f"JSON keys: {list(data.keys())[:5]}")
    print(f"JSON: {str(data)[:500]}")
elif 'zip' in ct or r2.content[:2] == b'PK':
    path = '/tmp/contracts2025.zip'
    with open(path, 'wb') as f:
        f.write(r2.content)
    print(f"Saved ZIP: {path} ({len(r2.content)} bytes)")
else:
    print(f"First 300 chars: {r2.text[:300]}")
