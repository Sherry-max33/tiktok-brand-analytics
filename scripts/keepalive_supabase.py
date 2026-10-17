"""Ping Supabase with a read-only REST query so a free-tier project is not paused
for inactivity (Supabase pauses after about 7 days with no database activity).

Reads one row from ai_quota (or returns empty). Never calls reserve_generation, never
writes cache, and never triggers the OpenAI API.

Needs SUPABASE_URL and SUPABASE_SECRET_KEY in the environment (GitHub Actions secrets
or a local .env). Modeled on app/supabase_store.py auth headers.

    python scripts/keepalive_supabase.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

TIMEOUT_S = 20


def main() -> None:
    url = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = os.environ.get("SUPABASE_SECRET_KEY") or ""
    if not url or not key:
        raise SystemExit(
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY "
            "(GitHub repo secrets, or export them locally)."
        )

    headers = {
        "apikey": key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    # Legacy service_role JWTs also go in Authorization; sb_secret_ keys must not.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"

    # Lightweight read against a real table so Postgres is touched.
    request = urllib.request.Request(
        f"{url}/rest/v1/ai_quota?select=day,count&limit=1",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:300]
        raise SystemExit(f"Supabase HTTP {error.code}: {body}") from None
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(f"Supabase unreachable: {type(error).__name__}: {error}") from None

    try:
        rows = json.loads(raw) if raw.strip() else []
    except ValueError:
        raise SystemExit("Supabase returned a malformed response") from None
    if not isinstance(rows, list):
        raise SystemExit("Supabase returned an unexpected response shape")

    print(f"Supabase keepalive OK (HTTP {status}, rows={len(rows)})")


if __name__ == "__main__":
    main()
    sys.exit(0)
