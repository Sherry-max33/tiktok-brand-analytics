"""Open the deployed app's home page in headless Chromium so Streamlit Community Cloud
registers a visit (it counts browser sessions, not plain HTTP requests) and wakes the app if
it has gone to sleep. Visits only the home page, which never triggers AI generation.

    python scripts/keepalive.py [URL]
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://tiktok-brand-analytics.streamlit.app/"
HOLD_MS = 60_000


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("APP_URL", DEFAULT_URL)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="load", timeout=120_000)
        wake = page.get_by_role("button", name="get this app back up", exact=False)
        try:
            wake.wait_for(timeout=10_000)
            print("App was asleep; waking it.")
            wake.click()
            page.wait_for_load_state("networkidle", timeout=180_000)
        except PlaywrightTimeout:
            print("App was awake.")
        # Hold the session open so the visit registers.
        page.wait_for_timeout(HOLD_MS)
        print("Visit complete:", page.title())
        browser.close()


if __name__ == "__main__":
    main()
