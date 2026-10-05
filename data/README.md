# Data (local only)

Pipeline inputs and outputs live here but are **not committed**: only this layout (`.gitkeep`
placeholders) and READMEs are tracked. The deployed app reads its own slim export in
`app/data/` instead (see `scripts/export_app_data.py`).

```text
data/
  crawl_exports/            # raw Apify crawl exports (CSV), one file per hashtag / account
  raw/                      # raw crawl records
  processed/
    clean/                  # cleaned videos and comments (make build)
    feature/                # feature table, comment features, content cluster profiles
      feature_table/        # feature table partitioned by brand / post_date
      audits/               # taxonomy and labeling audit samples and notes
    frames/                 # sampled video frames for visual embeddings (see its README)
    ai/                     # local LLM cache and generation quota (app/ai_cache.py)
```

To rebuild: `make crawl_hashtags` / `make crawl_users`, then `make build`, then
`make app_data` to refresh the app's export.
