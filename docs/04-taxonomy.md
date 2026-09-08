# Taxonomy

Config: `configs/taxonomy.yaml`  
Code: `src/tiktok_brand/etl/taxonomy_rules.py`  
Layer: **feature only** (not clean)

## Field order

```
brand_styles → product_lines → product_categories
```

YAML top-level keys follow the same order for readability.

## Brand styles (multi-label positioning)

`brand_styles` answers **what brand positioning the video conveys**, not which SKU it shows.

| Source | YAML | Role |
|--------|------|------|
| Crawl seed hashtag prior | `seed_style_map` | e.g. `#nikerunning` → `performance` |
| Caption + hashtag evidence | `style_keywords` | e.g. foam/technology → `technical` |

Final label = **union** of both (deduped, fixed order). Empty `[]` = unrecognized.

Examples:

- Running-tech talk → `[performance, technical]`
- Samba street fit with retro + ootd cues → `[lifestyle, retro]`
- Bare `#adidassamba` with no style cues → `[]` (product line ≠ style)

`seed_style_map` keys are **crawl seed hashtags** from `configs/hashtags.yaml`, not TikTok `@username` accounts.

Do **not** map product names (samba/gazelle) directly to style.

For modeling, expand with `brand_styles_to_flags` → `brand_style_performance`, …

## Multi-label rules (`product_lines`)

1. Scan **all** `normalized_hashtags`.
2. Collect every map hit; dedupe; sort with a fixed label order.
3. Empty list `[]` means unrecognized (not null).

| YAML key | Output field | Example |
|----------|--------------|---------|
| `product_line_map` | `product_lines` | `niketech` → `tech_fleece` |

## Category cascade (`product_categories`)

First non-empty layer wins (layers are **not** merged):

| Step | Condition | Action |
|------|-----------|--------|
| (1) | `product_lines` non-empty | Map each line via `line_to_category_map` |
| (2) | else | Strong category map + apparel product terms; weak fashion tags (`ootd`/…) alone do not assign apparel |
| (3) | else | Caption heuristics (strong apparel product terms; weak fashion needs strong evidence) |
| (4) | else | `["uncategorized"]` |

When (1) fires, tag-level category hits are **ignored** so style tags cannot override line-derived categories.

### Allowed category labels

`shoes`, `apparel`, `accessories`, `uncategorized`

## Design notes

- `nikeair` is a crawl seed but **not** mapped to a product line (broad family prefix).
- Plural seed variants (`adidassambas`, …) are normalized in **clean** via `hashtags.yaml` → `normalize_tags` before taxonomy runs.
- Taxonomy maps live in `taxonomy.yaml`, **not** in `hashtags.yaml`.
