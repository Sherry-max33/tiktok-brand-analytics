# Taxonomy

Config: `configs/taxonomy.yaml`  
Code: `src/tiktok_brand/etl/taxonomy_rules.py`  
Layer: **feature only** (not clean)

## Field order

```
brand_styles → product_lines → product_categories
```

YAML top-level keys follow the same order for readability.

## Multi-label rules (`brand_styles`, `product_lines`)

1. Scan **all** `normalized_hashtags`.
2. Collect every map hit; dedupe; sort with a fixed label order.
3. Empty list `[]` means unrecognized (not null).

Maps:

| YAML key | Output field | Example |
|----------|--------------|---------|
| `brand_style_map` | `brand_styles` | `nikerunning` → `performance` |
| `product_line_map` | `product_lines` | `niketech` → `tech_fleece` |

## Category cascade (`product_categories`)

First non-empty layer wins (layers are **not** merged):

| Step | Condition | Action |
|------|-----------|--------|
| (1) | `product_lines` non-empty | Map each line via `line_to_category_map` (e.g. `tech_fleece` → `apparel`, `samba` → `shoes`) |
| (2) | else | Scan hashtags via `product_category_map` (+ accessories keywords on tags) |
| (3) | else | Caption heuristics (`category_caption_keywords` + `accessories_keywords`) |
| (4) | else | `["uncategorized"]` |

When (1) fires, tag-level category hits are **ignored** so style tags (e.g. `adidasstyle` → apparel) cannot override line-derived categories.

### Caption heuristics (step 3)

| Category | Example keywords |
|----------|------------------|
| apparel | fit, wear, outfit, jacket, shirt, pants, look |
| shoes | shoe, sneaker, kicks, pair |
| accessories | bag, socks, hat, … (`accessories_keywords`) |

### Allowed category labels

`shoes`, `apparel`, `accessories`, `uncategorized`

## Design notes

- `nikeair` is a crawl seed but **not** mapped to a product line (broad family prefix).
- Plural seed variants (`adidassambas`, …) are normalized in **clean** via `hashtags.yaml` → `normalize_tags` before taxonomy runs.
- Taxonomy maps live in `taxonomy.yaml`, **not** in `hashtags.yaml`.
