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

`seed_style_map` keys are **crawl seed search terms** from `configs/hashtags.yaml`, not TikTok `@username` accounts.

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

## Known limitations (taxonomy V1)

Broad hashtags such as `#adidasoriginals` may occasionally produce overly broad product-line assignments, while products not covered by the current taxonomy may remain unidentified or inherit a less-specific matched label.

Two concrete issues, both left as-is in V1 so the feature table, notebooks, and existing analyses stay consistent:

1. **Broad sub-brand tag mapped to a product line.** `adidasoriginals` → `originals_apparel` (and, through the category cascade, `apparel`). Creators use `#adidasoriginals` as a general Adidas label, including on footwear: of the 114 videos tagged `originals_apparel`, 49 mention shoes in the caption or hashtags, and 8 mention apparel. Because layer (1) of the category cascade wins, those videos are categorized `apparel` and tag-level shoe evidence (`#adidasshoes`, `#basketballshoes`) is ignored.
2. **Uncovered products.** Products missing from `product_line_map` (e.g. AE 1: `#ae1`, `#adidasae1`) aren't identified. If the post also carries a broad tag, it inherits that less-specific label instead. Example: `7395623868435565830`, an AE 1 High vs Low basketball review, is labelled `originals_apparel` / `apparel`.

By contrast, `niketech` → `tech_fleece` is kept as reliable: "Nike Tech" is common shorthand for the Tech Fleece line.

**App display policy (presentation only, no data changes).** The Streamlit app doesn't show `originals_apparel` as a product line, and it treats a line owned by the other brand as a conflicting label. If no specific line remains, it shows "Brand · Product line unresolved" and omits the category, rather than asserting a label known to be unreliable or silently substituting a corrected one. See `BROAD_PRODUCT_LINES` / `_resolve_products` in `app/catalog.py`. Similarity scoring still uses the pipeline's `product_lines`.

**Planned (after V1 is complete):** Taxonomy V2 → rerun the feature pipeline → compare V1 vs V2 (label changes, category shifts, effect on notebook results) as a robustness check.
