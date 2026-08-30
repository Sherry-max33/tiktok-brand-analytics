"""
Clean table: 将爬虫原始 VideoRecord 统一清洗成『语义事实表』。

字段规则说明（未特别注明的列，都是从 raw 直接拷贝，不做变换，例如 platform/source_type/source_query/author_id 等）：

- caption_raw
  - 若 raw 中已有 caption_raw：原样保留。
  - 若只有 caption：复制一列为 caption_raw。

- hashtags
  - 若 raw 已有 list 类型 hashtags：原样保留。
  - 否则：从 caption_raw 中用 extract_hashtags 抽取，类型保证为 list[str]。

- normalized_hashtags
  - 对 hashtags 中每个标签，用 normalize_tags.py + configs/hashtags.yaml 的 normalize_tags 映射做归一化。

- brand
  - 若 raw.brand 非空：直接保留。
  - 否则：对 normalized_hashtags 中的标签：
    - 只出现 nike* → \"nike\"。
    - 只出现 adidas* → \"adidas\"。
    - 同时出现 nike* 和 adidas* → \"both\"。
    - 都没有 → None。

- seed_hashtag
  - 当 source_type == \"hashtag\" 时：取 source_query 作为 seed_hashtag。
  - 其他 source_type：为 None。

- brand_style
  - 对 normalized_hashtags 逐个标签，按 configs/hashtags.yaml.brand_style_map 做标签→风格映射。
  - 遇到的第一个命中的标签返回其风格（performance/lifestyle/technical/retro 等）；无命中时为 None。

- product_line
  - 对 normalized_hashtags 逐个标签，按 configs/hashtags.yaml.product_line_map 做标签→产品线映射，
    例如 niketech→tech_fleece、adidassamba→samba 等。
  - 遇到的第一个命中的标签返回其产品线；无命中时为 None。

- product_category
  - 使用三层优先级 derive_product_category：
    1) 若已有 product_line，先查 configs/hashtags.yaml.line_to_category_map（如 tech_fleece→apparel）。
    2) 否则，对 normalized_hashtags 查 configs/hashtags.yaml.product_category_map（如 adidasshoes→shoes）。
    3) 若仍为空，根据 caption_raw 文本关键词启发式：
       - 命中 [\"fit\",\"wear\",\"outfit\",\"jacket\",\"shirt\",\"pants\",\"look\"] → \"apparel\"。
       - 命中 [\"shoe\",\"sneaker\",\"kicks\",\"pair\"] → \"shoes\"。
       - 否则 → \"uncategorized\"。

- is_official_brand
  - author_username 小写后是否在 configs/accounts.yaml.official_accounts 中配置为该品牌官方账号。

- create_time
  - 若存在 create_time_ts：使用项目时区 tz，将 Unix 秒转为 ISO 字符串。
  - 否则为 NA。

- engagement_count / engagement_rate
  - 先把 view/like/comment/share/collect_count 转为 numeric。
  - engagement_count = like_count + comment_count + share_count + collect_count（缺失视为 0）。
  - engagement_rate = engagement_count / view_count；当 view_count<=0 时设为 NA。

- 去重逻辑
  - 按 crawled_at_ts 升序排序，同一 video_id 只保留最后一条记录（认为是最新快照）。
  - raw_payload 列在 clean 表中直接删除。
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml

from ..common.text import extract_hashtags
from ..common.time import ts_to_iso
from .normalize_tags import normalize_hashtags
from .feature_table import add_derived_metrics

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_clean_table(
    raw_paths: List[str | Path],
    hashtags_cfg_path: str | Path,
    accounts_cfg_path: str | Path,
    project_cfg_path: str | Path,
) -> pd.DataFrame:
    hashtags_cfg = load_yaml(hashtags_cfg_path)
    accounts_cfg = load_yaml(accounts_cfg_path)
    project_cfg = load_yaml(project_cfg_path)
    tz = project_cfg["time"]["timezone"]

    normalize_map = {k.lower(): v.lower() for k, v in (hashtags_cfg.get("normalize_tags") or {}).items()}
    product_map = {k.lower(): v for k, v in (hashtags_cfg.get("product_line_map") or {}).items()}
    style_map = {k.lower(): v for k, v in (hashtags_cfg.get("brand_style_map") or {}).items()}
    category_map = {k.lower(): v for k, v in (hashtags_cfg.get("product_category_map") or {}).items()}
    line_to_category = hashtags_cfg.get("line_to_category_map") or {}

    official_usernames = set()
    for brand, users in (accounts_cfg.get("official_accounts") or {}).items():
        for u in users:
            official_usernames.add(u.lower())

    rows: List[Dict[str, Any]] = []
    for p in raw_paths:
        name = Path(p).name.lower()
        if "comments" in name or "comment_" in name:
            continue
        rows.extend(read_jsonl(p))

    df = pd.DataFrame(rows)

    # Legacy raw harmonization (old JSONL used save_count)
    if "collect_count" not in df.columns and "save_count" in df.columns:
        df["collect_count"] = df["save_count"]

    # Basic harmonization
    if "caption_raw" not in df.columns and "caption" in df.columns:
        df["caption_raw"] = df["caption"]

    if "caption_raw" in df.columns:
        df["caption_raw"] = df["caption_raw"].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    # Extract hashtags if missing
    if "hashtags" not in df.columns:
        df["hashtags"] = df["caption_raw"].fillna("").map(extract_hashtags)

    def _clean_hashtag_list(tags: Any) -> List[str]:
        if not isinstance(tags, list):
            return []
        seen: set[str] = set()
        out: List[str] = []
        for item in tags:
            t = str(item).strip().lstrip("#").lower()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    df["hashtags"] = df["hashtags"].apply(_clean_hashtag_list)
    df["normalized_hashtags"] = df["hashtags"].apply(lambda tags: normalize_hashtags(tags, normalize_map))

    # Brand label (simple: if any nike* tag then nike; if any adidas* tag then adidas; else null)
    def infer_brand(tags):
        tags = [t.lower() for t in (tags or [])]
        has_nike = any(t.startswith("nike") for t in tags)
        has_adidas = any(t.startswith("adidas") for t in tags)
        if has_nike and not has_adidas:
            return "nike"
        if has_adidas and not has_nike:
            return "adidas"
        if has_nike and has_adidas:
            return "both"
        return None

    # Use raw brand when available, else infer from hashtags
    inferred = df["normalized_hashtags"].apply(infer_brand)
    if "brand" in df.columns:
        df["brand"] = df["brand"].fillna(inferred)
    else:
        df["brand"] = inferred

    # seed_hashtag: when source_type=hashtag, source_query is the hashtag
    df["seed_hashtag"] = df.apply(
        lambda r: r.get("source_query") if r.get("source_type") == "hashtag" else None,
        axis=1,
    )

    # brand_style: configs/hashtags.yaml brand_style_map 做标签→风格映射
    def infer_style(tags):
        for t in (tags or []):
            if t.lower() in style_map:
                return style_map[t.lower()]
        return None
    df["brand_style"] = df["normalized_hashtags"].apply(infer_style)

    # product_line
    def infer_product_line(tags):
        for t in (tags or []):
            t = t.lower()
            if t in product_map:
                return product_map[t]
        return None
    df["product_line"] = df["normalized_hashtags"].apply(infer_product_line)

    # product_category: 优先级
    # 1) 已识别的 product_line → line_to_category_map
    # 2) normalized_hashtags → product_category_map
    # 3) caption 文本关键词启发式推断
    def derive_product_category(row):
        line = row.get("product_line")
        hashtags = row.get("normalized_hashtags") or []
        caption = str(row.get("caption_raw") or "").lower()

        # 1. 从 Product Line 推断
        if line in line_to_category:
            return line_to_category[line]

        # 2. 从 Hashtags 映射
        for tag in hashtags:
            t = str(tag).lower()
            if t in category_map:
                return category_map[t]

        # 3. 关键词启发式推断 (Heuristic)
        apparel_keywords = ["fit", "wear", "outfit", "jacket", "shirt", "pants", "look"]
        if any(word in caption for word in apparel_keywords):
            return "apparel"

        shoes_keywords = ["shoe", "sneaker", "kicks", "pair"]
        if any(word in caption for word in shoes_keywords):
            return "shoes"

        accessories_keywords = [
            str(k).lower()
            for k in (hashtags_cfg.get("accessories_keywords") or [])
        ]
        if any(word in caption for word in accessories_keywords):
            return "accessories"
        for tag in hashtags:
            if str(tag).lower() in accessories_keywords:
                return "accessories"

        return "uncategorized"

    df["product_category"] = df.apply(derive_product_category, axis=1)

    # official flag
    df["author_username"] = df.get("author_username")
    df["is_official_brand"] = df["author_username"].fillna("").str.lower().isin(official_usernames)

    # create_time ISO
    if "create_time_ts" in df.columns:
        df["create_time"] = df["create_time_ts"].apply(lambda x: ts_to_iso(int(x), tz) if pd.notna(x) else pd.NA)

    # Derived metrics
    df = add_derived_metrics(df)

    # De-dup by video_id, keep last (assumes later crawl has newer stats)
    if "video_id" in df.columns:
        df = df.sort_values(by=["crawled_at_ts"], ascending=True)
        df = df.drop_duplicates(subset=["video_id"], keep="last")

    # Drop raw_payload (not needed in clean table; Parquet can't serialize empty struct)
    if "raw_payload" in df.columns:
        df = df.drop(columns=["raw_payload"])

    return df
