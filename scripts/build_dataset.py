from __future__ import annotations
from pathlib import Path
import glob
import yaml

from tiktok_brand.etl.build_clean_table import build_clean_table

def main() -> None:
    project_cfg = yaml.safe_load(Path("configs/project.yaml").read_text(encoding="utf-8"))
    raw_dir = Path(project_cfg["output"]["raw_dir"])
    processed_dir = Path(project_cfg["output"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(glob.glob(str(raw_dir / "*.jsonl")))
    df = build_clean_table(
        raw_paths=raw_paths,
        hashtags_cfg_path="configs/hashtags.yaml",
        accounts_cfg_path="configs/accounts.yaml",
        project_cfg_path="configs/project.yaml",
    )
    out = processed_dir / "tiktok_videos.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")

if __name__ == "__main__":
    main()
