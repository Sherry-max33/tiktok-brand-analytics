# Frames cache (local only)

Local video / frame cache for CLIP and visual features. **Not committed** — only this README is tracked.

## Layout

```text
data/processed/frames/{video_id}/
  video.mp4          # optional full download
  frame_000.jpg      # ~20% of duration
  frame_001.jpg      # ~40%
  frame_002.jpg      # ~60%
  frame_003.jpg      # ~80%
```

Configured via `configs/feature_rules.yaml` → `visual_embedding.frames_dir`.

## Populate

```bash
# Prefer Apify downloads already on disk, then CLIP:
PYTHONPATH=src python -m scripts.enrich_visual --from-cache
```

See also `docs/05-feature-engineering.md` (visual embeddings section).
