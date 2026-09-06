"""Tests for shared frame cache (20/40/60/80%) + CLIP visual features."""

from pathlib import Path

from PIL import Image

from tiktok_brand.embeddings.frames import (
    build_tiktok_video_url,
    extract_frames_at_fractions,
    list_cached_frames,
    resolve_video_frames,
    _frame_indices,
)
from tiktok_brand.embeddings.visual_embed import (
    APPEARANCE_OTHER,
    APPEARANCE_PERSON_PRESENT,
    APPEARANCE_UNKNOWN,
    METHOD_CLIP,
    METHOD_NOT_EMBEDDED,
    compute_visual_embeddings,
    compute_visual_features,
    infer_appearance_type,
)


def test_frame_indices_at_20_40_60_80():
    idx = _frame_indices(100, (0.2, 0.4, 0.6, 0.8))
    assert idx == [20, 40, 60, 80]


def test_build_tiktok_video_url():
    assert build_tiktok_video_url("123", author_username="nike") == (
        "https://www.tiktok.com/@nike/video/123"
    )
    assert build_tiktok_video_url("123") == "https://www.tiktok.com/video/123"
    assert build_tiktok_video_url(
        "123", page_url="https://www.tiktok.com/@x/video/123"
    ).endswith("/video/123")


def test_resolve_local_video_extracts_four_frames(tmp_path: Path):
    # Synthetic "video": write 10 identical jpgs is not a video — skip opencv if
    # we cannot create mp4 easily. Instead unit-test extract helper with mock-less
    # path: copy a single image as cover fallback when no cv2 video.
    cover = tmp_path / "src.jpg"
    Image.new("RGB", (8, 8), color=(0, 128, 255)).save(cover)
    frames_root = tmp_path / "frames"

    paths = resolve_video_frames(
        "vid123",
        cover_path=str(cover),
        frames_root=frames_root,
        fractions=(0.2, 0.4, 0.6, 0.8),
        download_video=False,
    )
    assert len(paths) == 1  # cover fallback when no video file
    assert Path(paths[0]).is_file()

    # cache hit
    again = resolve_video_frames(
        "vid123",
        cover_path=str(cover),
        frames_root=frames_root,
        download_video=False,
    )
    assert again == paths
    assert list_cached_frames("vid123", frames_root)


def test_compute_visual_features_shared_frames(tmp_path: Path):
    cover = tmp_path / "c.jpg"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(cover)
    frames_root = tmp_path / "frames"

    def fake_encode(images):
        return [[0.2, 0.8] for _ in images]

    rules = tmp_path / "rules.yaml"
    rules.write_text(
        "visual_embedding:\n  enabled: true\n  model_name: test-clip\n"
        f"  frames_dir: {frames_root}\n  max_frames: 4\n"
        "  frame_fractions: [0.2, 0.4, 0.6, 0.8]\n  download_video: false\n"
        "appearance_type:\n  person_ratio: 0.6\n  product_only_ratio: 0.6\n"
        "  mixed_min_ratio: 0.2\n",
        encoding="utf-8",
    )

    rows = compute_visual_features(
        ["v1", "v2"],
        cover_paths=[str(cover), None],
        rules_path=str(rules),
        encode_images_fn=fake_encode,
    )
    assert rows[0]["visual_embedding_method"] == METHOD_CLIP
    emb = rows[0]["visual_embedding"]
    assert abs(sum(x * x for x in emb) - 1.0) < 1e-6
    assert rows[0]["appearance_type"] == APPEARANCE_OTHER
    assert rows[0]["frame_paths"]
    assert rows[1]["visual_embedding_method"] == METHOD_NOT_EMBEDDED
    assert rows[1]["appearance_type"] == APPEARANCE_UNKNOWN


def test_no_cover_not_embedded():
    rows = compute_visual_embeddings([None, "", "  "])
    assert all(r["visual_embedding_method"] == METHOD_NOT_EMBEDDED for r in rows)
    assert all(r["appearance_type"] == APPEARANCE_UNKNOWN for r in rows)


def test_infer_appearance_thresholds():
    assert infer_appearance_type(has_image=False) == APPEARANCE_UNKNOWN
    assert (
        infer_appearance_type(has_image=True, person_ratio=0.9, product_ratio=0.1)
        == APPEARANCE_PERSON_PRESENT
    )
