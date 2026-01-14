from tiktok_brand.etl.normalize_tags import normalize_hashtags

def test_normalize_hashtags():
    mapping = {"adidassambas": "adidassamba"}
    assert normalize_hashtags(["AdidasSambas", "nike"], mapping) == ["adidassamba", "nike"]
