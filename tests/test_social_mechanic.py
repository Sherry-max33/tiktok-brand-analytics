from tiktok_brand.etl.social_mechanic_rules import infer_social_mechanics


def test_social_mechanic_strong_and_multilabel():
    assert infer_social_mechanics("join the challenge and use this sound") == [
        "challenge",
        "audio_driven",
    ]
    assert "bts" in infer_social_mechanics("behind the scenes of the shoot")
    assert "grwm_ootd_format" in infer_social_mechanics("get ready with me fit check")


def test_social_mechanic_weak_double_or_hashtag():
    # bare weak alone — too broad
    assert infer_social_mechanics("this is viral") == []
    assert infer_social_mechanics("just a template") == []
    # hashtag form of weak token (social_mechanic-only gate)
    assert "trend" in infer_social_mechanics("check this #viral clip")
    assert "challenge" in infer_social_mechanics("#challenge day one")
    # two weak hits
    assert "trend" in infer_social_mechanics("trending viral edit")
    assert "template_remix" in infer_social_mechanics("capcut template remix")


def test_social_mechanic_parallel_to_content_intent():
    assert infer_social_mechanics("bts on set today #bts") == ["bts"]
    assert "pov" in infer_social_mechanics("pov: you just copped these")
    assert "challenge" in infer_social_mechanics("try the challenge today")
    assert "audio_driven" in infer_social_mechanics("sound on for this asmr")
    assert "bts" in infer_social_mechanics("peek inside the studio")
