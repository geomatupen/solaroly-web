from backend.pvrt.web.app import _canonical_image_candidate_names


def test_canonical_image_candidates_prefer_original_manifest_name():
    candidates = _canonical_image_candidate_names(
        {"DJI_0001_T.png", "DJI_0002_T.png"},
        {"DJI_0001_T.jpeg"},
        {"DJI_0001_T.jpeg", "DJI_0002_T.jpeg"},
        {"DJI_0001_T.jpeg", "DJI_0002_T.jpeg"},
    )

    assert candidates == {"DJI_0001_T.jpeg", "DJI_0002_T.jpeg"}


def test_canonical_image_candidates_keep_generated_name_without_source_record():
    candidates = _canonical_image_candidate_names(
        {"prepared_only.png"},
        set(),
        set(),
        set(),
    )

    assert candidates == {"prepared_only.png"}
