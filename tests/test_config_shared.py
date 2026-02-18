from config_shared import infer_mode_from_sport_key, infer_level_from_sport_key


def test_infer_mode_from_sport_key():
    assert infer_mode_from_sport_key('tennis_atp_qatar_open') == 'ATP'
    assert infer_mode_from_sport_key('tennis_wta_dubai') == 'WTA'
    assert infer_mode_from_sport_key('tennis_unknown') is None


def test_infer_level_from_sport_key_aliases():
    lvl, fallback = infer_level_from_sport_key('tennis_atp_qatar_open', default=1.0)
    assert lvl == 1.7
    assert fallback is False

    lvl2, fallback2 = infer_level_from_sport_key('tennis_unknown', default=1.0)
    assert lvl2 == 1.0
    assert fallback2 is True
