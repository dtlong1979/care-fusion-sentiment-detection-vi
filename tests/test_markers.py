"""Unit tests for marker extraction (A2) and text normalization (A1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from care_fusion.markers import extract_markers           # noqa: E402
from care_fusion.preprocess import normalize_text          # noqa: E402


def seq(text, **kw):
    return extract_markers(text, **kw).marker_seq


def test_longest_match_double_paren_not_split():
    # ":))" must NOT be sliced into ":)" + ")"
    r = extract_markers("per là ai thế nhỉ cụ :))")
    assert r.marker_seq == [":))"]
    assert r.text_clean == "per là ai thế nhỉ cụ"


def test_repeat_count_and_intensity():
    r = extract_markers("ấn tượng vãi :v bàn tay vàng :v")
    assert r.marker_seq == [":v", ":v"]
    assert r.marker_counts == {":v": 2}
    assert r.n_marker == 2


def test_elongation_collapses_to_base():
    assert seq("vui quá :))))") == [":))"]
    assert seq("vui quá :))))", collapse_elongation=False) == [":))))"]


def test_sad_double_paren():
    assert seq("tao lại bị mọc lông chân rồi :((") == [":(("]


def test_single_smileys():
    assert seq("ok :) đi") == [":)"]
    assert seq("buồn :( thật") == [":("]


def test_heart_and_faces():
    assert seq("yêu <3 ^^") == ["<3", "^^"]
    assert seq("khóc T_T") == ["T_T"]


def test_emoji_unicode_extracted():
    r = extract_markers("bức ảnh xuất sắc ❤️")
    assert r.marker_seq and "❤" in r.marker_seq[0]
    assert "❤" not in r.text_clean


def test_emoji_and_emoticon_order_preserved():
    r = extract_markers("hay 😊 thật :))")
    assert r.marker_seq == ["😊", ":))"]


def test_no_marker():
    r = extract_markers("một câu bình thường")
    assert r.marker_seq == [] and r.n_marker == 0
    assert r.text_clean == "một câu bình thường"


def test_normalize_glues_detached_diacritic():
    # combining dot-below detached by a space should be re-attached
    out = normalize_text("thậ")          # already-combined baseline
    spaced = normalize_text("thâ ̣")      # detached by space
    assert spaced == out


def test_normalize_collapses_whitespace():
    assert normalize_text("  nhiều    khoảng   trắng ") == "nhiều khoảng trắng"
