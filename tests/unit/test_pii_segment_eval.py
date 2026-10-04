from scripts.pii_segment_eval import segments


def test_protected_span_extension_reaches_fixed_point_independent_of_order() -> None:
    text = "abcdefghijklmnopqrst"
    protected = [(11, 16), (8, 12)]

    assert list(segments(text, 10, protected)) == [
        (0, "abcdefghijklmnop"),
        (16, "qrst"),
    ]
