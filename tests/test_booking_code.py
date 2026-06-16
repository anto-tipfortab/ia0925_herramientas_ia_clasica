"""Tests for FS###### booking-code normalization (app.voice_pipeline)."""
import pytest

from app.voice_pipeline import normalize_booking_code


@pytest.mark.parametrize(
    "raw, expected_contains",
    [
        ("fs 123456", "FS123456"),          # lowercase + space
        ("FS123456", "FS123456"),           # already canonical
        ("Fs123456", "FS123456"),           # mixed case
        ("f s 123456", "FS123456"),         # split 'f s'
        ("fs 1 2 3 4 5 6", "FS123456"),     # spaced digits
        ("fs 1,2,3,4,5,6", "FS123456"),     # comma-separated digits
        ("FS-123456", "FS123456"),          # dash separator
        ("mi codigo es fs123456 gracias", "FS123456"),  # embedded in a sentence
    ],
)
def test_normalize_produces_canonical_code(raw, expected_contains):
    assert expected_contains in normalize_booking_code(raw)


def test_canonical_code_is_idempotent():
    assert normalize_booking_code("FS654321") == "FS654321"


def test_no_code_left_untouched():
    text = "buenos dias, quiero la piscina"
    assert normalize_booking_code(text) == text


def test_too_few_digits_not_normalized():
    # Only 5 digits → not a valid FS###### code → must not be coerced to FS#####.
    assert "FS12345" not in normalize_booking_code("fs 12345")


def test_only_first_six_digits_used():
    # Seven spoken digits: normalization consumes exactly six.
    out = normalize_booking_code("fs 1234567")
    assert "FS123456" in out
