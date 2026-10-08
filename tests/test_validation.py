import pytest

from app.utils.validation import is_url


@pytest.mark.parametrize("text,expected", [
    ("https://www.youtube.com/watch?v=abc", True),
    ("http://youtu.be/abc", True),
    ("  https://example.com  ", True),
    ("C:\\Videos\\ep1.mkv", False),
    ("/home/u/ep1.mkv", False),
    ("ftp://host/file", False),
    ("https://", False),
    ("", False),
])
def test_is_url(text, expected):
    assert is_url(text) is expected
