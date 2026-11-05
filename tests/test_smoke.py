"""M0 exit test: the wheel builds, installs and imports."""

import zarrstream


def test_import():
    assert zarrstream.__version__
