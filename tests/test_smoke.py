"""M0 exit test: the wheel builds, installs and imports."""

import zstream


def test_import():
    assert zstream.__version__
