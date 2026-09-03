"""Resolve the user-installed DJI DIRP SDK library.

DJI's proprietary runtime is intentionally not distributed with SolarOly.
Set ``DIRP_SDK_PATH=/opt/dji-tsdk/utility/bin/linux/release_x64/libdirp.so``
when DJI radiometric decoding is enabled.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

# No repository-relative fallback is used: users must obtain DJI's SDK under
# DJI's terms and explicitly select its native library.
DIRP_LIB: Path | None = (
    Path(os.environ["DIRP_SDK_PATH"]).expanduser()
    if os.getenv("DIRP_SDK_PATH")
    else None
)


def describe_dirp():
    sys = platform.system()
    exists = bool(DIRP_LIB and DIRP_LIB.exists())
    return f"system={sys}, DIRP_LIB={DIRP_LIB or 'not configured'} (exists={exists})"
