# backend/pvrt/config.py
from __future__ import annotations
import os, platform
from pathlib import Path

# Project root (…/projects/solaroly)
PROJECT_ROOT = Path(__file__).resolve().parents[2].parent

# Default guesses for libdirp location (adjust if your tree differs)
DEFAULTS = {
    "Linux":  PROJECT_ROOT / "third_party/utility/bin/linux/release_x64/libdirp.so",
    "Windows":PROJECT_ROOT / "third_party/utility/bin/win/release_x64/dirp.dll",
    "Darwin": PROJECT_ROOT / "third_party/utility/bin/macos/release_x64/libdirp.dylib",
}

# You can override via environment variable:
#   export PVRT_DIRP_LIB=/abs/path/to/libdirp.so
DIRP_LIB = Path(os.getenv("DIRP_SDK_PATH", str(DEFAULTS.get(platform.system(), ""))))

def describe_dirp():
    sys = platform.system()
    return f"system={sys}, DIRP_LIB={DIRP_LIB} (exists={DIRP_LIB.exists()})"
