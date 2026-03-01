#!/usr/bin/env python3
"""Direct test of rotation + mosaic workflow"""
from pathlib import Path
import subprocess
import sys
import os
import time
import json

PROJECT_ROOT = Path(__file__).parent

# Load active project
projects_file = PROJECT_ROOT / "backend" / "projects" / "projects.json"
if not projects_file.exists():
    print("ERROR: projects.json not found. Run the app first to create a project.")
    sys.exit(1)

projects_data = json.loads(projects_file.read_text())
active_project_id = projects_data.get("active_project")
if not active_project_id:
    print("ERROR: No active project set in projects.json")
    sys.exit(1)

session_id = "test_20260126_132902"  # Update this to an existing session
project_root = PROJECT_ROOT / "backend" / "projects" / active_project_id
session_dir = project_root / "test" / "outputs" / session_id

print(f"Testing mosaic workflow for session: {session_id}")
print(f"Session dir: {session_dir}")
print(f"Exists: {session_dir.exists()}")

# Check camera_meta
cm_path = session_dir / "camera_meta.json"
print(f"\ncamera_meta.json exists: {cm_path.exists()}")

if not cm_path.exists():
    print("ERROR: camera_meta.json not found!")
    sys.exit(1)

# Clean up old rotated_images if it exists
rotated_images_dir = session_dir / "rotated_images"
if rotated_images_dir.exists():
    import shutil
    print(f"Removing old rotated_images dir...")
    shutil.rmtree(rotated_images_dir)

print(f"\nBefore running script:")
print(f"  rotated_images exists: {rotated_images_dir.exists()}")

# Run the regenerate script
script = PROJECT_ROOT / "backend" / "pvrt" / "dataops" / "regenerate_geojson_from_preds.py"
print(f"\nRunning script: {script}")
print(f"Working directory: {PROJECT_ROOT}")

env = os.environ.copy()
proc = subprocess.run(
    [sys.executable, str(script), session_id],
    stdout=subprocess.DEVNULL,  # Don't capture stdout to avoid pipe deadlock
    stderr=subprocess.PIPE,
    text=True,
    cwd=str(PROJECT_ROOT),
    timeout=300,
    env=env
)

print(f"Exit code: {proc.returncode}")
if proc.stdout:
    print(f"\nStdout:")
    for line in proc.stdout.splitlines():
        if "[rotation]" in line:
            print(f"  {line}")

if proc.stderr:
    print(f"\nStderr (first 500 chars):\n{proc.stderr[:500]}")

# Sync and wait
print(f"\nSyncing filesystem...")
os.sync()
time.sleep(0.5)

# Check again
print(f"\nAfter running script:")
print(f"  rotated_images exists: {rotated_images_dir.exists()}")

if rotated_images_dir.exists():
    files = list(rotated_images_dir.glob("*"))
    print(f"  file count: {len(files)}")
    for f in sorted(files)[:3]:
        print(f"    - {f.name}")
    if len(files) > 3:
        print(f"    ... and {len(files)-3} more")
else:
    print("  ERROR: rotated_images directory was not created!")
