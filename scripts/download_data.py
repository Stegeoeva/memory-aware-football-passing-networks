"""Download only the pinned source files needed by this project (stdlib only).

Existing files are retained when their SHA256 matches. All downloads are verified
against data/manifest.json before replacing the destination. No git is required.
"""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    base = f"https://raw.githubusercontent.com/hudl/open-data/{manifest['commit']}/"
    for relative, expected in manifest["sha256"].items():
        target = ROOT / "data/open-data" / relative
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print(f"OK {relative}")
            continue
        url = base + urllib.parse.quote(relative)
        with urllib.request.urlopen(url,timeout=90) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(content)
        print(f"Downloaded {relative}")
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy2(ROOT/"data/open-data/img/SB - Icon Lockup - Colour positive.png",assets/"statsbomb-logo.png")
    print("Pinned data ready.")

if __name__ == "__main__":
    main()
