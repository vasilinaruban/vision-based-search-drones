from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

RESULTS_DIR = Path("results_json")
WORK_DIR = Path("results_for_zip")
ZIP_NAME = Path("results.zip")
CHECKSUM_FILE = Path("checksum.txt")


def find_best_json() -> Optional[Path]:
    """Берём последний по времени object_data_*.json."""
    if not RESULTS_DIR.exists():
        print(f"[ERROR] {RESULTS_DIR} does not exist")
        return None

    candidates = sorted(RESULTS_DIR.glob("object_data_*.json"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    if not candidates:
        print("[ERROR] No object_data_*.json found in results_json/")
        return None
    return candidates[0]


def prepare_workdir(json_path: Path) -> Path:
    """Копируем картинки и json в отдельную папку для архива."""
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Копируем json как results.json
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out_json = WORK_DIR / "results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Копируем изображения из results_json/ в корень архива
    for img in RESULTS_DIR.glob("*.jpg"):
        shutil.copy2(img, WORK_DIR / img.name)

    return WORK_DIR


def make_zip(folder: Path, zip_name: Path) -> Path:
    if zip_name.exists():
        zip_name.unlink()
    shutil.make_archive(zip_name.stem, "zip", root_dir=folder)
    return zip_name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum(zip_path: Path, checksum_file: Path) -> None:
    digest = sha256(zip_path)
    with checksum_file.open("w", encoding="utf-8") as f:
        f.write(digest + "\n")
    print(f"[OK] SHA256: {digest}")


def main() -> int:
    json_path = find_best_json()
    if json_path is None:
        return 1

    print(f"[INFO] Using JSON: {json_path}")
    folder = prepare_workdir(json_path)
    zip_path = make_zip(folder, ZIP_NAME)
    print(f"[OK] Created archive: {zip_path}")

    write_checksum(zip_path, CHECKSUM_FILE)
    print("[OK] checksum.txt written")

    print(
        "\nТеперь положите рядом GPX-файл трека полета — "
        "корневая папка будет содержать checksum.txt, results.zip и *.gpx, "
        "как требует техрегламент."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
