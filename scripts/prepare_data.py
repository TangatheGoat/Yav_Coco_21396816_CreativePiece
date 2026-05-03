from __future__ import annotations
import argparse
import random
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42          # reproducibility — must never change between runs
SAMPLE_SIZE   = 5_000       # images drawn from each class
SPLIT_SIZES   = {           # must sum to SAMPLE_SIZE
    "train": 3_500,
    "val":     750,
    "test":    750,
}
CLASSES       = ("real", "fake")
IMAGE_SUFFIX  = ".jpg"
# Deterministic per-class seed offsets. Must be updated if CLASSES changes.
# Do NOT use hash(class_name) — Python randomises string hashes at startup
# (PYTHONHASHSEED), so hash() produces different values on every interpreter
# invocation, silently breaking reproducibility.
PER_CLASS_OFFSETS: dict[str, int] = {"real": 0, "fake": 1}

DEFAULT_SOURCE = Path.home() / "Downloads" / "deepfake-faces"
DEFAULT_DEST   = Path("data") / "working"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a balanced deepfake-face dataset for PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Root directory that contains 'real/' and 'fake/' sub-folders.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination root (will be created if absent).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate the destination directory if it already exists.",
    )
    return parser.parse_args()


def validate_source(source: Path) -> None:
    """Exit loudly if the expected source sub-folders are missing."""
    if not source.is_dir():
        sys.exit(
            f"[ERROR] Source directory not found:\n"
            f"        {source}\n"
            f"        Check the --source path and try again."
        )
    for cls in CLASSES:
        cls_dir = source / cls
        if not cls_dir.is_dir():
            sys.exit(
                f"[ERROR] Expected sub-folder '{cls}' not found inside:\n"
                f"        {source}\n"
                f"        Folder must contain 'real/' and 'fake/' sub-directories."
            )


def guard_destination(dest: Path, force: bool) -> None:
    """Prevent silent overwrites unless --force is passed."""
    if dest.exists():
        if force:
            print(f"[INFO]  --force set. Removing existing destination: {dest}")
            shutil.rmtree(dest)
        else:
            sys.exit(
                f"[ERROR] Destination already exists:\n"
                f"        {dest}\n"
                f"        Re-run with --force to overwrite, or choose a different --dest."
            )


def list_images(directory: Path) -> list[Path]:
    """Return a sorted list of .jpg files in *directory*.

    Exits with a clear message if the folder is empty or contains fewer
    images than SAMPLE_SIZE — both conditions make random.sample() fail.
    """
    files = sorted(directory.glob(f"*{IMAGE_SUFFIX}"))
    if not files:
        sys.exit(
            f"[ERROR] No '{IMAGE_SUFFIX}' files found in:\n"
            f"        {directory}"
        )
    if len(files) < SAMPLE_SIZE:
        sys.exit(
            f"[ERROR] Found only {len(files):,} images in:\n"
            f"        {directory}\n"
            f"        Need at least {SAMPLE_SIZE:,} to sample from. "
            f"Lower SAMPLE_SIZE or point --source at the full dataset."
        )
    return files


def sample_and_split(
    images: list[Path],
    class_name: str,
) -> dict[str, list[Path]]:
    """Sample SAMPLE_SIZE files from *images* and partition into train/val/test.

    Uses a deterministic per-class seed derived from RANDOM_SEED and
    PER_CLASS_OFFSETS, so the split each class receives is independent of
    the order in which classes are processed and reproducible across Python
    invocations (no reliance on hash(), which is randomised at startup).
    """
    # Derive a stable, class-specific seed from a hard-coded offset.
    # hash(class_name) is intentionally avoided: Python randomises string
    # hashes at startup (PYTHONHASHSEED), so it returns a different value
    # on every interpreter invocation and would silently break reproducibility.
    per_class_seed = RANDOM_SEED + PER_CLASS_OFFSETS[class_name]
    rng = random.Random(per_class_seed)
    sampled = rng.sample(images, SAMPLE_SIZE)

    splits: dict[str, list[Path]] = {}
    cursor = 0
    for split_name, split_size in SPLIT_SIZES.items():
        splits[split_name] = sampled[cursor : cursor + split_size]
        cursor += split_size

    return splits


def copy_split(
    splits: dict[str, list[Path]],
    class_name: str,
    dest_root: Path,
) -> dict[str, int]:
    """Copy files to dest_root/<split>/<class_name>/ and return per-split counts."""
    counts: dict[str, int] = {}
    for split_name, files in splits.items():
        target_dir = dest_root / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for src_path in files:
            shutil.copy2(src_path, target_dir / src_path.name)

        counts[split_name] = len(files)
        print(
            f"    [{class_name:>4}] {split_name:<5} → "
            f"{target_dir}  ({len(files):,} files)"
        )
    return counts


def write_manifest(
    class_splits: dict[str, dict[str, list[Path]]],
    dest_root: Path,
) -> Path:
    """Write a CSV listing every sampled image, its class, and its assigned split.

    The manifest at data/working/manifest.csv provides cast-iron reproducibility
    evidence: re-running the script and diffing the manifest confirms that splits
    are deterministic. Columns: filename, class, split.
    """
    import csv

    manifest_path = dest_root / "manifest.csv"
    rows: list[tuple[str, str, str]] = []

    for class_name, splits in class_splits.items():
        for split_name, files in splits.items():
            for f in files:
                rows.append((f.name, class_name, split_name))

    # Sort for a stable, diff-friendly output order.
    rows.sort(key=lambda r: (r[2], r[1], r[0]))  # split → class → filename

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "class", "split"])
        writer.writerows(rows)

    print(f"  Manifest written → {manifest_path}  ({len(rows):,} rows)")
    return manifest_path


def print_summary(
    all_counts: dict[str, dict[str, int]],
    elapsed: float,
) -> None:
    """Print a formatted table of per-split, per-class image counts and total wall time."""
    print("\n" + "=" * 60)
    print("Dataset summary")
    print("=" * 60)
    header = f"{'Split':<8}" + "".join(f"{cls:>10}" for cls in CLASSES) + f"{'Total':>10}"
    print(header)
    print("-" * 60)
    grand_total = 0
    for split_name in SPLIT_SIZES:
        row = f"{split_name:<8}"
        split_total = 0
        for cls in CLASSES:
            n = all_counts[cls][split_name]
            row += f"{n:>10,}"
            split_total += n
        row += f"{split_total:>10,}"
        grand_total += split_total
        print(row)
    print("-" * 60)
    print(f"{'TOTAL':<8}" + " " * 20 + f"{grand_total:>10,}")
    print("=" * 60)
    print(f"Total script time: {elapsed:.1f} s\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full pipeline: validate → list → sample → copy → manifest → summarise."""
    t_start = time.perf_counter()   # wall-clock start for the entire script
    args = parse_args()

    # ── 1. Guard against bad paths and existing output ──────────────────────
    validate_source(args.source)
    guard_destination(args.dest, args.force)

    # ── 2. List all images per class ─────────────────────────────────────────
    print("\n[STEP 1/5] Scanning source directories …")
    class_images: dict[str, list[Path]] = {}
    for cls in CLASSES:
        cls_images = list_images(args.source / cls)
        class_images[cls] = cls_images
        print(f"  Found {len(cls_images):>7,} images in '{cls}/'")

    # ── 3. Sample & split each class independently (reproducible) ────────────
    print(
        f"\n[STEP 2/5] Sampling {SAMPLE_SIZE:,} images per class with "
        f"base seed {RANDOM_SEED} …"
    )
    class_splits: dict[str, dict[str, list[Path]]] = {}
    for cls in CLASSES:
        class_splits[cls] = sample_and_split(class_images[cls], cls)

    split_desc = " / ".join(
        f"{name} {size:,}" for name, size in SPLIT_SIZES.items()
    )
    print(f"  Split sizes per class: {split_desc}")

    # ── 4. Copy files ────────────────────────────────────────────────────────
    print(f"\n[STEP 3/5] Copying files to: {args.dest}")
    all_counts: dict[str, dict[str, int]] = {}
    for cls in CLASSES:
        all_counts[cls] = copy_split(class_splits[cls], cls, args.dest)

    # ── 5. Write manifest ────────────────────────────────────────────────────
    print(f"\n[STEP 4/5] Writing manifest …")
    write_manifest(class_splits, args.dest)

    # ── 6. Summary ───────────────────────────────────────────────────────────
    print("\n[STEP 5/5] Verifying output …")
    elapsed = time.perf_counter() - t_start   # total script wall time
    print_summary(all_counts, elapsed)


if __name__ == "__main__":
    main()