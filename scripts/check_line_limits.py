"""Fail when a project-owned source file exceeds the 500-line limit."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "scripts", "tests")
CODE_EXTENSIONS = {".py", ".ps1"}
MAX_LINES = 500


def source_files():
    for folder in CODE_DIRS:
        code_root = ROOT / folder
        if not code_root.exists():
            continue
        for path in sorted(code_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS:
                yield path


def main() -> int:
    files = list(source_files())
    if not files:
        print("No project source files found to check.")
        return 0

    counts = [
        (path, len(path.read_text(encoding="utf-8").splitlines()))
        for path in files
    ]
    over_limit = [(path, count) for path, count in counts if count > MAX_LINES]

    if over_limit:
        for path, count in over_limit:
            print(
                f"OVER LIMIT: {path.relative_to(ROOT)} has {count} lines "
                f"(max {MAX_LINES})"
            )
        return 1

    largest_path, largest_count = max(counts, key=lambda item: item[1])
    print(
        f"OK: {len(counts)} project code files; all are at most {MAX_LINES} lines. "
        f"Largest: {largest_path.relative_to(ROOT)} ({largest_count} lines)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
