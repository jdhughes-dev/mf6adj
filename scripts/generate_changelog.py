"""Generate the changelog entry for a new release.

The entry is built by git-cliff from the commits merged since the last tag,
which selects on the conventional commit type in each subject. What is kept and
what is dropped is set in `cliff.toml`.

git-cliff writes a whole changelog rather than an entry, and this file is not
wholly generated: the breaking changes of a release are written by hand and
have to survive the next one. So the entry is generated on its own and inserted
below the header, leaving everything already in the file alone.

Usage:
    python scripts/generate_changelog.py --version 1.2.3
    python scripts/generate_changelog.py --version 1.2.3 --dry-run
"""

import argparse
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

_project_root = Path(__file__).parent.parent
_changelog_path = _project_root / "changelog" / "CHANGELOG.md"
_config_path = _project_root / "cliff.toml"

# the range git-cliff has been read against; a new major version of it may
# write the entry differently
_requirement = "git-cliff>=2.13,<3"


def git_cliff_command():
    """Return the command that runs git-cliff, or None if there is none.

    Prefers an installed git-cliff, which is what the pixi environment has, and
    falls back to running it through uv, which is what the release workflow has.
    """
    if shutil.which("git-cliff"):
        return ["git-cliff"]
    if shutil.which("uvx"):
        return ["uvx", "--from", _requirement, "git-cliff"]
    return None


def build_entry(version):
    """Return the entry for a version, built from the commits since the last tag."""
    command = git_cliff_command()
    if command is None:
        raise SystemExit(
            f"git-cliff was not found, and neither was uvx to run it with.\n"
            f"Install it with `pixi install`, or `uv tool install {_requirement}`."
        )

    result = subprocess.run(
        [
            *command,
            "--config",
            str(_config_path),
            "--unreleased",
            "--tag",
            f"v{version}",
        ],
        cwd=_project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"git-cliff failed:\n{result.stderr}")

    entry = result.stdout.strip()
    if not entry:
        raise SystemExit(
            "git-cliff produced no entry. Nothing has been merged since the "
            "last tag whose type reaches the changelog; see cliff.toml."
        )
    return entry


def insert_entry(entry, dry_run=False):
    """Insert the entry below the header, above the entry of the last release."""
    content = _changelog_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    insert_pos = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("## "):
            insert_pos = i
            break

    updated = "".join(lines[:insert_pos]) + entry + "\n\n" + "".join(lines[insert_pos:])

    if dry_run:
        print(f"--- DRY RUN: {_changelog_path.name} would gain ---")
        print(entry)
    else:
        _changelog_path.write_text(updated, encoding="utf-8")
        print(f"Updated {_changelog_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="generate_changelog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        help="Version number for the new release (e.g. 1.2.3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the entry without writing to disk",
    )
    args = parser.parse_args()

    insert_entry(build_entry(args.version), dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
