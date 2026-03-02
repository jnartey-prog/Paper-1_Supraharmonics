from pathlib import Path


EXCLUDE_DIRS = {
    ".uv-cache",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "venv",
    ".venv",
    "env",
    "node_modules",
    "build",
    "dist",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".ruff_cache",
}

INCLUDE_EXTENSIONS = {".py"}


def is_excluded(path: Path, excluded_dirs: set[str]) -> bool:
    """Return True if any path segment matches an excluded directory name."""
    return any(part in excluded_dirs for part in path.parts)


def iter_included_files(
    root_path: Path, include_extensions: set[str], excluded_dirs: set[str]
) -> list[Path]:
    """Collect and return sorted source files that match allowed extensions."""
    files: list[Path] = []
    for path in root_path.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path.relative_to(root_path), excluded_dirs):
            continue
        if path.suffix.lower() in include_extensions:
            files.append(path)
    return sorted(files, key=lambda p: str(p.relative_to(root_path)).replace("\\", "/"))


def build_project_tree(root_path: Path, included_files: list[Path]) -> str:
    """Create a tree-style text view using only included files and their parent folders."""
    lines = [f"{root_path.name}/"]
    seen_dirs: set[Path] = set()

    for file_path in included_files:
        rel = file_path.relative_to(root_path)
        parts = rel.parts

        current = Path()
        for depth, part in enumerate(parts[:-1], start=1):
            current = current / part
            if current not in seen_dirs:
                lines.append(f"{'    ' * depth}{part}/")
                seen_dirs.add(current)

        lines.append(f"{'    ' * len(parts)}{parts[-1]}")

    return "\n".join(lines) + "\n"


def bundle_files(
    root_dir: str = ".",
    output_file: str = "project_code_bundle.txt",
    include_extensions: set[str] | None = None,
    excluded_dirs: set[str] | None = None,
) -> int:
    """Write a project tree and file contents into a single text bundle."""
    root_path = Path(root_dir).resolve()
    include_extensions = include_extensions or INCLUDE_EXTENSIONS
    excluded_dirs = excluded_dirs or EXCLUDE_DIRS
    output_path = Path(output_file).resolve()

    print(f"Bundling source files from: {root_path}")
    print(f"Output file: {output_path}")

    files = iter_included_files(root_path, include_extensions, excluded_dirs)

    with output_path.open("w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("PROJECT STRUCTURE\n")
        out.write("=" * 80 + "\n\n")
        out.write(build_project_tree(root_path, files))
        out.write("\n" + "=" * 80 + "\n")
        out.write("FILE CONTENTS\n")
        out.write("=" * 80 + "\n")

        bundled_count = 0
        for file_path in files:
            rel_path = file_path.relative_to(root_path).as_posix()
            out.write(f"\n{'=' * 80}\n")
            out.write(f"FILE: {rel_path}\n")
            out.write(f"{'=' * 80}\n\n")

            try:
                content = file_path.read_text(encoding="utf-8")
                out.write(content)
                if content and not content.endswith("\n"):
                    out.write("\n")
                bundled_count += 1
                print(f"Added: {rel_path}")
            except Exception as exc:
                message = f"Error reading file: {exc}"
                out.write(message + "\n")
                print(f"{message} ({rel_path})")

    print(f"\nBundle created: {output_path}")
    print(f"Total files bundled: {bundled_count}")
    return bundled_count


def main() -> None:
    """Run the bundling process with default settings."""
    bundle_files()


if __name__ == "__main__":
    main()
