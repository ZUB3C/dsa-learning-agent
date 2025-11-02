"""
Script to export all source code from src/ directory to a text file for LLM context.
Generates directory tree structure followed by file contents.
Only includes .py files and ignores __pycache__ directories.
"""

import os
from pathlib import Path

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# Files to exclude from output (relative to project root)
EXCLUDE_FILES = [
    "src/scripts/generate_endpoint_report.py",
]

# Files outside src/ to forcefully include (relative to project root)
INCLUDE_EXTERNAL_FILES = [
    "README.md",
]


def is_excluded(filepath: Path, project_root: Path) -> bool:
    """Check if file should be excluded."""
    try:
        relative_path = filepath.relative_to(project_root)
        return str(relative_path) in EXCLUDE_FILES
    except ValueError:
        return False


def generate_tree(
    directory: Path,
    prefix: str = "",
    exclude_path: Path | None = None,
    project_root: Path | None = None,
) -> list[str]:
    """Generate a visual tree structure of the directory."""
    tree_lines: list[str] = []

    try:
        # Get all items in directory and sort them
        items = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name))

        for index, item in enumerate(items):
            # Skip __pycache__ directories
            if item.is_dir() and item.name == "__pycache__":
                continue

            # Skip non-.py files
            if item.is_file() and not item.name.endswith(".py"):
                continue

            is_last_item = index == len(items) - 1

            # Skip the script itself
            if exclude_path and item.resolve() == exclude_path.resolve():
                continue

            # Skip excluded files
            if project_root and is_excluded(item, project_root):
                continue

            # Create tree characters
            connector = "└── " if is_last_item else "├── "
            tree_lines.append(f"{prefix}{connector}{item.name}")

            # Recursively process directories
            if item.is_dir():
                extension = "    " if is_last_item else "│   "
                subtree = generate_tree(item, prefix + extension, exclude_path, project_root)
                if subtree:  # Only add if directory has .py files
                    tree_lines.extend(subtree)

    except PermissionError:
        pass

    return tree_lines


def read_all_files(
    directory: Path, exclude_path: Path | None = None, project_root: Path | None = None
) -> list[dict[str, str]]:
    """Read all .py files recursively from directory, excluding __pycache__."""
    file_contents: list[dict[str, str]] = []

    for root, dirs, files in os.walk(directory):
        # Exclude __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]

        # Sort for consistent output
        dirs.sort()
        files.sort()

        for filename in files:
            # Only process .py files
            if not filename.endswith(".py"):
                continue

            filepath = Path(root) / filename

            # Skip the script itself
            if exclude_path and filepath.resolve() == exclude_path.resolve():
                continue

            # Skip excluded files
            if project_root and is_excluded(filepath, project_root):
                continue

            # Get relative path for cleaner output
            relative_path = filepath.relative_to(directory.parent)

            try:
                # Try to read as text file
                with Path(filepath).open(encoding="utf-8") as f:
                    content = f.read()

                file_contents.append({"path": str(relative_path), "content": content})
            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read
                file_contents.append({
                    "path": str(relative_path),
                    "content": "[Unable to read file]",
                })

    return file_contents


def read_external_files(project_root: Path) -> list[dict[str, str]]:
    """Read external files that should be forcefully included."""
    file_contents: list[dict[str, str]] = []

    for file_path in INCLUDE_EXTERNAL_FILES:
        filepath = project_root / file_path

        if not filepath.exists():
            print(f"⚠️  Warning: External file not found: {file_path}")
            continue

        try:
            with filepath.open(encoding="utf-8") as f:
                content = f.read()

            file_contents.append({"path": file_path, "content": content})
        except (UnicodeDecodeError, PermissionError) as e:
            print(f"⚠️  Warning: Could not read {file_path}: {e}")
            file_contents.append({
                "path": file_path,
                "content": "[Unable to read file]",
            })

    return file_contents


def count_tokens(text: str, model: str = "claude-sonnet-4-5") -> int:
    """Offline token counting for Claude models (approximation)."""
    if "claude" in model.lower():
        # Use p50k_base encoding as approximation for Claude
        try:
            encoding = tiktoken.get_encoding("p50k_base")
            # Claude typically uses 16-30% more tokens than GPT models
            estimated_tokens = len(encoding.encode(text))
            return int(estimated_tokens * 1.2)  # Add 20% buffer
        except Exception:
            return len(text) // 4
    else:
        # For GPT models
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def count_lines(content: str) -> tuple[int, int]:
    """
    Count total lines and non-blank lines in content.

    Returns:
        tuple: (total_lines, non_blank_lines)

    """
    lines = content.splitlines()
    total_lines = len(lines)
    non_blank_lines = sum(1 for line in lines if line.strip())
    return total_lines, non_blank_lines


def main() -> None:
    """Main function to generate codebase context file."""
    # Get the script's own path to exclude it
    script_path: Path = Path(__file__).resolve()

    # Set source directory (parent of scripts directory)
    src_dir: Path = Path(__file__).parent.parent

    # Project root directory
    project_root: Path = src_dir.parent

    # Output file location (in project root)
    output_file: Path = project_root / "codebase_context.txt"

    # Model used for token counting
    model = "claude-sonnet-4-5"

    print(f"Scanning directory: {src_dir}")
    print(f"Output file: {output_file}")
    print("Filtering: Only .py files, excluding __pycache__")
    print(f"Excluded files: {', '.join(EXCLUDE_FILES) if EXCLUDE_FILES else 'None'}")
    print(
        f"External files: {', '.join(INCLUDE_EXTERNAL_FILES) if INCLUDE_EXTERNAL_FILES else 'None'}\n"  # noqa: E501
    )

    # Collect all content
    all_content: list[str] = []

    # Initialize line counters
    total_lines = 0
    non_blank_lines = 0

    with Path(output_file).open("w", encoding="utf-8") as f:
        # Write header
        header = "=" * 80 + "\n"
        header += "CODEBASE CONTEXT FOR LLM\n"
        header += "=" * 80 + "\n\n"
        f.write(header)
        all_content.append(header)

        # Write directory structure
        structure_section = "DIRECTORY STRUCTURE\n"
        structure_section += "-" * 80 + "\n"
        structure_section += f"{src_dir.name}/\n"

        tree: list[str] = generate_tree(
            src_dir, exclude_path=script_path, project_root=project_root
        )
        for line in tree:
            structure_section += line + "\n"

        # Add external files to tree
        if INCLUDE_EXTERNAL_FILES:
            structure_section += "\nExternal files:\n"
            for ext_file in INCLUDE_EXTERNAL_FILES:
                if (project_root / ext_file).exists():
                    structure_section += f"├── {ext_file}\n"

        structure_section += "\n" + "=" * 80 + "\n\n"
        f.write(structure_section)
        all_content.append(structure_section)

        # Write file contents
        content_header = "FILE CONTENTS\n"
        content_header += "-" * 80 + "\n\n"
        f.write(content_header)
        all_content.append(content_header)

        # Read external files first
        external_files: list[dict[str, str]] = read_external_files(project_root)

        # Read src files
        src_files: list[dict[str, str]] = read_all_files(
            src_dir, exclude_path=script_path, project_root=project_root
        )

        # Combine all files
        all_files = external_files + src_files

        for file_info in all_files:
            # Count lines for this file
            file_total, file_non_blank = count_lines(file_info["content"])
            total_lines += file_total
            non_blank_lines += file_non_blank

            file_section = "\n" + "=" * 80 + "\n"
            file_section += f"FILE: {file_info['path']}\n"
            file_section += "=" * 80 + "\n\n"
            file_section += file_info["content"]
            file_section += "\n\n"

            f.write(file_section)
            all_content.append(file_section)

    # Calculate statistics (don't write to file, only print)
    full_text = "".join(all_content)
    char_count = len(full_text)
    token_count = count_tokens(full_text, model=model)

    # Print summary
    print(f"✓ Successfully exported {len(all_files)} files")
    print(f"  • External files: {len(external_files)}")
    print(f"  • Source files: {len(src_files)}")
    print("\n📊 Statistics:")
    print(f"  • Total files: {len(all_files)}")
    print(f"  • Total lines: {total_lines:,}")
    print(f"  • Code lines (non-blank): {non_blank_lines:,}")
    print(f"  • Total characters: {char_count:,}")
    print(f"  • Total tokens (estimated): {token_count:,}")

    if TIKTOKEN_AVAILABLE:
        if "claude" in model.lower():
            print(f"  • Token encoding: p50k_base approximation for {model} (+20% buffer)")
        else:
            print(f"  • Token encoding: cl100k_base for {model}")
    else:
        print("  • Token estimation: ~4 chars/token (tiktoken not available)")

    print(f"\n💾 Output: {output_file}")


if __name__ == "__main__":
    main()
