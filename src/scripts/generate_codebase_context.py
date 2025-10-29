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


def generate_tree(
    directory: Path, prefix: str = "", exclude_path: Path | None = None
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

            # Create tree characters
            connector = "└── " if is_last_item else "├── "
            tree_lines.append(f"{prefix}{connector}{item.name}")

            # Recursively process directories
            if item.is_dir():
                extension = "    " if is_last_item else "│   "
                subtree = generate_tree(item, prefix + extension, exclude_path)
                if subtree:  # Only add if directory has .py files
                    tree_lines.extend(subtree)

    except PermissionError:
        pass

    return tree_lines


def read_all_files(directory: Path, exclude_path: Path | None = None) -> list[dict[str, str]]:
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


def main() -> None:  # noqa: PLR0915
    """Main function to generate codebase context file."""
    # Get the script's own path to exclude it
    script_path: Path = Path(__file__).resolve()

    # Set source directory (parent of scripts directory)
    src_dir: Path = Path(__file__).parent.parent

    # Output file location (in project root)
    output_file: Path = src_dir.parent / "codebase_context.txt"

    # Model used for token counting
    model = "claude-sonnet-4-5"

    print(f"Scanning directory: {src_dir}")
    print(f"Output file: {output_file}")
    print("Filtering: Only .py files, excluding __pycache__\n")

    # Collect all content
    all_content: list[str] = []

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

        tree: list[str] = generate_tree(src_dir, exclude_path=script_path)
        for line in tree:
            structure_section += line + "\n"

        structure_section += "\n" + "=" * 80 + "\n\n"
        f.write(structure_section)
        all_content.append(structure_section)

        # Write file contents
        content_header = "FILE CONTENTS\n"
        content_header += "-" * 80 + "\n\n"
        f.write(content_header)
        all_content.append(content_header)

        files: list[dict[str, str]] = read_all_files(src_dir, exclude_path=script_path)

        for file_info in files:
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
    code_lines_count = len([i for i in full_text.split("\n") if i.strip()])
    lines_count = len(list(full_text.split("\n")))

    # Print summary
    print(f"✓ Successfully exported {len(files)} Python files")
    print("\n📊 Statistics:")
    print(f"  • Total files: {len(files)}")
    print(f"  • Total characters: {char_count:,}")
    print(f"  • Total code lines: {code_lines_count:,} ({lines_count:,} with whitespaces)")
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
