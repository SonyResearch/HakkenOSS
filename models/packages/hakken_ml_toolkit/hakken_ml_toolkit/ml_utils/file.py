from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

# TODO: Move to the file_manager package


class FileUtils:
    @staticmethod
    def find(
        root_folder: str | Path,
        pattern: str = "*",
        recursive: bool = True,
        include_dirs: bool = False,
        exclude_patterns: list[str] | None = None,
    ) -> list[Path]:
        # Convert to Path object and resolve to absolute path
        root_path = Path(root_folder).resolve()

        if not root_path.exists():
            raise FileNotFoundError(root_path)

        if not root_path.is_dir():
            raise NotADirectoryError(root_path)

        try:
            # Compile exclude patterns if provided
            exclude_regexes = None
            if exclude_patterns:
                exclude_regexes = [re.compile(pattern) for pattern in exclude_patterns]

            # Initialize results list
            results: list[Path] = []

            # Define a helper function to check if path should be excluded
            def is_excluded(path: Path) -> bool:
                if not exclude_regexes:
                    return False
                return any(regex.search(str(path)) for regex in exclude_regexes)

            # Define the search function
            def search_directory(current_path: Path) -> None:
                try:
                    # Use rglob if recursive, else glob
                    glob_func = current_path.rglob if recursive else current_path.glob

                    for path in glob_func(pattern):
                        # Skip if path matches exclude patterns
                        if is_excluded(path):
                            continue

                        # Skip directories if not include_dirs
                        if path.is_dir() and not include_dirs:
                            continue

                        results.append(path)

                except PermissionError:
                    logger.warning(f"Permission denied accessing {current_path}")
                    raise
                except Exception:
                    logger.exception(f"Error processing {current_path}")
                    raise

            # Start the search
            search_directory(root_path)

            # Sort results for consistency
            results.sort()

        except Exception:
            logger.exception(f"Error searching in {root_folder}")
            raise
        else:
            logger.info(f"Found {len(results)} paths in {root_path}")
            return results
