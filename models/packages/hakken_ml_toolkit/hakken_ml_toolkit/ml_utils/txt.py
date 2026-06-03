from pathlib import Path

from loguru import logger


class TXTUtils:
    """Utility class for handling text file operations."""

    @staticmethod
    def read_lines(  # noqa: PLR0913
        filepath: str | Path,
        encoding: str = "utf-8",
        strip_whitespace: bool = True,
        skip_empty_lines: bool = True,
        skip_comments: bool = True,
        comment_char: str = "#",
    ) -> list[str]:
        filepath = Path(filepath)

        if not filepath.exists():
            logger.exception(f"File not found: {filepath}")
            raise FileNotFoundError()

        try:
            with filepath.open("r", encoding=encoding) as file:
                lines = file.readlines()

            # Process the lines according to the parameters
            processed_lines = []
            for line_raw in lines:
                line = line_raw
                if strip_whitespace:
                    line = line_raw.strip()

                # Skip empty lines if requested
                if skip_empty_lines and not line:
                    continue

                # Skip comments if requested
                if skip_comments and line.startswith(comment_char):
                    continue

                processed_lines.append(line)

        except UnicodeDecodeError:
            logger.exception(f"Encoding error while reading {filepath}")
            raise
        except OSError:
            logger.exception(f"IO error while reading {filepath}")
            raise
        except Exception:
            logger.exception(f"Unexpected error while reading {filepath}")
            raise
        else:
            return processed_lines
