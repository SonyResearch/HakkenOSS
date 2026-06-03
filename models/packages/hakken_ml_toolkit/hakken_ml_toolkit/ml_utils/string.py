from __future__ import annotations

from loguru import logger


class StringUtils:
    @staticmethod
    def to_txt_file(my_str: str, file_path: str, mode: str = "w") -> str:
        if my_str[-1] != "\n":
            my_str += "\n"
        with open(file_path, mode) as f:
            f.write(my_str)

        logger.info(f"Saved to {file_path}")
        return file_path
