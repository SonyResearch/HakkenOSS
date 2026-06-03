import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from hakken_agents.utils.lightrag.storage import BaseKVStorage
from hakken_agents.utils.rag_anything.parser import (
    DoclingParser,
    MineruExecutionError,
    MineruParser,
)

from .config import ParseDocumentConfig
from .enums import ParserType


class DocumentParser:
    def __init__(self, config: ParseDocumentConfig, parse_cache: BaseKVStorage | None = None):
        self.config = config
        self.parse_cache: BaseKVStorage | None = parse_cache

    def update_config(self, **kwargs):
        self.config = self.config.model_copy(update=kwargs)

    @property
    def file_path(self) -> Path:
        file_path = Path(self.config.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path

    async def parse(self, **kwargs) -> tuple[list[dict[str, Any]], str]:
        """
        Parse document with caching support

        Args:
            file_path: Path to the file to parse
            output_dir: Output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)

        Returns:
            tuple[list[dict[str, Any]], str]: (content_list, doc_id)
        """

        logger.info(f"Starting document parsing: {self.file_path}")

        parser_kwargs = self.config.get_parser_kwargs(**kwargs)

        # Generate cache key based on file and configuration
        cache_key = self.generate_cache_key(**parser_kwargs)
        logger.info(f"Cache key: {cache_key}")
        # Check cache first
        cached_result = await self.get_cached_result(cache_key=cache_key, **parser_kwargs)
        if cached_result is not None:
            content_list, doc_id = cached_result
            logger.info(f"Using cached parsing result for: {self.file_path}")
            if self.config.display_stats:
                logger.info(f"* Total blocks in cached content_list: {len(content_list)}")
            return content_list, doc_id

        # Choose appropriate parsing method based on file extension
        ext = self.file_path.suffix.lower()

        try:
            doc_parser = (
                DoclingParser() if self.config.parser == ParserType.DOCLING else MineruParser()
            )

            # Log parser and method information
            logger.info(
                f"Using {self.config.parser} parser with method: {self.config.parse_method}"
            )

            if ext in [".pdf"]:
                logger.info("Detected PDF file, using parser for PDF...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=self.file_path,
                    output_dir=self.config.output_dir,
                    method=str(self.config.parse_method),
                    **parser_kwargs,
                )
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            ]:
                logger.info("Detected image file, using parser for images...")
                # Use the selected parser's image parsing capability
                if hasattr(doc_parser, "parse_image"):
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=self.file_path,
                        output_dir=self.config.output_dir,
                        **kwargs,
                    )
                else:
                    # Fallback to MinerU for image parsing if current parser doesn't support it
                    logger.warning(
                        f"{self.config.parser} parser doesn't support image parsing, falling back to MinerU"
                    )
                    content_list = MineruParser().parse_image(
                        image_path=self.file_path,
                        output_dir=self.config.output_dir,
                        **parser_kwargs,
                    )
            elif ext in [
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".html",
                ".htm",
                ".xhtml",
            ]:
                logger.info("Detected Office or HTML document, using parser for Office/HTML...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=self.file_path,
                    output_dir=self.config.output_dir,
                    **parser_kwargs,
                )
            else:
                # For other or unknown formats, use generic parser
                logger.info(
                    f"Using generic parser for {ext} file (method={self.config.parse_method})..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_document,
                    file_path=self.file_path,
                    method=str(self.config.parse_method),
                    output_dir=self.config.output_dir,
                    **parser_kwargs,
                )

        except MineruExecutionError as e:
            logger.error(f"Mineru command failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during parsing with {self.config.parser} parser: {e!s}")
            raise e

        msg = f"Parsing {self.file_path} complete! Extracted {len(content_list)} content blocks"
        logger.info(msg)

        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content was extracted")

        # Generate doc_id based on content
        doc_id = self.generate_content_based_doc_id(content_list)

        # Store result in cache
        await self.store_cached_result(content_list, doc_id, cache_key=cache_key, **parser_kwargs)

        # Display content statistics if requested
        if self.config.display_stats:
            logger.info("\nContent Information:")
            logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            logger.info("* Content block types:")
            for block_type, count in block_types.items():
                logger.info(f"  - {block_type}: {count}")

        return content_list, doc_id

    async def store_cached_result(
        self,
        content_list: list[dict[str, Any]],
        doc_id: str,
        cache_key: str | None = None,
        **kwargs,
    ) -> None:
        """
        Store parsing result in cache

        Args:
            cache_key: Cache key to store under
            content_list: Content list to cache
            doc_id: Content-based document ID
            file_path: Path to the file for mtime storage
            parse_method: Parse method used
            **kwargs: Additional parser parameters
        """
        if self.parse_cache is None:
            return

        cache_key = cache_key or self.generate_cache_key(**kwargs)

        try:
            # Get file modification time
            file_mtime = self.file_path.stat().st_mtime

            # Create parsing configuration
            parse_config = {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
            }

            # Add relevant kwargs to config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                    "vlm_url",
                ]
            }
            parse_config.update(relevant_kwargs)

            cache_data = {
                cache_key: {
                    "content_list": content_list,
                    "doc_id": doc_id,
                    "mtime": file_mtime,
                    "parse_config": parse_config,
                    "cached_at": time.time(),
                    "cache_version": "1.0",
                }
            }
            await self.parse_cache.upsert(cache_data)
            # Ensure data is persisted to disk
            await self.parse_cache.index_done_callback()
            logger.info(f"Stored parsing result in cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error storing to parse cache: {e}")

    def generate_cache_key(self, **kwargs) -> str:
        """
        Generate cache key based on file path and parsing configuration

        Args:
            file_path: Path to the file
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            str: Cache key for the file and configuration
        """
        mtime = self.file_path.stat().st_mtime

        # Create configuration dict for cache key
        config_dict = {
            "file_path": str(self.file_path.absolute()),
            "mtime": mtime,
            "parser": self.config.parser,
            "parse_method": self.config.parse_method,
        }

        # Add relevant kwargs to config
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "lang",
                "device",
                "start_page",
                "end_page",
                "formula",
                "table",
                "backend",
                "source",
            ]
        }
        config_dict.update(relevant_kwargs)

        # Generate hash from config
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def generate_content_based_doc_id(self, content_list: list[dict[str, Any]]) -> str:
        """
        Generate doc_id based on document content

        Args:
            content_list: Parsed content list

        Returns:
            str: Content-based document ID with doc- prefix
        """
        from hakken_agents.utils.lightrag.utils import compute_mdhash_id

        # Extract key content for ID generation
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # For text content, use the text
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # For other content types, use key identifiers
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # For other types, use string representation
                    content_hash_data.append(str(item))

        # Create a content signature
        content_signature = "\n".join(content_hash_data)

        # Generate doc_id from content
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def get_cached_result(
        self, cache_key: str | None = None, **kwargs
    ) -> tuple[list[dict[str, Any]], str] | None:
        """
        Get cached parsing result if available and valid

        Args:
            TODO
        Returns:
            tuple[List[Dict[str, Any]], str] | None: (content_list, doc_id) or None if not found/invalid
        """
        if self.parse_cache is None:
            return None

        try:
            cache_key = cache_key or self.generate_cache_key(**kwargs)

            cached_data = await self.parse_cache.get_by_id(cache_key)
            if not cached_data:
                return None

            # Check file modification time
            current_mtime = self.file_path.stat().st_mtime
            cached_mtime = cached_data.get("mtime", 0)

            if current_mtime != cached_mtime:
                logger.debug(f"Cache invalid - file modified: {cache_key}")
                return None

            # Check parsing configuration
            cached_config = cached_data.get("parse_config", {})
            current_config = {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
            }

            # Add relevant kwargs to current config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                    "vlm_url",
                ]
            }
            current_config.update(relevant_kwargs)

            if cached_config != current_config:
                logger.debug(f"Cache invalid - config changed: {cache_key}")
                return None

            content_list = cached_data.get("content_list", [])
            doc_id = cached_data.get("doc_id")

            if content_list and doc_id:
                logger.debug(f"Found valid cached parsing result for key: {cache_key}")
                return content_list, doc_id
            logger.debug(f"Cache incomplete - missing content or doc_id: {cache_key}")
            return None

        except Exception as e:
            logger.warning(f"Error accessing parse cache: {e}")

        return None
