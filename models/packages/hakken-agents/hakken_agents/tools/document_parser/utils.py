import re
from typing import Any

from hakken_agents.tools.document_parser.schemas import Chunk


def _extract_text(item: dict[str, Any]) -> str | None:
    """Extract text from content item. Returns None if not a text-bearing type."""
    if item.get("type") == "text":
        return item.get("text", "")
    if item.get("type") == "list":
        items = item.get("list_items", [])
        return "\n".join(str(i) for i in items) if items else ""
    return None


def _split_at_sentence_boundaries(text: str, max_size: int) -> list[str]:
    """Split text at sentence boundaries. No chunk exceeds max_size."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_size:
        return [text]

    # Split on sentence boundaries: . ! ? followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        sep_len = 1 if current else 0
        part_len = len(part) + sep_len
        if current_len + part_len <= max_size:
            current.append(part)
            current_len += part_len
        else:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            if len(part) > max_size:
                chunks.append(part)
            else:
                current.append(part)
                current_len = len(part)

    if current:
        chunks.append(" ".join(current))
    return chunks


def create_chunks(
    content_list: list[dict[str, Any]],
    max_chunk_size: int = 1000,
    doc_id: str = "default_doc_id",
) -> list[Chunk]:
    """
    Create chunks from parsed document content.

    - Excludes header and page_number blocks.
    - Section headers (text_level) are stored in levels only, not in chunk text.
    - Splits at sentence boundaries when exceeding max_chunk_size.
    - List items are joined with newlines.
    - Chunk.size is the character count of text.
    """
    EXCLUDED_TYPES = {"header", "page_number"}
    chunks: list[Chunk] = []
    current_levels: dict[str, Any] = {}
    current_parts: list[str] = []
    current_len = 0
    current_page_idx: int | None = None

    for item in content_list:
        if item.get("type") in EXCLUDED_TYPES:
            continue

        text = _extract_text(item)
        if text is None or not text.strip():
            continue
        text = text.strip()
        page_idx = item.get("page_idx", 0)

        # Section header: flush current content first (it belongs to previous section),
        # then update levels. Header text goes in levels only, not in chunk text.
        if "text_level" in item:
            header = item.get("text", "").strip()
            if header:
                if current_parts:
                    full = "\n\n".join(current_parts)
                    for sub in _split_at_sentence_boundaries(full, max_chunk_size):
                        if sub.strip():
                            chunks.append(
                                Chunk(
                                    text=sub,
                                    size=len(sub),
                                    levels=dict(current_levels),
                                    page_idx=current_page_idx or 0,
                                    doc_id=doc_id,
                                )
                            )
                    current_parts = []
                    current_len = 0
                level_key = f"level_{item['text_level']}"
                level_num = (
                    int(item["text_level"]) if isinstance(item["text_level"], (int, str)) else 1
                )
                # When we see level N, clear level_{N+1}, level_{N+2}, etc. (hierarchy reset)
                current_levels = {
                    k: v
                    for k, v in current_levels.items()
                    if not (k.startswith("level_") and int(k.split("_")[1]) > level_num)
                }
                current_levels[level_key] = header
                continue  # Do not add header text to chunk content

        sep = "\n\n" if current_parts else ""
        added_len = len(sep) + len(text)

        if current_len + added_len <= max_chunk_size:
            current_parts.append(text)
            current_len += added_len
            if current_page_idx is None:
                current_page_idx = page_idx
        else:
            # Combine current content with new text, then split at sentence boundaries
            combined_parts = current_parts + [text]
            full = "\n\n".join(combined_parts)
            for sub in _split_at_sentence_boundaries(full, max_chunk_size):
                if sub.strip():
                    chunks.append(
                        Chunk(
                            text=sub,
                            size=len(sub),
                            levels=dict(current_levels),
                            page_idx=current_page_idx or page_idx,
                            doc_id=doc_id,
                        )
                    )
            current_parts = []
            current_len = 0

    if current_parts:
        full = "\n\n".join(current_parts)
        for sub in _split_at_sentence_boundaries(full, max_chunk_size):
            if sub.strip():
                chunks.append(
                    Chunk(
                        text=sub,
                        size=len(sub),
                        levels=dict(current_levels),
                        page_idx=current_page_idx or 0,
                        doc_id=doc_id,
                    )
                )

    return chunks
