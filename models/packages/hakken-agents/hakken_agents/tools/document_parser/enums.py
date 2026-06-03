from strenum import StrEnum


class ParserType(StrEnum):
    DOCLING = "docling"
    MINERU = "mineru"


class ParseMethodType(StrEnum):
    AUTO = "auto"
    OCR = "ocr"
    TXT = "txt"
