import logging


class RawFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return str(record.msg)

    def add_labels(self, labels: dict) -> None:
        pass
