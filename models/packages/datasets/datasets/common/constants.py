from strenum import StrEnum


class DataSplits(StrEnum):
    TRAIN = "train"
    VALID = "val"
    TEST = "test"
    ALL = "all"
