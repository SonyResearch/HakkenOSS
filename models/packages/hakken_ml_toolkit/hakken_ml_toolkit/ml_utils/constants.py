from strenum import StrEnum


class ActivationType(StrEnum):
    ELU = "elu"
    IDENTITY = "identity"
    LEAKY_RELU = "lrelu"
    PRELU = "prelu"
    RELU = "relu"
    SELU = "selu"
    SIGMOID = "sigmoid"
    SIN = "sin"
    SOFTMAX = "softmax"
    TANH = "tanh"


class InitStrategy(StrEnum):
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
