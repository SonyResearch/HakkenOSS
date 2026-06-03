class ParsingError(Exception):
    pass


class ParsingInputError(ParsingError):
    pass


class ParsingLogicError(ParsingError):
    pass


class GroundingError(Exception):
    pass


class GroundingLogicError(GroundingError):
    pass


class GroundingInputError(GroundingError):
    pass
