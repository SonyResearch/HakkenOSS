class InputError(Exception):
    """Mixin for input-related exceptions, throw 422 error"""

    pass


class LogicError(Exception):
    """Mixin for internal logic-related exceptions, throw 500 error"""

    pass


class SearchError(Exception):
    pass


class SearchInputError(SearchError, InputError):
    pass


class SearchLogicError(SearchError, LogicError):
    pass


class KGError(Exception):
    pass


class KGInputError(KGError, InputError):
    pass


class KGLogicError(KGError, LogicError):
    pass


class PredictorError(Exception):
    pass


class PredictorLogicError(Exception):
    pass


class PredictorInputError(Exception):
    pass


class ScoreAggregatorError(Exception):
    pass


class InitializationError(Exception):
    pass
