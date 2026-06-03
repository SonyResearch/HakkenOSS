class ScoreScalerNotFoundError(Exception):
    def __init__(self, path: str):
        message = (
            f"Score scaler file not found or could not be loaded: '{path}'. "
            "Ensure the path is correct and the file exists."
        )
        super().__init__(message)
