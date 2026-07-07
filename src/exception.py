class TitanicException(Exception):
    def __init__(self, message):
        super().__init__(message)


class DatasetNotFoundError(TitanicException):
    pass


class DataProcessingError(TitanicException):
    pass


class ModelTrainingError(TitanicException):
    pass


class PredictionError(TitanicException):
    pass
