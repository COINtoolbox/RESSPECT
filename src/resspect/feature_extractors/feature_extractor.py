from abc import ABC, abstractmethod

#!
#! Maybe we should just get rid of this class???
#!

class ResspectFeatureExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, band: str) -> list:
        pass

    @abstractmethod
    def fit_all(self):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_features(cls, filters: list) -> list[str]:
        raise NotImplementedError()
