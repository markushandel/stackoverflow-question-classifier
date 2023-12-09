from abc import ABC, abstractmethod

class AbstractFeatureSelector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass

