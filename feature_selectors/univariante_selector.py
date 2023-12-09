from sklearn.feature_selection import SelectKBest, f_classif
from .base_selector import AbstractFeatureSelector

class UnivariateFeatureSelector(AbstractFeatureSelector):
    def __init__(self, k=10):
        super().__init__()
        self.selector = SelectKBest(f_classif, k=k)

    def fit(self, X, y):
        self.selector.fit(X, y)

    def transform(self, X):
        return self.selector.transform(X)
