from abc import ABC, abstractmethod

class AbstractDataProcessor(ABC):
    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def transform(self, df):
        pass

    def process_data(self, X_train, X_test):
        # Fit the processor with the training data
        self.fit(X_train)

        # Transform both training and test data
        X_train_transformed = self.transform(X_train)
        X_test_transformed = self.transform(X_test)

        return X_train_transformed, X_test_transformed
