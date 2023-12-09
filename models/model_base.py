from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV


class AbstractModelTrainer(ABC):
    def __init__(self, param_grid):
        self.model = None
        self.param_grid = param_grid
        self.grid_search = None

    def optimize_hyperparameters(self, x, y):
        # Use multithreading by performing grid search in parallel
        self.grid_search = (
            RandomizedSearchCV(self.model, self.param_grid, cv=10, scoring="f1_macro", return_train_score=True, n_jobs=-1))
        self.grid_search.fit(x, y)
        self.model = self.grid_search.best_estimator_

    def train(self, x, y):
        self.optimize_hyperparameters(x, y)

    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred

    def evaluate_model(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return self.evaluate_predictions(y_pred, y_test)

    @staticmethod
    def evaluate_predictions(y_pred, y):
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred,
                      average='macro')
        return accuracy, f1
