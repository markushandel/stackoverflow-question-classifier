from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from .custom_hyper_param import CustomRandomizedSearchCV


class AbstractModelTrainer(ABC):
    def __init__(self, param_grid):
        self.model = None
        self.param_grid = param_grid

    def optimize_hyperparameters(self, x, y):
        # Use multithreading by performing grid search in parallel
        custom_search = RandomizedSearchCV(self.model, self.param_grid, cv=10, scoring="f1_macro", return_train_score=True, n_jobs=-1)

        # custom_search = CustomRandomizedSearchCV(
        #     model=self.model,
        #     param_distributions=self.param_grid,
        #     cv=5,                           # Number of cross-validation folds
        #     scoring='f1_macro',             # Scoring metric
        #     n_iter=10,                      # Number of iterations for random search
        #     n_jobs=-1,                      # Use all cores
        #     random_state=42                 # Seed for reproducibility
        # )
        # Fit the search object to your data
        custom_search.fit(x, y)

        # Retrieve the best model and its parameters and score
        self.model = custom_search.best_estimator_

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
