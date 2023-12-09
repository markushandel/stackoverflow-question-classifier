from sklearn.model_selection import ParameterSampler, GridSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.model_selection import cross_val_score


class CustomRandomizedSearchCV:
    def __init__(self, model, param_distributions, cv, scoring, n_iter, n_jobs, random_state=None):
        self.model = model
        self.param_distributions = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.grid_search = None

    def sample_params(self):
        # Randomly sample a number of parameters
        return list(ParameterSampler(self.param_distributions, n_iter=self.n_iter, random_state=self.random_state))

    def refine_params(self, best_params):
        # Create a grid around the best random search parameters
        # Assuming parameters are numeric, you could define a grid manually
        # or by using a rule to define the range and step for each parameter.
        # Here, we're assuming a simple case where we just take a step around the best value.
        param_grid = {}
        for param, value in best_params.items():
            if isinstance(value, int):
                param_grid[param] = range(max(1, value - 1), value + 2)
            elif isinstance(value, float):
                param_grid[param] = np.linspace(max(0, value - value * 0.1), value + value * 0.1, num=3)
            else:
                param_grid[param] = [value]  # For categorical parameters
        return param_grid

    def fit(self, X, y):
        # Perform the initial random search
        random_search = ParameterSampler(self.param_distributions, n_iter=self.n_iter, random_state=self.random_state)
        best_score = -np.inf
        best_params = None
        for params in random_search:
            self.model.set_params(**params)
            scores = cross_val_score(self.model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        # Refine the search around the best parameters from the random search
        param_grid = self.refine_params(best_params)
        self.grid_search = GridSearchCV(self.model, param_grid, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs)
        self.grid_search.fit(X, y)

        # Set the best estimator
        self.best_estimator_ = self.grid_search.best_estimator_
        self.best_score_ = self.grid_search.best_score_
        self.best_params_ = self.grid_search.best_params_

    def predict(self, X):
        return self.best_estimator_.predict(X)
