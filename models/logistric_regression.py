from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler

from .model_base import AbstractModelTrainer
from sklearn.pipeline import make_pipeline, Pipeline

# Define default parameters for Logistic Regression
default_logistic_param_grid = {
    'classifier__estimator__penalty': ['l2'],
    'classifier__estimator__max_iter': [5000],
    'classifier__estimator__C': [0.1, 1, 10, 100],
    'classifier__estimator__solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],
}

class LogisticRegressionModel(AbstractModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = default_logistic_param_grid
        super().__init__(param_grid)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', OneVsOneClassifier(estimator=LogisticRegression(max_iter=2000)))
        ])
