from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from .model_base import AbstractModelTrainer

# Define default parameters for SVM
default_svm_param_grid = {
    'classifier__estimator__C': [0.1, 1, 10, 100],
    'classifier__estimator__max_iter': [5000],
    'classifier__estimator__dual': [False]
}

class SVMModel(AbstractModelTrainer):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = default_svm_param_grid
        super().__init__(param_grid)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', OneVsOneClassifier(estimator=LinearSVC(max_iter=2000)))
        ])