from sklearn.naive_bayes import GaussianNB
from .model_base import AbstractModelTrainer

class GaussianNBModel(AbstractModelTrainer):
    def __init__(self):
        super().__init__(param_grid={})  # GaussianNB doesn't have significant hyperparameters for grid search
        self.model = GaussianNB()