import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsOneClassifier

from .model_base import AbstractModelTrainer  # Import your AbstractModelTrainer class

class KerasNeuralNetworkModel(AbstractModelTrainer):
    def __init__(self):
        super().__init__(param_grid={})

    def train(self, x, y):
        self.model = KerasClassifier(model=lambda: create_model(x.shape[1]), epochs=10, batch_size=32, verbose=0)
        self.model = OneVsOneClassifier(self.model)
        self.model.fit(x, y)

    def predict(self, x):
        # Make predictions
        y_pred = self.model.predict(x)
        return y_pred


def create_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Use 'softmax' for multi-class classification, 'sigmoid' for 1-class
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model