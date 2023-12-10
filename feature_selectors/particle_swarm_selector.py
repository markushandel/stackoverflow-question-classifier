import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from .base_selector import AbstractFeatureSelector

class PSOFeatureSelector(AbstractFeatureSelector):
    def __init__(self, n_particles, n_iterations, classifier=RandomForestClassifier(), inertia_weight=0.6, cognitive_weight=1, social_weight=1):
        super().__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.classifier = classifier
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.global_best_position = None
        self.global_best_score = float('-inf')

    def fit(self, X, y):
        n_features = X.shape[1]
        particle_positions = np.random.rand(self.n_particles, n_features) > 0.5  # Random initialization
        particle_velocities = np.random.rand(self.n_particles, n_features)
        local_best_positions = particle_positions.copy()
        local_best_scores = np.array([float('-inf')] * self.n_particles)

        for j in range(self.n_iterations):

            self.inertia_weight = max(0.4, self.inertia_weight - 0.01)

            for i in range(self.n_particles):
                # Convert boolean array to indices
                selected_features = np.where(particle_positions[i])[0]
                if len(selected_features) == 0:
                    continue

                # Evaluate particle
                score = cross_val_score(self.classifier, X[:, selected_features], y, cv=5, n_jobs=-1).mean()
                if score > local_best_scores[i]:
                    local_best_scores[i] = score
                    local_best_positions[i] = particle_positions[i].copy()
                    if score > self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = particle_positions[i].copy()

            # Update velocity and position
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()
                particle_velocities[i] = (self.inertia_weight * particle_velocities[i] +
                          self.cognitive_weight * r1 * np.logical_xor(local_best_positions[i], particle_positions[i]) +
                          self.social_weight * r2 * np.logical_xor(self.global_best_position, particle_positions[i]))

                # Sigmoid function to map velocity to probability
                sigmoid = 1 / (1 + np.exp(-particle_velocities[i]))
                particle_positions[i] = np.random.rand(n_features) < sigmoid

        self.selected_features = np.where(self.global_best_position)[0]

    def transform(self, X):
        if self.selected_features is None:
            raise RuntimeError("The feature selector needs to be fitted first.")
        return X[:, self.selected_features]