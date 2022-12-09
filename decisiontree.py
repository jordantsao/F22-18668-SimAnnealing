import numpy as np
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score

class DecisionTree:
    def __init__(self):
        self.tree_clf = DecisionTreeClassifier()
        self.best_model = None

    # Train the model with gridsearch to prevent overfitting
    def train_with_gridsearch(self, features, values):
        depths = np.arange(10, 21)
        num_leafs = [1, 5, 10, 20, 50, 100]

        param_grid = {'criterion':['gini', 'entropy'], 'max_depth': depths, 'min_samples_leaf': num_leafs}

        self.grid_search = GridSearchCV(self.tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True)
        self.grid_search.fit(features, values)
        self.best_model = self.grid_search.best_estimator_
        self.best_model.fit(features, values)

    # Calculate accuracy and f1 score predicting provided data
    def get_metrics(self, features, values):
        accuracy = accuracy_score(values, self.best_model.predict(features))
        f1 = f1_score(values, self.best_model.predict(features))
        accuracy_pct = "{:.0%}".format(accuracy)
        f1_pct = "{:.0%}".format(f1)
        return accuracy_pct, f1_pct
