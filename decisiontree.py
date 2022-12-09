import numpy as np
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

DT_GRID = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [4, 6, 8, 10, 12, 14, 16, 18, 20],
    'min_samples_leaf': [1, 5, 10, 20, 50, 100],
    'min_samples_split': [2, 5, 10, 15, 20, 25],
    'max_features': [None, 'sqrt', 'log2']
}

class DecisionTree:
    def __init__(self):
        self.tree_clf = DecisionTreeClassifier()
        self.best_model = self.tree_clf

    def set_params(self, params):
        self.best_model.set_params(**params)

    def train(self, features, labels):
        start = time.time()
        self.best_model.fit(features, labels)
        end = time.time()
        return end - start

    # optimize the model with gridsearch to prevent overfitting
    def gridsearch(self, features, values):
        start = time.time()
        param_grid = {
            'criterion':['gini', 'entropy', 'log_loss'],
            'max_depth': np.arange(4, 21),
            'min_samples_leaf': [1, 5, 10, 20, 50, 100],
        }
        self.grid_search = GridSearchCV(self.tree_clf, param_grid, cv=10, scoring="accuracy", return_train_score=True)
        self.grid_search.fit(features, values)
        end = time.time()
        self.best_model = self.grid_search.best_estimator_
        return end - start

    # optimize the model with randomsearch to prevent overfitting
    def randomsearch(self, features, labels, param_grid):
        start = time.time()
        self.random_search = RandomizedSearchCV(estimator=self.tree_clf,
                                                param_distributions=param_grid,
                                                n_iter=100,
                                                scoring=['accuracy', 'f1'],
                                                refit='f1',
                                                cv=10,
                                                verbose=1)
        self.random_search.fit(features, labels)
        end = time.time()
        self.best_model = self.random_search.best_estimator_
        return end - start

    # Calculate accuracy and f1 score predicting provided data
    def get_metrics(self, features, labels):
        prediction = self.best_model.predict(features)

        accuracy = accuracy_score(labels, prediction)
        f1 = f1_score(labels, prediction)
        precision = precision_score(labels, prediction)
        recall = recall_score(labels, prediction)

        # accuracy_pct = "{:.0%}".format(accuracy)
        # f1_pct = "{:.0%}".format(f1)
        # precision_pct = "{:.0%}".format(precision)
        # recall_pct = "{:.0%}".format(recall)

        return accuracy, f1, precision, recall
