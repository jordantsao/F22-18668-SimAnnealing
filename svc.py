import numpy as np
import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SVC_GRID = {
    'kernel': ['linear', 'rbf', 'sigmoid'],                         # kernel type
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],                      # regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100, 1000], # kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'degree': [1, 2, 3, 4],                                         # for polynomial kernel
    'tol': [0.001, 0.01, 0.1, 1],                                   # for stopping criterion
    'max_iter': [10, 50, 100, 500, 1000, -1]                        # hard limit on iterations
}

class SVCClassifier:
    def __init__(self):
        self.svc_clf = SVC()
        self.best_model = self.svc_clf
    
    def set_params(self, params):
        self.best_model.set_params(**params)

    def train(self, features, labels):
        start = time.time()
        self.best_model.fit(features, labels)
        end = time.time()
        return end - start

    # optimize the model with gridsearch to prevent overfitting
    def gridsearch(self, features, labels):
        start = time.time()
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 1000, 100, 10, 1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'rbf', 'sigmoid']
        }
        self.grid_search = GridSearchCV(self.svc_clf, param_grid, cv=10, scoring=['accuracy', 'f1'], refit='f1', return_train_score=True)
        self.grid_search.fit(features, labels)
        end = time.time()
        self.best_model = self.grid_search.best_estimator_
        return end - start
    
    # optimize the model with randomsearch to prevent overfitting
    def randomsearch(self, features, labels, param_grid):
        start = time.time()
        self.random_search = RandomizedSearchCV(estimator=self.svc_clf,
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