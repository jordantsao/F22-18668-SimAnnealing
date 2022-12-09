import numpy as np
import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

    # Train the model with gridsearch to prevent overfitting
    def gridsearch(self, features, values):
        start = time.time()
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': ['scale', 'auto', 1000, 100, 10, 1, 0.1, 0.01, 0.001],'kernel': ['linear', 'rbf', 'sigmoid']}
        self.grid_search = GridSearchCV(self.svc_clf, param_grid, cv=10, scoring="f1", return_train_score=True)
        self.grid_search.fit(features, values)
        end = time.time()
        self.best_model = self.grid_search.best_estimator_
        return end - start

    # Calculate accuracy and f1 score predicting provided data
    def get_metrics(self, features, values):
        prediction = self.best_model.predict(features)

        accuracy = accuracy_score(values, prediction)
        f1 = f1_score(values, prediction)
        precision = precision_score(values, prediction)
        recall = recall_score(values, prediction)

        # accuracy_pct = "{:.0%}".format(accuracy)
        # f1_pct = "{:.0%}".format(f1)
        # precision_pct = "{:.0%}".format(precision)
        # recall_pct = "{:.0%}".format(recall)

        return accuracy, f1, precision, recall