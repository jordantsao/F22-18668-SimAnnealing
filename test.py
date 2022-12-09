from dataset import Dataset
from svc import SVCClassifier, SVC_GRID
from decisiontree import DecisionTree, DT_GRID

import warnings
warnings.filterwarnings("ignore")

fe = Dataset("feature-envy.arff")
# model = SVCClassifier()
model = DecisionTree()
grid  = DT_GRID

time = model.gridsearch(fe.X_train, fe.Y_train)
# time = model.randomsearch(fe.X_train, fe.Y_train, grid)

print (f'Duration: {time}')
model.train(fe.X_train, fe.Y_train)
print(model.get_metrics(fe.X_test, fe.Y_test))

TRIALS = 1000
avg_accuracy, avg_f1, avg_precision, avg_recall = 0, 0, 0, 0
for i in range(TRIALS):
    fe.split_train_test()
    model.train(fe.X_train, fe.Y_train)
    a, f, p, r = model.get_metrics(fe.X_test, fe.Y_test)
    avg_accuracy += a
    avg_f1 += f
    avg_precision += p
    avg_recall += r
avg_accuracy /= TRIALS
avg_f1 /= TRIALS
avg_precision /= TRIALS
avg_recall /= TRIALS
print(f'Accuracy: {avg_accuracy}, F1: {avg_f1}, Precision: {avg_precision}, Recall: {avg_recall}')
