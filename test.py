from dataset import Dataset
from svc import SVCClassifier

import warnings
warnings.filterwarnings("ignore")

fe = Dataset("feature-envy.arff")
svc = SVCClassifier()
time = svc.gridsearch(fe.X_train, fe.Y_train)
print (f'Duration: {time}')
svc.train(fe.X_train, fe.Y_train)
print(svc.get_metrics(fe.X_test, fe.Y_test))
# svc.set_params({
#     'kernel': 'sigmoid',
#     'C': 100,
#     'gamma': 0.001,
#     'degree': 2,
#     'tol': 0.1,
#     'max_iter': 100
# })

TRIALS = 100
avg_acc, avg_f1, avg_prec, avg_rec = 0, 0, 0, 0
for i in range(TRIALS):
    fe.split_train_test()
    svc.train(fe.X_train, fe.Y_train)
    acc, f1, prec, rec = svc.get_metrics(fe.X_test, fe.Y_test)
    avg_acc += acc
    avg_f1 += f1
    avg_prec += prec
    avg_rec += rec
avg_acc /= TRIALS
avg_f1 /= TRIALS
avg_prec /= TRIALS
avg_rec /= TRIALS
print(f'Accuracy: {avg_acc}, F1: {avg_f1}, Precision: {avg_prec}, Recall: {avg_rec}')
