from dataset import Dataset
from decisiontree import DecisionTree
from svc import SVCClassifier
from optimizer import SVC_GRID, Simulation

import warnings
warnings.filterwarnings("ignore")

def main():
    fe = Dataset("feature-envy.arff")
    dc = Dataset("data-class.arff")
    gc = Dataset("god-class.arff")
    lm = Dataset("long-method.arff")
    # print(fe.df[fe.df.is_feature_envy == b'true'])
    # print(len(fe.X_train[0]))

    svc = SVCClassifier()

    grid = SVC_GRID
    sim = Simulation(grid, svc, fe)
    sim.begin()
    sim.test()

    # dt = DecisionTree()
    # dt.train_with_gridsearch(fe.X_train, fe.Y_train)
    # print(dt.get_metrics(fe.X_test, fe.Y_test))

if __name__ == "__main__":
    main()