from dataset import Dataset
from decisiontree import DecisionTree
from svc import SVCClassifier, SVC_GRID
from decisiontree import DecisionTree, DT_GRID
from annealer import Annealer

import warnings
warnings.filterwarnings("ignore")

def main():
    # initialize datasets
    fe = Dataset("feature-envy.arff")
    # print(fe.df[fe.df.is_feature_envy == b'true'])
    # print(len(fe.X_train[0]))

    # initialize classifier
    model = SVCClassifier()
    grid = SVC_GRID
    # model = DecisionTree()
    # grid = DT_GRID

    # initialize annealer
    sim = Annealer(grid, model, fe)
    sim.begin_simulation()
    sim.test()

if __name__ == "__main__":
    main()