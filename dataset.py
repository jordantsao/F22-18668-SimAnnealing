import pandas as pd
from scipy.io import arff

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class Dataset:

    # Initialization
    def __init__(self, filename):
        # The file is in the Attribute-Relation File Format. So, after importing it, we have to transform its data into a pandas DataFrame.
        self.name = filename[:-5]
        self.data = arff.loadarff(filename)
        self.df = pd.DataFrame(self.data[0])
        self.split_and_fill()
        self.normalize()
        self.split_train_test()

    # Split dataframe into X and Y
    def split_and_fill(self):
        # Splice labels and encode 'True' to 1, 'False' to 0
        Y_data = self.df.iloc[:, -1].values
        encoder = preprocessing.LabelEncoder()
        self.Y = encoder.fit_transform(Y_data)

        # Splice features and populate unfilled columns with median
        X_copy = self.df.iloc[:, :-1].copy()
        imputer = SimpleImputer(strategy="median")
        self.X = imputer.fit_transform(X_copy)

    def normalize(self):
        # scaler = preprocessing.StandardScaler().fit(self.X)
        # self.X = scaler.transform(self.X)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.X = min_max_scaler.fit_transform(self.X)

    # Split dataset into training and test sets
    def split_train_test(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.20)
