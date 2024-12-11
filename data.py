import numpy as np

class Data:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.Xtrain, self.Ytrain = self.load_data(train_file)
        self.Xtest, self.Ytest = self.load_data(test_file)
    
    def load_data(self, file):
        data = np.genfromtxt(file, delimiter=",")
        X = data[:, 1:]
        Y = data[:, 0]
        return X, Y
    def __str__(self):
        return f"Xtrain: {self.Xtrain}\nYtrain: {self.Ytrain}\nXtest: {self.Xtest}\nYtest: {self.Ytest}"

data = Data("ftrain56.csv", "ftest56.csv")
print(data)

data = np.genfromtxt("ftrain56.csv", delimiter=",")
Xtrain = data[:, 1:]
Ytrain = data[:, 0]
test_data = np.genfromtxt("ftest56.csv", delimiter=",")
Xtest = test_data[:, 1:]
Ytest = test_data[:, 0]