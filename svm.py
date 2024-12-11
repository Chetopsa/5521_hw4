import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
# from data import Data

#  variable for toggling between the data sets
CLOUD = True
def MakeConfusionMatrix(z, y, labels=None):
    z = np.array(z).reshape(-1)
    y = np.array(y).reshape(-1)
    unique_labels = np.unique(np.concatenate([y, z]))
    if labels is None:
        labels = unique_labels
    else:
        labels = np.array(labels)
    labels = sorted(labels)
    nlabels = len(labels)
    confusion = np.array([
        [np.sum((z == labels[i]) & (y == labels[j])) for j in range(nlabels)]
        for i in range(nlabels)
    ])
    index = [f'predicted_{labels[i]}' for i in range(nlabels)]
    columns = [f'truly_{labels[i]}' for i in range(nlabels)]
    confusion_matrix = pd.DataFrame(confusion, index=index, columns=columns)
    return confusion_matrix

#  load data
if not CLOUD:
    data = np.genfromtxt("ftrain56.csv", delimiter=",")
    Xtrain = data[:, 1:]
    Ytrain = data[:, 0]
    test_data = np.genfromtxt("ftest56.csv", delimiter=",")
    Xtest = test_data[:, 1:]
    Ytest = test_data[:, 0]
else:
    NUM_NONCLOUD = 400
    train_set_param = int(NUM_NONCLOUD*.70)
    test_set_param1 = train_set_param + 1
    test_set_param2 = NUM_NONCLOUD

    cloud_data = np.genfromtxt("cloud_features.csv", delimiter=",")
    non_cloud_data = np.genfromtxt("noncloud_features.csv", delimiter=",")
    Xtrain = np.concatenate((cloud_data[:69, 1:], non_cloud_data[:train_set_param, 1:]))
    Ytrain = np.concatenate((cloud_data[:69, 0], non_cloud_data[:train_set_param, 0]))
    Xtest = np.concatenate((cloud_data[70:, 1:], non_cloud_data[test_set_param1:test_set_param2, 1:]))
    Ytest = np.concatenate((cloud_data[70:, 0], non_cloud_data[test_set_param1:test_set_param2, 0]))
print("done reading data")

# print(f"Xtrain: {Xtrain}\nYtrain: {Ytrain}\nXtest: {Xtest}\nYtest: {Ytest}")

c_values = [0.1, 1, 10, 100]
best_c = None
best_accuracy = 0
for c in c_values:
    
    svc = svm.LinearSVC(C=c).fit(Xtrain, Ytrain)
    Ypred_train = svc.predict(Xtrain)
    Ypred_test = svc.predict(Xtest)
    

    train_accuracy = np.mean(Ypred_train == Ytrain)
    test_accuracy = np.mean(Ypred_test == Ytest)
    print(f"C={c}: Train Accuracy={train_accuracy:.4f}, Test Accuracy={test_accuracy:.4f}")
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_c = c

svc = svm.SVC(C=best_c,kernel='linear').fit(Xtrain, Ytrain)
Ypred = svc.predict(Xtest)
# print(Ypred)
# Display the test data using matplotlib

# Create a confusion matrix

# print(Ytest)
# labels = [Ytest[1], Ytest[0]]
confusion_matrix = MakeConfusionMatrix(Ypred, Ytest )
print(confusion_matrix)
'''
Same thing but with kenrel svm
'''
# knernel svm
best_c_rbf = None
best_accuracy_rbf = 0
for c in c_values:
    svc = svm.SVC(C=c, kernel='rbf').fit(Xtrain, Ytrain)
    svc_rbf = svm.SVC(C=c, kernel='rbf').fit(Xtrain, Ytrain)
    Ypred_train_rbf = svc_rbf.predict(Xtrain)
    Ypred_test_rbf = svc_rbf.predict(Xtest)
    
    train_accuracy_rbf = np.mean(Ypred_train_rbf == Ytrain)
    test_accuracy_rbf = np.mean(Ypred_test_rbf == Ytest)
    
    print(f"C={c}: Train Accuracy={train_accuracy_rbf:.4f}, Test Accuracy={test_accuracy_rbf:.4f}")
    
    if test_accuracy_rbf > best_accuracy_rbf:
        best_accuracy_rbf = test_accuracy_rbf
        best_c_rbf = c

svc_rbf = svm.SVC(C=best_c_rbf,kernel='rbf')  # Using RBF kernel
svc_rbf.fit(Xtrain, Ytrain)  # Train the model

# Predict the test set
Ypred_rbf = svc_rbf.predict(Xtest)
# print("Predictions using RBF kernel:", Ypred_rbf)

# print confusion matrix
confusion_matrix_rbf = MakeConfusionMatrix(Ypred_rbf, Ytest)
print("\nBest RBF Kernel Model Confusion Matrix:")
print(confusion_matrix_rbf)

# print final accuracies
print("\nFinal Accuracies:")
print(f"Linear Kernel: Train Accuracy={np.mean(svc.predict(Xtrain) == Ytrain):.4f}, Test Accuracy={np.mean(Ypred == Ytest):.4f}")
print(f"RBF Kernel: Train Accuracy={np.mean(svc_rbf.predict(Xtrain) == Ytrain):.4f}, Test Accuracy={np.mean(Ypred_rbf == Ytest):.4f}")





'''confusion matrix using matplot lib'''
# fig, ax = plt.subplots(figsize=(15, 7))
# disp = ConfusionMatrixDisplay(confusion_matrix(Ytest, Ypred_rbf), display_labels=[0, 1])
# disp.plot(ax=ax)
# plt.xticks(rotation = 90)
# plt.show()


