import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# Defining the function that checks the accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Defining the sigmod function
def sigmoid(x):
    fn = 1 / (1 + np.exp(-x))
    return fn


def LogisticRegression(X, y, alpha, iterations, X_test):
    # Extracting rows and feature number from the independent matrix
    rows, cols = X.shape

    # initialising a matrix of zeroes for storing the values of theta.
    theta = np.zeros(cols)

    # initialising the bias
    bias = 0

    # implementing gradient descent to get value of theta and bias
    for i in range(iterations):
        # compute linear combination
        linearComb = np.dot(X, theta) + bias
        # apply sigmoid function
        y_pred = sigmoid(linearComb)
        # compute gradients
        dw = (1 / rows) * np.dot(X.T, (y_pred - y))
        db = (1 / rows) * np.sum(y_pred - y)
        # update parameters
        theta = theta - alpha * dw
        bias = bias - alpha * db

    # compute linear combination for the final iteration
    linearComb = np.dot(X_test, theta) + bias
    y_pred = sigmoid(linearComb)

    # Convert the continuos values to binary values.
    res = []
    for i in y_pred:
        if i > 0.5:
            res.append(1)
        else:
            res.append(0)

    return np.array(res)


def FDA(X, y, n):
    # creating two classes of data
    # class 1 = data with y=0
    X1 = X[y == 0]
    # class 2 = data with y=1
    X2 = X[y == 1]

    # compute mean of each class
    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)

    # compute within class scatter matrix
    S1 = np.dot((X1 - mean1).T, (X1 - mean1))
    S2 = np.dot((X2 - mean2).T, (X2 - mean2))
    SW = S1 + S2

    # compute between class scatter matrix
    SB = np.dot((mean2 - mean1).reshape(-1, 1), (mean2 - mean1).reshape(1, -1))
    # print(SW)
    # print(SB)
    # A = SW^-1 * SB
    A = np.linalg.inv(SW).dot(SB)
    # Get eigenvalues and eigenvectors of SW^-1 * SB
    eigenvalues, eigenvectors = np.linalg.eig(A)
    # -> eigenvector v = [:,i] column vector, transpose for easier calculations

    # sort eigenvalues high to low
    eigenvectors = eigenvectors.T
    idxs = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idxs]
    # print(eigenvalues)
    eigenvectors = eigenvectors[idxs]
    # print(eigenvectors)

    # store first n eigenvectorss
    linear_discriminants = eigenvectors[0:n]
    # print(self.linear_discriminants)
    return np.dot(X, linear_discriminants.T)


df = pd.read_csv("Question_2\Heart.csv")

# print(df)
df["ChestPain"] = (
    df["ChestPain"]
    .map({"typical": 0, "asymptomatic": 1, "nonanginal": 2, "nontypical": 3})
    .astype(int)
)
df["Thal"].fillna("No Thal", inplace=True)
df["Ca"].fillna(-1, inplace=True)
df["Thal"] = (
    df["Thal"].map({"fixed": 0, "normal": 1, "reversable": 2, "No Thal": 3}).astype(int)
)
df["AHD"] = df["AHD"].map({"No": 0, "Yes": 1}).astype(int)
df.drop(
    df.columns[df.columns.str.contains("unnamed", case=False)], axis=1, inplace=True
)

learning_rate = 0.01
iterations = 100000
randomState = 42

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = np.asarray(X)
y = np.asarray(y)


pca = PCA(n_components=5)
X_PCA = pca.fit_transform(X, y)


X_FDA = FDA(X, y, 3)

X_PCA_FDA = FDA(X_PCA, y, 3)


# print("X ", X.shape)
# print("X PCA:", X_PCA.shape)
# print("X FDA:", X_FDA.shape)
# print("X PCA FDA:", X_PCA_FDA.shape)

# print("y:")
# print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=randomState
)


predictions = LogisticRegression(X_train, y_train, learning_rate, iterations, X_test)
accuracy = np.sum(y_test == predictions) / len(y_test)
print("Logistic Regression classification accuracy:", accuracy)


X_train, X_test, y_train, y_test = train_test_split(
    X_PCA, y, test_size=0.2, random_state=randomState
)


predictions = LogisticRegression(X_train, y_train, learning_rate, iterations, X_test)
accuracy = np.sum(y_test == predictions) / len(y_test)
print("Logistic Regression classification accuracy with PCA:", accuracy)


X_train, X_test, y_train, y_test = train_test_split(
    X_FDA, y, test_size=0.2, random_state=randomState
)


predictions = LogisticRegression(X_train, y_train, learning_rate, iterations, X_test)
accuracy = np.sum(y_test == predictions) / len(y_test)
print("Logistic Regression classification accuracy with FDA:", accuracy)


X_train, X_test, y_train, y_test = train_test_split(
    X_PCA_FDA, y, test_size=0.2, random_state=randomState
)


predictions = LogisticRegression(X_train, y_train, learning_rate, iterations, X_test)
accuracy = np.sum(y_test == predictions) / len(y_test)
print("Logistic Regression classification accuracy with PCA and FDA:", accuracy)
