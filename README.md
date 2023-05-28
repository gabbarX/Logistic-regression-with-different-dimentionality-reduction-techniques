# Logistic-regression-with-different-dimentionality-reduction-techniques


This code is an implementation of logistic regression with feature dimensionality reduction techniques such as Principal Component Analysis (PCA) and Fisher's Linear Discriminant Analysis (FDA). It aims to classify heart disease based on given features.

## Dependencies
- pandas: A library used for data manipulation and analysis.
- numpy: A library used for numerical computations.
- scikit-learn: A machine learning library that provides tools for data preprocessing, model selection, and evaluation.

## Functions
### accuracy(y_true, y_pred)
This function calculates the accuracy of the predicted values y_pred compared to the true values y_true. It returns the accuracy as a floating-point number.

### sigmoid(x)
This function applies the sigmoid function to the input x and returns the result.

### LogisticRegression(X, y, alpha, iterations, X_test)
This function implements logistic regression using gradient descent optimization. It takes the following parameters:

X: The independent variable matrix for training.
y: The dependent variable vector for training.
alpha: The learning rate or step size for gradient descent.
iterations: The number of iterations for gradient descent.
X_test: The independent variable matrix for testing.
It returns an array of binary predictions based on the trained logistic regression model.

### FDA(X, y, n)
This function performs Fisher's Linear Discriminant Analysis on the input data X and corresponding labels y. It reduces the dimensionality of the data to n dimensions. It returns the transformed data matrix.
