import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Assuming student_scores.csv is present in the repository
filename = 'student_scores.csv'

try:
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    X = data[:, 0].reshape(-1, 1)  # Hours Studied
    y = data[:, 1]  # Marks Scored

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a Linear Regression model on the training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict marks for the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE) and R² Score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}, R² Score: {r2}')

    # Plot actual data points and regression line
    plt.scatter(X_test, y_test, color='blue', label='Actual Data')
    plt.plot(X_test, y_pred, color='red', label='Regression Line')
    plt.xlabel('Hours Studied')
    plt.ylabel('Marks Scored')
    plt.title('Simple Linear Regression for Predicting Marks Scored')
    plt.legend()
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found.")