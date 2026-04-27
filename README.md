# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Split the dataset into features (X = Hours Studied) and target (y = Marks Scored), then divide it into training (80%) and testing (20%) sets using train_test_split.
2.Model Training Fit a Linear Regression model on the training data. The model learns two parameters — the intercept (b0) and slope (b1) — by minimizing the sum of squared errors between actual and predicted values, forming the equation: Marks = b0 + b1 × Hours.
3.Model Evaluation Use the trained model to predict marks on the test set, then evaluate performance using Mean Squared Error (MSE) and R² Score to measure how well the regression line fits the data.
4.Prediction & Visualization Predict marks for a new input (e.g., 7.5 hours) using the learned equation, and plot the actual data points alongside the regression line to visually confirm the model's fit.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dhivakar B
RegisterNumber:  212225040075
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "Hours_Studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks_Scored":  [35, 40, 50, 55, 60, 65, 70, 80, 85, 95]
}
df = pd.DataFrame(data)

# Display dataset
print("Dataset:\n", df.head())
df

X = df[["Hours_Studied"]]   
y = df["Marks_Scored"]      

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Parameters:")
print("Intercept (b0):", model.intercept_)
print("Slope (b1):", model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(8,6))
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', linewidth=2, label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression: Predicting Marks")
plt.legend()
plt.grid(True)
plt.show()

hours = 7.5
predicted_marks = model.predict([[hours]])
print(f"\nPredicted marks for {hours} hours of study = {predicted_marks[0]:.2f}")
*/
```

## Output:
<img width="827" height="781" alt="image" src="https://github.com/user-attachments/assets/18a6c61a-dac4-466a-89f5-bb6af33bea83" />
<img width="842" height="814" alt="image" src="https://github.com/user-attachments/assets/83f648ac-97d4-4e90-aaa5-fbd2dae4fa8b" />
<img width="602" height="332" alt="image" src="https://github.com/user-attachments/assets/328520a8-b513-4d62-8b2d-880d4b81ee7a" />
<img width="1042" height="726" alt="image" src="https://github.com/user-attachments/assets/92e1d8ff-8121-410d-a11e-f17858cac4ea" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
