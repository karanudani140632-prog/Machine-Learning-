import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Generate a sample dataset
data = {
    'X1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Y':  [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[['X1', 'X2']]
Y = df['Y']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluation
mae = metrics.mean_absolute_error(Y_test, Y_pred)
mse = metrics.mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
