from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
iris = load_iris()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    random_state=0
)

# Parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}

print("Parameter grid:")
print(param_grid)

# Grid Search
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

# Results
print("\nTest set score: {:.2f}".format(
    grid_search.score(X_test, y_test)
))

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(
    grid_search.best_score_
))

print("\nBest estimator:")
print(grid_search.best_estimator_)

# Convert cv results to DataFrame
results = pd.DataFrame(grid_search.cv_results_)

print("\nFirst 5 rows of cv_results_:")
print(results.head())

# Create score matrix for heatmap
scores = results['mean_test_score'].to_numpy().reshape(
    len(param_grid['C']),
    len(param_grid['gamma'])
)

print("\nScore matrix shape:", scores.shape)

# Plot heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    scores,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=param_grid['gamma'],
    yticklabels=param_grid['C']
)

plt.xlabel("gamma")
plt.ylabel("C")
plt.title("GridSearchCV Mean Cross-Validation Scores")

plt.tight_layout()
plt.show()
