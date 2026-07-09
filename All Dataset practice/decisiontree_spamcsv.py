import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("D:\Mechine Learning 4063\All Dataset practice\spambase.csv")

feat_labels = df.columns[:-1]

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)

# Train model
clf.fit(X_train, y_train)

feature_importances = clf.feature_importances_

print("Feature Importance:\n")

for feature, importance in zip(feat_labels, feature_importances):
    print(f"{feature}: {importance:.4f}")

print("\nFeatures Ranked:\n")

indices = np.argsort(feature_importances)[::-1]

for i in range(len(feat_labels)):
    print("%2d) %-30s %f" % (
        i + 1,
        feat_labels[indices[i]],
        feature_importances[indices[i]]
    ))
