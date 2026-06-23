from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

iris = load_iris()

logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

scores = cross_val_score(logreg, iris.data, iris.target, cv=3)
print("3-fold scores:", scores)
print("Average:", scores.mean())

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("5-fold scores:", scores)
print("Average:", scores.mean())
