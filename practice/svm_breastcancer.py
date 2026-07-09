from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Load dataset
b_cancer = load_breast_cancer()

X = b_cancer.data
y = b_cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Create SVM model
svc_model = SVC(kernel='rbf', C=1)

# Train model
svc_model.fit(X_train, y_train)

# Predict
y_pred = svc_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy of our model is equal to:", round(accuracy, 2), "%")
