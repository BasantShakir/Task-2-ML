


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset
knn_data = pd.DataFrame({
    'StudyHours': [2,3,4,5,6,7,8,9,10,11],
    'SleepHours': [8,7,6,7,8,6,5,7,8,6],
    'Result': ['Failed','Failed','Failed','Passed','Passed','Passed','Failed','Passed','Passed','Passed']
})

knn_data['ResultNum'] = knn_data['Result'].map({'Failed':0, 'Passed':1})

X = knn_data[['StudyHours','SleepHours']].values
y = knn_data['ResultNum'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_students = np.array([[5,6], [4,8], [9,7]])
new_scaled = scaler.transform(new_students)
preds = knn.predict(new_scaled)

print("\nNew Predictions:")
for i, p in enumerate(preds):
    print(f"Student {i+1} -> {'Passed' if p==1 else 'Failed'}")

