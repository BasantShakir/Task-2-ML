# task2.py
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB

# ----- KNN dataset -----
df_knn = pd.DataFrame({
    "Student": [f"Student {i}" for i in range(1,11)],
    "StudyHours": [2,3,4,5,6,7,8,9,10,11],
    "SleepHours": [8,7,6,7,8,6,5,7,8,6],
    "Result": ["Failed","Failed","Failed","Passed","Passed","Passed","Failed","Passed","Passed","Passed"]
})

# Encode target
le_knn = LabelEncoder()
y_knn = le_knn.fit_transform(df_knn["Result"])  # Passed/Failed -> integers
X_knn = df_knn[["StudyHours","SleepHours"]].values

# Train KNN
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_knn, y_knn)

# New students to predict (تعديل هنا لأي عينات جديدة)
new_students = np.array([[5,6], [4.5,6], [8,6]])  # كل صف: [StudyHours, SleepHours]
preds_knn = knn.predict(new_students)
probs_knn = knn.predict_proba(new_students)
preds_knn_labels = le_knn.inverse_transform(preds_knn)

print("=== KNN Predictions ===")
for i, s in enumerate(new_students):
    print(f"New student {i+1} (Study={s[0]}, Sleep={s[1]}): Predicted -> {preds_knn_labels[i]}, Probabilities -> Failed={probs_knn[i][0]:.3f}, Passed={probs_knn[i][1]:.3f}")

# ----- Naive Bayes dataset -----
df_nb = pd.DataFrame({
    "Outlook":["Sunny","Sunny","Overcast","Rainy","Rainy","Rainy","Overcast","Sunny","Sunny","Rainy","Sunny","Overcast","Overcast","Rainy"],
    "Temperature":["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity":["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Windy":[False,True,False,False,False,True,True,False,False,False,True,True,False,True],
    "Play":["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
})

# Encode categorical features to integers (CategoricalNB)
encoders = {}
X_nb = pd.DataFrame()
for col in ['Outlook','Temperature','Humidity','Windy']:
    le = LabelEncoder()
    # convert booleans to strings before encoding to keep consistency
    X_nb[col] = le.fit_transform(df_nb[col].astype(str))
    encoders[col] = le

y_nb = LabelEncoder().fit_transform(df_nb["Play"])  # Yes/No -> 1/0

# Train Categorical Naive Bayes
nb = CategoricalNB()
nb.fit(X_nb, y_nb)

# New day to predict (تغيير هنا لأي سيناريو جديد)
new_day = {"Outlook":"Sunny", "Temperature":"Cool", "Humidity":"High", "Windy":False}
new_day_encoded = [
    encoders['Outlook'].transform([new_day['Outlook']])[0],
    encoders['Temperature'].transform([new_day['Temperature']])[0],
    encoders['Humidity'].transform([new_day['Humidity']])[0],
    encoders['Windy'].transform([str(new_day['Windy'])])[0]
]

pred_nb = nb.predict([new_day_encoded])[0]
proba_nb = nb.predict_proba([new_day_encoded])[0]
label_nb = "Yes" if pred_nb == 1 else "No"

print("\n=== Naive Bayes Prediction ===")
print("New day features:", new_day)
print(f"Predicted Play -> {label_nb}  (Probabilities: No={proba_nb[0]:.3f}, Yes={proba_nb[1]:.3f})")
