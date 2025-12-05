from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
import numpy as np

# KNN 

X_knn = np.array([
    [2, 8],
    [3, 7],
    [4, 6],
    [5, 7],
    [6, 8],
    [7, 6],
    [8, 5],
    [9, 7],
    [10, 8],
    [11, 6]
])

y_knn = np.array(["Failed", "Failed", "Failed", "Passed", "Passed",
                  "Passed", "Failed", "Passed", "Passed", "Passed"])

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_knn, y_knn)

new_student = np.array([[6, 7]])
knn_prediction = knn_model.predict(new_student)
print("KNN Prediction for student:", knn_prediction[0])


# Naive Bayes 

outlook_map = {"Sunny": 0, "Overcast": 1, "Rainy": 2}
temp_map = {"Hot": 0, "Mild": 1, "Cool": 2}
humidity_map = {"High": 0, "Normal": 1}
wind_map = {False: 0, True: 1}
play_map = {"No": 0, "Yes": 1}

X_nb = np.array([
    [outlook_map["Sunny"],     temp_map["Hot"],  humidity_map["High"],  wind_map[False]],
    [outlook_map["Sunny"],     temp_map["Hot"],  humidity_map["High"],  wind_map[True]],
    [outlook_map["Overcast"],  temp_map["Hot"],  humidity_map["High"],  wind_map[False]],
    [outlook_map["Rainy"],     temp_map["Mild"], humidity_map["High"],  wind_map[False]],
    [outlook_map["Rainy"],     temp_map["Cool"], humidity_map["Normal"],wind_map[False]],
    [outlook_map["Rainy"],     temp_map["Cool"], humidity_map["Normal"],wind_map[True]],
    [outlook_map["Overcast"],  temp_map["Cool"], humidity_map["Normal"],wind_map[True]],
    [outlook_map["Sunny"],     temp_map["Mild"], humidity_map["High"],  wind_map[False]],
    [outlook_map["Sunny"],     temp_map["Cool"], humidity_map["Normal"],wind_map[False]],
    [outlook_map["Rainy"],     temp_map["Mild"], humidity_map["Normal"],wind_map[False]],
    [outlook_map["Sunny"],     temp_map["Mild"], humidity_map["Normal"],wind_map[True]],
    [outlook_map["Overcast"],  temp_map["Mild"], humidity_map["High"],  wind_map[True]],
    [outlook_map["Overcast"],  temp_map["Hot"],  humidity_map["Normal"],wind_map[False]],
    [outlook_map["Rainy"],     temp_map["Mild"], humidity_map["High"],  wind_map[True]],
])

y_nb = np.array([
    play_map["No"], play_map["No"], play_map["Yes"], play_map["Yes"], play_map["Yes"],
    play_map["No"], play_map["Yes"], play_map["No"], play_map["Yes"], play_map["Yes"],
    play_map["Yes"], play_map["Yes"], play_map["Yes"], play_map["No"]
])

nb_model = CategoricalNB()
nb_model.fit(X_nb, y_nb)

new_day = np.array([[outlook_map["Sunny"], temp_map["Mild"], humidity_map["High"], wind_map[False]]])
nb_prediction = nb_model.predict(new_day)

print("Naive Bayes Play?:", "Yes" if nb_prediction[0] == 1 else "No")
