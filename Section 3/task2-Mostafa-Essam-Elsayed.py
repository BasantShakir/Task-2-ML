#KNN
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
data = {
    "StudyHours": [2,3,4,5,6,7,8,9,10,11],
    "SleepHours": [8,7,6,7,8,6,5,7,8,6],
    "Result": ["Failed","Failed","Failed","Passed","Passed","Passed","Failed","Passed","Passed","Passed"]
}

df = pd.DataFrame(data)

X = df[["StudyHours","SleepHours"]]
y = df["Result"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

new_study = 6
new_sleep = 7

prediction = model.predict([[new_study, new_sleep]])
print("Prediction for new student =", prediction[0])

#--------------------------------------------------------
#Naive Bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = {
    "Outlook": ["Sunny","Sunny","Overcast","Rainy","Rainy","Rainy","Overcast","Sunny","Sunny","Rainy","Sunny","Overcast","Overcast","Rainy"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Windy": [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
    "Play": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
}

df = pd.DataFrame(data)

label_outlook = LabelEncoder()
label_temp = LabelEncoder()
label_hum = LabelEncoder()
label_wind = LabelEncoder()
label_play = LabelEncoder()

df["Outlook"] = label_outlook.fit_transform(df["Outlook"])
df["Temperature"] = label_temp.fit_transform(df["Temperature"])
df["Humidity"] = label_hum.fit_transform(df["Humidity"])
df["Windy"] = label_wind.fit_transform(df["Windy"])
df["Play"] = label_play.fit_transform(df["Play"])

X = df[["Outlook","Temperature","Humidity","Windy"]]
y = df["Play"]

model = GaussianNB()
model.fit(X, y)

new_day = [
    label_outlook.transform(["Sunny"])[0],
    label_temp.transform(["Mild"])[0],
    label_hum.transform(["Normal"])[0],
    label_wind.transform([False])[0]
]

prediction = model.predict([new_day])
result = label_play.inverse_transform(prediction)[0]

print("Should we play today? :", result)
