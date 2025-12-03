
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import cross_val_score

weather = pd.DataFrame([
    ['Sunny','Hot','High', False, 'No'],
    ['Sunny','Hot','High', True,  'No'],
    ['Overcast','Hot','High', False, 'Yes'],
    ['Rainy','Mild','High', False, 'Yes'],
    ['Rainy','Cool','Normal', False, 'Yes'],
    ['Rainy','Cool','Normal', True,  'No'],
    ['Overcast','Cool','Normal', True,  'Yes'],
    ['Sunny','Mild','High', False, 'No'],
    ['Sunny','Cool','Normal', False, 'Yes'],
    ['Rainy','Mild','Normal', False, 'Yes'],
    ['Sunny','Mild','Normal', True,  'Yes'],
    ['Overcast','Mild','High', True,  'Yes'],
    ['Overcast','Hot','Normal', False, 'Yes'],
    ['Rainy','Mild','High', True,  'No']
], columns=['Outlook','Temperature','Humidity','Windy','Play'])

X = weather[['Outlook','Temperature','Humidity','Windy']]
y = weather['Play'].map({'No':0, 'Yes':1}).values

enc = OrdinalEncoder()
X_enc = enc.fit_transform(X)

nb = CategoricalNB()
nb.fit(X_enc, y)

cv_scores = cross_val_score(CategoricalNB(), X_enc, y, cv=5)

print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

sample = pd.DataFrame([
    ['Sunny','Cool','High', False],
    ['Rainy','Mild','Normal', True]
], columns=['Outlook','Temperature','Humidity','Windy'])

sample_enc = enc.transform(sample)
preds = nb.predict(sample_enc)

for i,p in enumerate(preds):
    print(f"{list(sample.iloc[i])} -> Play: {'Yes' if p==1 else 'No'}")
 