import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


calories_df = pd.read_csv("calories.csv")   
exercise_df = pd.read_csv("exercise.csv")    

print("Calories file head:")
print(calories_df.head())
print("\nExercise file head:")
print(exercise_df.head())

df = exercise_df.merge(calories_df, on="User_ID")

print("\nMerged data shape:", df.shape)
print("Columns:", df.columns.tolist())

target_col = "Calories"  
y = df[target_col]

X = df.drop(columns=["User_ID", target_col])

print("\nFeatures used:", X.columns.tolist())

numeric_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
categorical_features = ["Gender"]  
missing_cols = [c for c in numeric_features + categorical_features if c not in X.columns]
if missing_cols:
    print("\n WARNING: these columns not found in X:", missing_cols)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = LinearRegression()

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel performance on test set:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R^2  : {r2:.4f}")


sample = pd.DataFrame([{
    "Gender": "male",    
    "Age": 25,
    "Height": 175,
    "Weight": 70,
    "Duration": 60,       
    "Heart_Rate": 130,    
    "Body_Temp": 37.0     
}])

predicted_calories = clf.predict(sample)[0]

print("\nSample input:")
print(sample)
print(f"\nPredicted Calories Burned: {predicted_calories:.2f}")
