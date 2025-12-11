
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import inspect

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------ Config ------------------------
DEFAULT_CSV = "calories.csv"
RANDOM_STATE = 42
MODEL_FILENAME = "calories_ransac_model.joblib"
# --------------------------------------------------------

def find_col_by_keywords(cols, keywords):
    lows = [c.lower() for c in cols]
    for kw in keywords:
        for orig, low in zip(cols, lows):
            if kw in low:
                return orig
    return None

def locate_csv(path_hint):
    if path_hint and os.path.exists(path_hint):
        return path_hint
    for name in [DEFAULT_CSV, "Calories.csv", "calories_burned.csv", "data.csv"]:
        if os.path.exists(name):
            return name
    return None

def make_onehot_encoder():
    """Create OneHotEncoder with the correct argument name depending on sklearn version."""
    sig = inspect.signature(OneHotEncoder)
    if 'sparse_output' in sig.parameters:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def make_ransac_regressor(base_estimator, **fixed_kwargs):
    """
    Construct a RANSACRegressor compatible with multiple sklearn versions.
    Uses the correct keyword name for the base estimator (e.g. 'estimator' or 'base_estimator').
    fixed_kwargs are other args like random_state, min_samples, residual_threshold...
    """
    sig = inspect.signature(RANSACRegressor)
    params = {}
    # choose correct param name for the base estimator
    if 'estimator' in sig.parameters:
        params['estimator'] = base_estimator
    elif 'base_estimator' in sig.parameters:
        params['base_estimator'] = base_estimator
    else:
        # fallback: try positional usage (not ideal but a last resort)
        return RANSACRegressor(base_estimator, **fixed_kwargs)
    # add other provided kwargs only if supported
    for k, v in fixed_kwargs.items():
        if k in sig.parameters:
            params[k] = v
    return RANSACRegressor(**params)

def main():
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    csv_path = locate_csv(csv_arg)
    if csv_path is None:
        print("Error: CSV file not found. Place 'calories.csv' in the same folder or pass its path:")
        print(r"Example: python task3.py 'D:\\task ML\\Task 3\\calories.csv'")
        sys.exit(1)

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    cols = df.columns.tolist()
    mapping = {}
    mapping['target'] = find_col_by_keywords(cols, ['calories', 'calories_burned', 'calorie'])
    mapping['age'] = find_col_by_keywords(cols, ['age'])
    mapping['weight'] = find_col_by_keywords(cols, ['weight', 'kg'])
    mapping['height'] = find_col_by_keywords(cols, ['height', 'cm'])
    mapping['duration'] = find_col_by_keywords(cols, ['duration', 'min', 'minutes', 'time', 'exercise_time'])
    mapping['heartrate'] = find_col_by_keywords(cols, ['heart', 'bpm', 'heart_rate', 'heartrate'])
    mapping['activity'] = find_col_by_keywords(cols, ['activity', 'activity_type', 'workout', 'exercise', 'mode'])

    # fallback for target
    if mapping['target'] is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise RuntimeError("No numeric column found to use as target. Edit mapping manually.")
        mapping['target'] = numeric_cols[-1]
        print(f"Target not auto-detected. Using last numeric column as target: {mapping['target']}")

    print("\nDetected mapping:")
    for k, v in mapping.items():
        print(f"  {k}: {v}")

    feature_keys = ['age', 'weight', 'height', 'duration', 'heartrate', 'activity']
    feature_cols = [mapping[k] for k in feature_keys if mapping.get(k)]
    target_col = mapping['target']

    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns detected. Please check CSV or update mapping.")

    data = df[feature_cols + [target_col]].copy()
    data = data.dropna(subset=[target_col])
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in data.columns if c not in numeric_cols and c != target_col]

    for c in numeric_cols:
        data[c] = data[c].fillna(data[c].median())
    for c in categorical_cols:
        data[c] = data[c].fillna('missing').astype(str)

    # compute BMI if possible
    if mapping.get('weight') and mapping.get('height'):
        wcol = mapping['weight']; hcol = mapping['height']
        def compute_bmi(row):
            try:
                h = float(row[hcol])
                w = float(row[wcol])
                if h > 2:
                    h = h / 100.0
                if h <= 0:
                    return np.nan
                return w / (h * h)
            except:
                return np.nan
        data['BMI'] = data.apply(compute_bmi, axis=1)
        data['BMI'] = data['BMI'].fillna(data['BMI'].median())
        if 'BMI' not in numeric_cols:
            numeric_cols.append('BMI')

    X = data.drop(columns=[target_col])
    y = data[target_col].astype(float)

    print(f"\nPrepared dataset: samples={len(data)}, features={X.shape[1]}")
    print("\nSample processed features:")
    print(X.head().to_string(index=False))

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', make_onehot_encoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    base = LinearRegression()
    # create RANSACRegressor compatibly
    ransac = make_ransac_regressor(base, random_state=RANDOM_STATE, min_samples=0.5)
    pipeline = Pipeline(steps=[
        ('pre', preprocessor),
        ('ransac', ransac)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Evaluation on test set ---")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R2  : {r2:.3f}")

    try:
        scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=5)
        print(f"\nCross-val MSE (5-fold) mean: {-np.mean(scores):.3f}  std: {np.std(scores):.3f}")
    except Exception as e:
        print("\nCross-val skipped or failed:", e)

    save_obj = {
        'pipeline': pipeline,
        'feature_columns': X.columns.tolist(),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    joblib.dump(save_obj, MODEL_FILENAME)
    print(f"\nSaved model to: {os.path.abspath(MODEL_FILENAME)}")

    # prediction for required example
    example = {}
    if mapping.get('age'): example[mapping['age']] = 25
    if mapping.get('weight'): example[mapping['weight']] = 70
    if mapping.get('height'): example[mapping['height']] = 175
    if mapping.get('duration'): example[mapping['duration']] = 60
    if mapping.get('heartrate'): example[mapping['heartrate']] = 130
    if mapping.get('activity'): example[mapping['activity']] = 'Brisk Walking'

    example_row = {c: np.nan for c in X.columns}
    for k, v in example.items():
        if k in example_row:
            example_row[k] = v
    for c in example_row:
        if c in numeric_features:
            if pd.isna(example_row[c]):
                example_row[c] = X[c].median()
        else:
            if pd.isna(example_row[c]):
                example_row[c] = 'missing'

    example_df = pd.DataFrame([example_row], columns=X.columns)
    pred_value = pipeline.predict(example_df)[0]

    print("\n--- Prediction for required example ---")
    print("Input used:")
    print(example_row)
    print(f"Predicted Calories Burned ≈ {pred_value:.2f}")

    print("\nFirst 10 test true vs predicted:")
    comp = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred}).reset_index(drop=True)
    print(comp.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
