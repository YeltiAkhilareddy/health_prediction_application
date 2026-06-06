import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    f1_score
)

from imblearn.over_sampling import SMOTE

# =====================================
# 1. LOAD DATASET
# =====================================

df = pd.read_csv(
    r"D:\health_prediction_app\framingham.csv"
)

# =====================================
# 2. RENAME COLUMNS
# =====================================

df.rename(
    columns={
        "totChol": "cholesterol"
    },
    inplace=True
)

# =====================================
# 3. FEATURES
# =====================================

features = [
    "age",
    "male",
    "glucose",
    "cholesterol",
    "currentSmoker",
    "diabetes",
    "sysBP"
]

target = "TenYearCHD"

df = df[features + [target]]

# =====================================
# 4. HANDLE MISSING VALUES
# =====================================

for col in features:
    df[col] = df[col].fillna(
        df[col].median()
    )

# =====================================
# 5. SPLIT DATA
# =====================================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# 6. SMOTE
# =====================================

smote = SMOTE(
    random_state=42
)

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)

# =====================================
# 7. SAMPLE WEIGHTS
# =====================================

sample_weights = []

for _, row in X_train_res.iterrows():

    weight = 1.0

    if row["glucose"] > 250:
        weight *= 2

    if row["cholesterol"] > 300:
        weight *= 2

    if row["sysBP"] > 160:
        weight *= 2

    if row["diabetes"] == 1:
        weight *= 1.5

    sample_weights.append(weight)

# =====================================
# 8. TRAIN MODEL
# =====================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_res,
    y_train_res,
    sample_weight=sample_weights
)

# =====================================
# 9. PREDICTIONS
# =====================================

y_prob = model.predict_proba(X_test)[:, 1]

# =====================================
# 10. THRESHOLD TUNING
# =====================================

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60
]

best_threshold = 0.50
best_f1 = 0

for threshold in thresholds:

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    f1 = f1_score(
        y_test,
        y_pred
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\nBest Threshold:", best_threshold)
print("Best F1 Score:", round(best_f1, 4))

# =====================================
# 11. FINAL EVALUATION
# =====================================

y_pred = (
    y_prob >= best_threshold
).astype(int)

print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nROC-AUC:")
print(
    roc_auc_score(
        y_test,
        y_prob
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================
# 12. FEATURE IMPORTANCE
# =====================================

print("\nFeature Importance")

for feature, importance in zip(
    features,
    model.feature_importances_
):
    print(
        f"{feature}: {importance:.4f}"
    )

# =====================================
# 13. SAVE MODEL
# =====================================

with open(
    "health_model.pkl",
    "wb"
) as f:
    pickle.dump(
        model,
        f
    )

with open(
    "threshold.pkl",
    "wb"
) as f:
    pickle.dump(
        best_threshold,
        f
    )

print("\nModel Saved Successfully")
print(
    f"Saved Threshold: {best_threshold}"
)