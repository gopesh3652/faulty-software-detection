# train_model.py

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("combined_datasetiii.csv")
df.fillna(df.mean(), inplace=True)

# Split features/target
X = df.drop(columns=['FNP'])
y = df['FNP']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Train model with GridSearch
xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_grid = {
    'max_depth': [3],
    'n_estimators': [100],
    'learning_rate': [0.05],
    'subsample': [1.0],
    'colsample_bytree': [1.0]
}
grid_search = GridSearchCV(xgb_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train_res)

best_model = grid_search.best_estimator_
y_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Find best threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Save model, scaler, and threshold
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("threshold.txt", "w") as f:
    f.write(str(optimal_threshold))

print("✅ Model, scaler, and threshold saved.")

# Save columns
with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
