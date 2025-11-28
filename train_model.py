# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv(r"C:\\Users\\balaj\\imp\\churn predicton\\Telco-Customer-Churn.csv")

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Drop ID
df.drop(['customerID'], axis=1, inplace=True)

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

# Pipeline = preprocessing + model
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf),
])

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Random Forest + Pipeline Accuracy:", acc)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save pipeline
joblib.dump(clf, "churn_rf_pipeline.pkl")
print("Saved model as churn_rf_pipeline.pkl")

# (Optional) Save column names (for app input)
joblib.dump(X.columns.tolist(), "input_columns.pkl")
print("Saved input columns.")
