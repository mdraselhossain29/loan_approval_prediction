"""
Loan Approval Prediction - Model Training Script
Train and save all models for the web application
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

print("=" * 60)
print("🚀 Starting Model Training Pipeline")
print("=" * 60)

# Create models directory
os.makedirs('models', exist_ok=True)

# Load dataset
print("\n📊 Loading dataset...")
df = pd.read_csv('loan_data.csv')  # তোমার ডাটা ফাইলের পাথ দাও
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Preprocessing
print("\n🔧 Preprocessing data...")

# Encode target
df['Approval'] = df['Approval'].map({'Approved': 1, 'Rejected': 0})

# Encode categorical
df['Employment_Status'] = df['Employment_Status'].map({'employed': 1, 'unemployed': 0})

# Text preprocessing
if 'Text' in df.columns:
    df['clean_text'] = df['Text'].str.lower()
else:
    df['clean_text'] = "loan application"

# Features
numeric_cols = ['Income', 'Credit_Score', 'Loan_Amount', 'DTI_Ratio', 'Employment_Status']
text_col = 'clean_text'

# Split data
X = df[numeric_cols + [text_col]]
y = df['Approval']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Preprocessing pipeline
print("\n🔨 Building preprocessing pipeline...")

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

text_transformer = TfidfVectorizer(max_features=100, stop_words='english')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('text', text_transformer, text_col)
])

# Calculate scale_pos_weight for imbalanced data
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Base models
print("\n🤖 Training base models...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

# Meta-learner
meta_learner = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

# Stacking ensemble
print("\n🎯 Building stacking ensemble...")
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm)
    ],
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

# Create pipelines
stacking_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])

xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb)
])

# Train models
print("\n🏋️ Training stacking ensemble...")
stacking_pipe.fit(X_train, y_train)

print("\n🏋️ Training XGBoost for SHAP...")
xgb_pipe.fit(X_train, y_train)

# Evaluate
print("\n📈 Evaluating models...")
y_pred_stack = stacking_pipe.predict(X_test)
y_pred_proba = stacking_pipe.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred_stack)
f1 = f1_score(y_test, y_pred_stack)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nStacking Ensemble Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC-AUC: {auc:.4f}")

# Save models
print("\n💾 Saving models...")
joblib.dump(stacking_pipe, 'models/stacking_model.pkl')
joblib.dump(xgb_pipe, 'models/xgb_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')

print("\n✅ All models saved successfully!")
print("models saved in 'models/' directory:")
print("  - stacking_model.pkl")
print("  - xgb_model.pkl")
print("  - preprocessor.pkl")
print("\n🎉 Training completed successfully!")