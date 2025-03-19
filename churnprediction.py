import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# Load dataset (Assuming train_df and test_df are provided)
X = train_df.drop(['CustomerID', 'Churn'], axis=1)  # Features
y = train_df['Churn']  # Target variable

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# **Workaround for Ordinal Encoding (Older Versions of Scikit-Learn)**
def ordinal_encode_fit_transform(df, categorical_columns):
    encoder = OrdinalEncoder()
    df[categorical_columns] = encoder.fit_transform(df[categorical_columns].astype(str))  # Convert to string first
    return df, encoder

# Apply Ordinal Encoding to Training Data
X, ordinal_encoder = ordinal_encode_fit_transform(X, categorical_cols)

# Train-Test Split (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# **Preprocessing Pipeline: Standard Scale Numerical Data**
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols)  
], remainder='passthrough')  # Keeps categorical data as-is (already encoded)

# Dummy Classifier Pipeline (Stratified Strategy)
dummy_clf = Pipeline([
    ('preprocess', preprocessor),
    ('model', DummyClassifier(strategy="stratified"))
])

# Train the Dummy Classifier
dummy_clf.fit(X_train, y_train)

# Predict probabilities for validation set
y_val_pred_proba = dummy_clf.predict_proba(X_val)[:, 1]  # Get probabilities of class 1

# Calculate ROC-AUC score
roc_auc = roc_auc_score(y_val, y_val_pred_proba)
print(f"ROC-AUC Score (Dummy Classifier without One-Hot Encoding): {roc_auc:.4f}")

# **Process Test Data**
X_test = test_df.drop(['CustomerID'], axis=1)

# Apply the same ordinal encoding transformation to test data
X_test[categorical_cols] = ordinal_encoder.transform(X_test[categorical_cols].astype(str))

# Make Predictions
final_predictions = dummy_clf.predict_proba(X_test)[:, 1]  # Probabilities for test set

# Create Submission DataFrame
prediction_df = pd.DataFrame({
    'CustomerID': test_df['CustomerID'].values,
    'predicted_probability': final_predictions
})

# Ensure the format is correct
print(prediction_df.shape)  # Should be (104480, 2)
print(prediction_df.head(10)) 
