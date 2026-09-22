import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import joblib

st.set_page_config(layout="wide")

st.title("Customer Churn Prediction App")
st.write("This app predicts customer churn using a Logistic Regression model.")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv('/content/churn_prediction.csv') # Adjust path if needed
    return df

df = load_data()

if st.checkbox('Show raw data'):
    st.subheader('Raw Data')
    st.dataframe(df)

st.subheader("Data Description")
st.dataframe(df.describe())

# --- Load Pre-trained Model, Scaler, and Encoder ---
# The model, scaler, and encoder are loaded here to avoid re-training/re-fitting on each app run.
# Ensure 'bank.sav', 'scaler.sav', and 'encoder.sav' are available in the working directory.
try:
    log_reg_model = joblib.load('bank.sav')
    scaler = joblib.load('scaler.sav')
    encoder = joblib.load('encoder.sav')
    st.success("Pre-trained Model, Scaler, and Encoder Loaded Successfully!")
except FileNotFoundError:
    st.error("Error: Pre-trained model, scaler, or encoder files not found. Please ensure 'bank.sav', 'scaler.sav', and 'encoder.sav' are in the same directory as the app.")
    st.stop() # Stop the app if files are not found

# --- Feature Engineering & Preprocessing ---
# Using the loaded encoder and scaler to transform data.

st.sidebar.header('Model Parameters') # Keep for test_size slider for demonstration
test_size = st.sidebar.slider('Test Set Size', 0.1, 0.5, 0.2, 0.05)
random_state = 42 # Fixed for reproducibility

st.subheader("Data Preprocessing and Model Evaluation with Loaded Model")

# Assuming 'churn' is the target variable
X = df.drop('churn', axis=1)
y = df['churn']

# Drop customer_id as it's an identifier
if 'customer_id' in X.columns:
    X = X.drop('customer_id', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

st.write(f"Shape of X_train: {X_train.shape}")
st.write(f"Shape of X_test: {X_test.shape}")

# Identify categorical and numerical columns for processing
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
# These columns were identified with missing values in the original notebook
numerical_cols_with_missing = ['dependents', 'city']

# Impute missing numerical values with the median (using X_train's median for consistency)
for col in numerical_cols_with_missing:
    if col in X_train.columns: # Check if column still exists after dropping 'customer_id'
        median_val = X_train[col].median()
        X_train[col].fillna(median_val, inplace=True)
        X_test[col].fillna(median_val, inplace=True)

# Impute missing categorical values with the mode (using X_train's mode for consistency)
for col in categorical_cols:
    if col in X_train.columns:
        mode_val = X_train[col].mode()[0]
        X_train[col].fillna(mode_val, inplace=True)
        X_test[col].fillna(mode_val, inplace=True)

st.write("Missing values imputed.")

# One-hot encode categorical features using the LOADED encoder
X_train_encoded = encoder.transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])

X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X_test.index)

X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded_df], axis=1)
X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded_df], axis=1)

st.write("Categorical features one-hot encoded using loaded encoder.")

# Scale numerical features using the LOADED scaler
# Select only numerical columns for scaling after other processing steps
final_numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[final_numerical_cols] = scaler.transform(X_train[final_numerical_cols])
X_test_scaled[final_numerical_cols] = scaler.transform(X_test[final_numerical_cols])

st.write("Numerical features scaled using loaded scaler.")

# --- Model Evaluation (using the loaded model) ---
st.subheader("Model Evaluation")

y_pred_log_reg_scaled = log_reg_model.predict(X_test_scaled)

st.write("#### Accuracy Score")
accuracy = accuracy_score(y_test, y_pred_log_reg_scaled)
st.write(f"Accuracy: {accuracy:.4f}")

st.write("#### Classification Report")
st.text(classification_report(y_test, y_pred_log_reg_scaled))

st.write("#### Confusion Matrix Heatmap")
conf_matrix = confusion_matrix(y_test, y_pred_log_reg_scaled)
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Non-Churn', 'Predicted Churn'],
            yticklabels=['Actual Non-Churn', 'Actual Churn'], ax=ax_cm)
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('True Label')
ax_cm.set_title('Confusion Matrix Heatmap (Scaled Logistic Regression)')
st.pyplot(fig_cm)

st.write("#### ROC Curve")
y_pred_proba = log_reg_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
ax_roc.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
ax_roc.legend()
ax_roc.grid(True)
st.pyplot(fig_roc)

st.write(f"AUC Score: {auc_score:.4f}")
