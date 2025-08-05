import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
file_path = "C:\\Users\\SANDHYARANI\\Desktop\\coe\\sbacase.csv"
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit()

df = pd.read_csv(file_path, encoding='latin-1')
#Preprocessing
drop_columns = ['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'ApprovalDate', 'DisbursementDate', 'ChgOffDate']
df.drop(columns=drop_columns, inplace=True, errors='ignore')
#Drop rows with missing target
df.dropna(subset=['MIS_Status'], inplace=True)
# Target encoding: Default = 1 if CHGOFF
df['Default'] = df['MIS_Status'].apply(lambda x: 1 if x.strip() == "CHGOFF" else 0)
# Selected features
selected_features = [
    'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob', 'FranchiseCode', 'RevLineCr', 'LowDoc',
    'DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv'
]
# Imputation
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])
for col in cat_cols:
    df[col] = df[col].fillna("Unknown")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
# Ensuring all selected features are present
for feature in selected_features:
    if feature not in df.columns:
        df[feature] = 0
df_model = df[selected_features + ['Default']]
#Split & Resample
X = df_model.drop(columns=['Default'])
y = df_model['Default']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
#Normalize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_lambda=1.5,
    reg_alpha=0.5,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
joblib.dump(model, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "model_features.pkl")
print("\n🔥 Model, Scaler & Feature List saved successfully!")
