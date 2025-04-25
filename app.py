import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Drop 'Unnamed: 0' if exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Impute missing values
imputer = SimpleImputer(strategy='most_frequent')
df[['Saving accounts', 'Checking account']] = imputer.fit_transform(df[['Saving accounts', 'Checking account']])

# Label Encoding
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

le_housing = LabelEncoder()
df['Housing'] = le_housing.fit_transform(df['Housing'])

le_saving = LabelEncoder()
df['Saving accounts'] = le_saving.fit_transform(df['Saving accounts'].astype(str))

le_purpose = LabelEncoder()
df['Purpose'] = le_purpose.fit_transform(df['Purpose'])

# Risk scoring function
def assign_credit_risk(row):
    risk_score = 0
    if row['Credit amount'] > 15000:
        risk_score += 2
    if row['Duration'] > 36:
        risk_score += 2
    if row['Saving accounts'] in ['little', 'moderate']:
        risk_score += 1
    if row['Checking account'] in ['little', 'moderate']:
        risk_score += 1
    if row['Job'] in [0, 1]:  # unskilled
        risk_score += 1
    if row['Age'] < 25 or row['Age'] > 60:
        risk_score += 1
    return 'bad' if risk_score >= 4 else 'good'

df['Credit Risk'] = df.apply(assign_credit_risk, axis=1)

# Encode 'Credit Risk'
le_risk = LabelEncoder()
df['Credit Risk'] = le_risk.fit_transform(df['Credit Risk'])

# One Hot Encoding for categorical features
categorical_features = ['Saving accounts', 'Checking account']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_data = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

# Combine Data
df = df.drop(categorical_features, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# Features & Target split
X = df.drop('Credit Risk', axis=1)
y = df['Credit Risk']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)

# Streamlit UI
st.title("Credit Risk Prediction App")

st.sidebar.header("Input Customer Information")

# User inputs
age = st.sidebar.slider("Age", 18, 80, 30)
job = st.sidebar.selectbox("Job (0-Unskilled, 1-UnskilledResident, 2-Skilled, 3-Highly Skilled)", [0, 1, 2, 3])
credit_amount = st.sidebar.number_input("Credit Amount", 250, 20000, 1000)
duration = st.sidebar.slider("Duration (months)", 4, 72, 12)
housing = st.sidebar.selectbox("Housing (0-free, 1-own, 2-rent)", [0, 1, 2])
purpose = st.sidebar.selectbox("Purpose (encoded)", sorted(df['Purpose'].unique()))
sex = st.sidebar.selectbox("Sex (0-Female, 1-Male)", [0, 1])
saving_account = st.sidebar.selectbox("Saving Account Category", encoder.categories_[0])
checking_account = st.sidebar.selectbox("Checking Account Category", encoder.categories_[1])

# Prepare input in correct order
if st.sidebar.button("Predict Credit Risk"):
    input_values = []
    for col in X.columns:
        if col == 'Age':
            input_values.append(age)
        elif col == 'Job':
            input_values.append(job)
        elif col == 'Credit amount':
            input_values.append(credit_amount)
        elif col == 'Duration':
            input_values.append(duration)
        elif col == 'Housing':
            input_values.append(housing)
        elif col == 'Purpose':
            input_values.append(purpose)
        elif col == 'Sex':
            input_values.append(sex)
        elif col.startswith('Saving accounts_'):
            sa_cat = col.split('_', 1)[1]
            input_values.append(1 if sa_cat == saving_account else 0)
        elif col.startswith('Checking account_'):
            ca_cat = col.split('_', 1)[1]
            input_values.append(1 if ca_cat == checking_account else 0)
        else:
            input_values.append(0)

    input_df = pd.DataFrame([input_values], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prediction = dt.predict(input_scaled)
    predicted_label = le_risk.inverse_transform(prediction)[0]

    st.subheader(f"Credit Risk Prediction: **{predicted_label.upper()} RISK**")


