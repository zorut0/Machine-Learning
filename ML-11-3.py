import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("CREDITSCORE.csv")

label_encoder = LabelEncoder()
data['Credit_Mix'] = label_encoder.fit_transform(data['Credit_Mix'])

# Define features and target variable
features = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Credit_Mix", "Outstanding_Debt", "Credit_History_Age", "Monthly_Balance"
]

X = data[features].to_numpy()
y = data["Credit_Score"].to_numpy()

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
nb_predictions = nb_model.predict(x_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

print("=== Model Comparison: Random Forest vs Naive Bayes ===")
print(f"Random Forest Accuracy : {rf_accuracy:.4f}")
print(f"Naive Bayes Accuracy   : {nb_accuracy:.4f}")
print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_predictions))

print("\n--- Naive Bayes Classification Report ---")
print(classification_report(y_test, nb_predictions))
