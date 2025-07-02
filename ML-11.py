import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("CREDITSCORE.csv")

# Encode 'Credit_Mix' which is a string column (Standard, Good, Bad)
credit_mix_encoder = LabelEncoder()
data['Credit_Mix'] = credit_mix_encoder.fit_transform(data['Credit_Mix'])

# Features to be used for prediction
features_list = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Credit_Mix", "Outstanding_Debt", "Credit_History_Age", "Monthly_Balance"
]

# Convert to numpy array
x = data[features_list].to_numpy()
y = data["Credit_Score"].to_numpy()

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# User input
print("\n--- Credit Score Prediction ---")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed: "))
h = float(input("Number of delayed payments: "))

# Encode Credit Mix input to numerical value
credit_mix_input = input("Credit Mix (Bad, Standard, Good): ").capitalize()
if credit_mix_input not in credit_mix_encoder.classes_:
    print("Invalid Credit Mix input. Please enter one of:", list(credit_mix_encoder.classes_))
    exit()

credit_mix_encoded = credit_mix_encoder.transform([credit_mix_input])[0]

i = float(input("Outstanding Debt: "))
j = float(input("Credit History Age: "))
k = float(input("Monthly Balance: "))

# Format user features into 2D array
user_features = np.array([[a, b, c, d, e, f, g, h, credit_mix_encoded, i, j, k]])

# Make prediction
predicted = model.predict(user_features)
print("Predicted Credit Score =", predicted[0])
