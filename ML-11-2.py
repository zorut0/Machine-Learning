import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("CREDITSCORE.csv")

# Encode 'Credit_Mix' (categorical: 'Bad', 'Standard', 'Good')
credit_mix_encoder = LabelEncoder()
data['Credit_Mix'] = credit_mix_encoder.fit_transform(data['Credit_Mix'])

# Features and target
features_list = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
     "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Credit_Mix", "Outstanding_Debt", "Credit_History_Age", "Monthly_Balance"
]

X = data[features_list].to_numpy()
y = data["Credit_Score"].to_numpy()

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train Naive Bayes model (GaussianNB for continuous features)
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

# Collect user input
print("\n--- Credit Score Prediction Using Naive Bayes ---")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit Cards: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed: "))
h = float(input("Number of delayed payments: "))

credit_mix_input = input("Credit Mix (Bad, Standard, Good): ").capitalize()
if credit_mix_input not in credit_mix_encoder.classes_:
    print("❌ Invalid Credit Mix. Please enter one of:", list(credit_mix_encoder.classes_))
    exit()

credit_mix_encoded = credit_mix_encoder.transform([credit_mix_input])[0]

i = float(input("Outstanding Debt: "))
j = float(input("Credit History Age: "))
k = float(input("Monthly Balance: "))

# Prepare user data for prediction
user_features = np.array([[a, b, c, d, f, g, h, credit_mix_encoded, i, j, k]])

# Predict using the model
prediction = nb_model.predict(user_features)
print("\n✅ Predicted Credit Score:", prediction[0])
