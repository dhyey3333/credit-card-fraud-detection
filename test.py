import joblib
import pandas as pd

# train_model.py saves the model and the scaler with joblib (plain pickle can't read them).
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

# Example input: one transaction. V1..V28 are the dataset's anonymised PCA features (0 = average).
transaction = pd.DataFrame([[0.0] * len(FEATURES)], columns=FEATURES)
transaction["Time"] = 100
transaction["Amount"] = 500

prediction = model.predict(scaler.transform(transaction))

if prediction[0] == 1:
    print("Fraud Detected")
else:
    print("Legitimate Transaction")
