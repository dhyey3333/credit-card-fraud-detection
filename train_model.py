import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv("creditcard.csv")

FEATURES = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

X = data[FEATURES]
y = data['Class']

# -----------------------------
# SCALE
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================================
# 🔹 LOGISTIC REGRESSION
# =========================================================
print("Training Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

lr_model.fit(X_train, y_train)

joblib.dump(lr_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

y_pred_lr = lr_model.predict(X_test)

print("\n📊 LR PERFORMANCE")
print("Accuracy :", round(accuracy_score(y_test, y_pred_lr), 4))
print("Precision:", round(precision_score(y_test, y_pred_lr), 4))
print("Recall   :", round(recall_score(y_test, y_pred_lr), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_lr), 4))

# =========================================================
# 🔹 ANN (FOR DL REQUIREMENT)
# =========================================================
print("\nTraining ANN...")

ann_model = Sequential([
    Dense(32, activation='relu', input_shape=(30,)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=2048,
    verbose=1
)

ann_model.save("ann_model.h5")

y_pred_ann = (ann_model.predict(X_test) > 0.5).astype(int)

print("\n📊 ANN PERFORMANCE")
print("Accuracy :", round(accuracy_score(y_test, y_pred_ann), 4))
print("Precision:", round(precision_score(y_test, y_pred_ann), 4))
print("Recall   :", round(recall_score(y_test, y_pred_ann), 4))
print("F1 Score :", round(f1_score(y_test, y_pred_ann), 4))
