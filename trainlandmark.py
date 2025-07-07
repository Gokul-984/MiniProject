import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle

# Load dataset
df = pd.read_csv("landmark_dataset.csv")
X = df.iloc[:, 1:].values  # Landmark features
y = df.iloc[:, 0].values   # Labels

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
mlp.fit(X_train, y_train)

# Save models
with open("landmark_mlp.pkl", "wb") as f:
    pickle.dump(mlp, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("MLP model trained and saved!")
