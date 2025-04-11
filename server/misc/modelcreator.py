import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Paths
DATA_PATH = "data/exercise.csv"
SCALER_OUTPUT_PATH = "models/scaler.pkl"

# Load and clean data
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

# Select features
features = ['Age', 'Height', 'Weight', 'Gender', 'Duration', 'Heart_Rate', 'Body_Temp']
X = df[features].astype(np.float32)

# Create and save the scaler
scaler = StandardScaler()
scaler.fit(X)

# Make sure models/ directory exists
os.makedirs("models", exist_ok=True)

with open(SCALER_OUTPUT_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pkl created and saved to 'models/'")
