import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import pickle
import os
import subprocess



def train_and_export_model(exercise_path, calories_path, tflite_path, scaler_path):
    print("📦 Starting model training...")

    # Load and merge datasets
    try:
        exercise_df = pd.read_csv(exercise_path)
        calories_df = pd.read_csv(calories_path)
        df = pd.merge(exercise_df, calories_df, on="User_ID")
        df = df.dropna()
        df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
        print("📊 Datasets loaded and merged.")
    except Exception as e:
        print("❌ Failed to load or merge data:", e)
        return

    features = ['Age', 'Height', 'Weight', 'Gender', 'Duration', 'Heart_Rate', 'Body_Temp']
    target = 'Calories'

    try:
        X = df[features].astype(np.float32)
        y = df[target].astype(np.float32)
    except Exception as e:
        print("❌ Feature extraction failed:", e)
        return

    print("✅ Features shape:", X.shape)

    # Standardize features
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"📦 Scaler saved to: {scaler_path}")
    except Exception as e:
        print("❌ Failed to scale features or save scaler:", e)
        return

    # Train sklearn model (optional evaluation)
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_scaled, y)

    # Convert to TFLite using Keras
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, InputLayer

        keras_model = Sequential([
            InputLayer(input_shape=(X.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        keras_model.compile(optimizer='adam', loss='mse')
        keras_model.fit(X_scaled, y, epochs=50, verbose=0)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        print("✅ Model converted to TFLite format")
    except Exception as e:
        print("❌ TFLite conversion failed:", e)
        return

    # Save the TFLite model
    try:
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model saved to: {tflite_path}")

        # ✅ Generate header after model is saved
        
        subprocess.call(["python", "generate_header.py"])

    except Exception as e:
        print("❌ Failed to save TFLite model or generate header:", e)



if __name__ == "__main__":
    train_and_export_model(
        "data/exercise.csv",
        "data/calories.csv",
        "models/calorie_nn.tflite",
        "models/scaler.pkl"
    )
