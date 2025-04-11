import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load and merge datasets
exercise = pd.read_csv("data/exercise.csv")
calories = pd.read_csv("data/calories.csv")
df = pd.merge(exercise, calories, on="User_ID")
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

# Final input features (model uses only these)
feature_cols = ['Age', 'Weight', 'Gender', 'Duration', 'Heart_Rate']
X = df[feature_cols]
y = df['Calories']

# Scale inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build and train model
model = Sequential([
    Input(shape=(len(feature_cols),)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Save float model
float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_model = float_converter.convert()
with open("calorie_model.tflite", "wb") as f:
    f.write(float_model)

# Quantization
def representative_data_gen():
    for i in range(min(100, len(X_scaled))):
        yield [X_scaled[i].reshape(1, -1).astype(np.float32)]

quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
quant_converter.representative_dataset = representative_data_gen
quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
quant_converter.inference_input_type = tf.int8
quant_converter.inference_output_type = tf.int8

quantized_model = quant_converter.convert()
with open("calorie_model_quant.tflite", "wb") as f:
    f.write(quantized_model)

# Convert to Arduino header file
with open("calorie_model_quant.h", "w") as f:
    f.write("const unsigned char calorie_nn_quant[] = {\n")
    for i, b in enumerate(quantized_model):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int calorie_nn_quant_len = {len(quantized_model)};\n")
