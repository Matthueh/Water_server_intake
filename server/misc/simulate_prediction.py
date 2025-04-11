import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import random

# === Load model and scaler ===
model_path = "calorie_model_quant.tflite"
scaler_path = "scaler.pkl"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# === Simulate input values ===
def fabricate_input():
    input_data = {
        'Age': random.randint(18, 65),
        'Height': random.randint(150, 200),
        'Weight': random.randint(50, 120),
        'Gender': random.choice(['male', 'female']),
        'Duration': random.randint(15, 240),
        'Heart_Rate': random.randint(60, 180),
        'Body_Temp': round(random.uniform(97.0, 102.0), 1)
    }
    return input_data

def predict(input_data):
    df = pd.DataFrame([input_data])
    df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

    # === Filter only model features ===
    model_features = ['Age', 'Weight', 'Gender', 'Duration', 'Heart_Rate']
    X_scaled = scaler.transform(df[model_features].astype(np.float32))

    # === Quantize input ===
    input_scale, input_zero_point = input_details[0]['quantization']
    quantized_input = (X_scaled / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()

    # === Dequantize output ===
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_scale, output_zero_point = output_details[0]['quantization']
    predicted_calories = (output_data[0][0] - output_zero_point) * output_scale

    # === Water Loss ===
    water_loss_mL = predicted_calories / 2.42
    hike_minutes = input_data['Duration']
    sun_hydration_mL = (hike_minutes / 60) * 700
    total_water_mL = water_loss_mL + sun_hydration_mL
    total_water_L = total_water_mL / 1000

    # === Output ===
    print("\n🧪 === Simulated Input ===")
    for k, v in input_data.items():
        print(f"{k}: {v}")

    print(f"\n🔢 Predicted Calories Burned: {predicted_calories:.2f}")
    print(f"💧 Water Lost from Exercise: {water_loss_mL:.2f} mL")
    print(f"🌤️  Baseline Need from Sun Exposure ({hike_minutes} min): {sun_hydration_mL:.2f} mL")
    print(f"🧃 Total Water Recommended: {total_water_mL:.2f} mL ({total_water_L:.2f} L)\n")

# === Run it ===
if __name__ == "__main__":
    test_input = fabricate_input()
    predict(test_input)
