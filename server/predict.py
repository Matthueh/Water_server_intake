import numpy as np
import tensorflow as tf
import pickle
import pandas as pd

def predict_water_intake(input_data, model_path, scaler_path):
    # Expected input keys: ['Age', 'Height', 'Weight', 'Gender', 'Duration', 'Heart_Rate', 'Body_Temp']
    required_keys = ['Age', 'Height', 'Weight', 'Gender', 'Duration', 'Heart_Rate', 'Body_Temp']
    if not all(k in input_data for k in required_keys):
        raise ValueError("Missing required input keys")

    df = pd.DataFrame([input_data])[required_keys]

    # Encode gender
    df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(df.values.astype(np.float32))

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X_scaled.astype(np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_calories = float(output_data[0][0])

    # Convert to water loss
    water_loss_liters = predicted_calories / 2.42

    print(f"🔢 Predicted Calories: {predicted_calories:.2f} → 💧 Estimated Water Loss: {water_loss_liters:.2f} L")
    return water_loss_liters
