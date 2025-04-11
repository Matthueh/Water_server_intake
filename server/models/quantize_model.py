import tensorflow as tf
import numpy as np

# Load your float model
with open("calorie_nn.tflite", "rb") as f:
    float_model = f.read()

# Convert from TFLite (not supported directly), so use representative dummy model
# Here we simulate by building a dummy float32 keras model structure
# and attaching the quantization parameters

# If you trained a model using Keras, do this instead:
# model = tf.keras.models.load_model("your_model.h5")
# converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Load as interpreter
interpreter = tf.lite.Interpreter(model_content=float_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']

# Representative dataset
def representative_data_gen():
    for _ in range(100):
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        yield [dummy_input]

# Create a quantized model
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [interpreter._get_concrete_function()]
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quant_model = converter.convert()

# Save quantized model
with open("calorie_model_quant.tflite", "wb") as f:
    f.write(quant_model)

print("✅ Quantized model saved as calorie_model_quant.tflite")
