def convert_tflite_to_header(tflite_path, header_path, var_name="calorie_nn_tflite"):
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    with open(header_path, 'w') as h:
        h.write(f"unsigned char {var_name}[] = {{\n")
        for i, byte in enumerate(model_data):
            if i % 12 == 0:
                h.write("\n  ")
            h.write(f"0x{byte:02x}, ")
        h.write(f"\n}};\n")
        h.write(f"unsigned int {var_name}_len = {len(model_data)};\n")

# Usage example
if __name__ == "__main__":
    convert_tflite_to_header(
        "models/calorie_nn.tflite",
        "models/calorie_nn_quant.h"
    )
