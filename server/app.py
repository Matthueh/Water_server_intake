from flask import Flask, request, Response, send_file
from predict import predict_water_intake  # still useful for internal validation/testing
from retrain import train_and_export_model
from apscheduler.schedulers.background import BackgroundScheduler
import os
import datetime
import zipfile
import io

app = Flask(__name__)
app.url_map.strict_slashes = False

# === Paths ===
MODEL_PATH = "models/calorie_nn.tflite"
SCALER_PATH = "models/scaler.pkl"
HEADER_PATH = "models/calorie_model_quant.h"
EXERCISE_PATH = "data/exercise.csv"
CALORIES_PATH = "data/calories.csv"
LOG_PATH = "logs/prediction_log.csv"

# === Folder Setup ===
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# === Retraining Schedule ===
def scheduled_training():
    print(f"[🔁] Retraining model at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    train_and_export_model(EXERCISE_PATH, CALORIES_PATH, MODEL_PATH, SCALER_PATH)

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_training, 'interval', minutes=1)
scheduler.start()
print("⏳ Automatic retraining scheduled every 1 minute.")

# === File Download Endpoints ===
@app.route("/model", methods=["GET"])
def get_model():
    return send_file(MODEL_PATH, as_attachment=True)

@app.route("/scaler", methods=["GET"])
def get_scaler():
    return send_file(SCALER_PATH, as_attachment=True)

@app.route("/arduino-model", methods=["GET"])
def get_arduino_header():
    return send_file(HEADER_PATH, as_attachment=True)

# === New Arduino Client File Request ===
@app.route("/request-model", methods=["POST"])
def send_model_package():
    try:
        print(f"[📥] Arduino requested files at {datetime.datetime.now()} from {request.remote_addr}")

        # Zip header + scaler
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.write(HEADER_PATH, arcname="calorie_model_quant.h")
            zf.write(SCALER_PATH, arcname="scaler.pkl")
        zip_buffer.seek(0)

        return Response(
            zip_buffer,
            mimetype='application/zip',
            headers={"Content-Disposition": "attachment; filename=model_package.zip"}
        )
    except Exception as e:
        return {"error": str(e)}, 500

# === Launch Server ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
