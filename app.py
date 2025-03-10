from flask import Flask, request, jsonify
import os
import tensorflow.lite as tflite
import numpy as np
from PIL import Image

app = Flask(__name__)

# Define class labels
CLASS_LABELS = ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma', 'Normal']

# Load TFLite model
MODEL_PATH = r"./model.tflite"  # Replace with the actual model filename
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

@app.route("/")  # This is the homepage
def home():
    return "Cancer Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        image = Image.open(file).resize((150, 150))  # Resize to model input size
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])[0]

        # Get the predicted class index
        predicted_index = np.argmax(output_data)

        # Map the index to the class label
        predicted_label = CLASS_LABELS[predicted_index]
        confidence_score = float(np.max(output_data))  # Get confidence score
        print(f"✅ Prediction: {predicted_label}, Confidence: {confidence_score}")  # Debugging

        return jsonify({"prediction": predicted_label, "confidence": confidence_score})

    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
