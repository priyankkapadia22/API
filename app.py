from flask import Flask, request, jsonify
import os
import tensorflow.lite as tflite
import numpy as np
from PIL import Image
from collections import Counter

app = Flask(__name__)

# Define class labels
CLASS_LABELS = ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma', 'Normal']

# Model directory
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Define model paths
MODEL_PATHS = {
    "model1": os.path.join(MODEL_DIR, "lung_cancer_mobilenet.tflite"),
    "model2": os.path.join(MODEL_DIR, "lung_cancer_resnet.tflite"),
    "model3": os.path.join(MODEL_DIR, "lung_cancer_vgg19.tflite"),
}

# Load interpreters dynamically when needed to save RAM
def load_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@app.route("/")
def home():
    return "🚀 Optimized Cancer Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        # Preprocess image
        image = Image.open(file).convert("RGB").resize((150, 150))
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = []
        confidence_scores = []
        model_predictions = []

        # Load and use each model one by one to optimize RAM usage
        for model_name, model_path in MODEL_PATHS.items():
            interpreter = load_interpreter(model_path)

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]["index"], image)
            interpreter.invoke()

            raw_output = interpreter.get_tensor(output_details[0]["index"])
            raw_output = np.squeeze(raw_output)

            predicted_class = int(np.argmax(raw_output))
            confidence_score = float(np.max(raw_output))

            predictions.append(predicted_class)
            confidence_scores.append(confidence_score)

            model_predictions.append({
                "model": model_name,
                "predicted_class": CLASS_LABELS[predicted_class],
                "confidence": confidence_score
            })

        # Majority Voting
        class_votes = Counter(predictions)
        final_class = class_votes.most_common(1)[0][0]

        # Compute final confidence
        relevant_confidences = [conf for i, conf in enumerate(confidence_scores) if predictions[i] == final_class]
        final_confidence = float(np.mean(relevant_confidences))

        final_prediction = {
            "final_predicted_class": CLASS_LABELS[final_class],
            "final_confidence": final_confidence
        }

        return jsonify({
            "individual_model_predictions": model_predictions,
            "final_prediction": final_prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
