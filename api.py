import os
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load Model & Processor
processor = AutoImageProcessor.from_pretrained("shreyasguha/22class_skindiseases_57acc")
model = AutoModelForImageClassification.from_pretrained("shreyasguha/22class_skindiseases_57acc")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Class Labels (Manually Defined)
disease_labels = {
    0: "Acne",
    1: "Eczema",
    2: "Psoriasis",
    3: "Melanoma",
    4: "Basal Cell Carcinoma",
    5: "Squamous Cell Carcinoma",
    6: "Actinic Keratosis",
    7: "Vitiligo",
    8: "Lupus",
    9: "Tinea",
    10: "Urticaria",
    11: "Rosacea",
    12: "Scabies",
    13: "Cellulitis",
    14: "Impetigo",
    15: "Molluscum Contagiosum",
    16: "Herpes Simplex",
    17: "Warts",
    18: "Melasma",
    19: "Lichen Planus",
    20: "Alopecia Areata",
    21: "Dermatitis"
}

# Create Flask API
app = Flask(__name__)

# API Route for Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Load & Process Image
        img = Image.open(file).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        # Get Prediction
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
        predicted_disease = disease_labels.get(predicted_class_idx, "Unknown Disease")

        # Return JSON Response
        return jsonify({"prediction": predicted_disease})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run Flask API
if __name__ == "__main__":
    app.run(debug=True)
