import os
import torch
from flask import Flask, request, render_template
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

# Create Flask App
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to Predict Disease
def predict_skin_disease(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()
        return disease_labels.get(predicted_class_idx, "Unknown Disease")

    except Exception as e:
        return f"Error: {str(e)}"

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No file selected")

        if file:
            # Save Image
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Predict Disease
            predicted_disease = predict_skin_disease(filepath)
            return render_template("index.html", prediction=predicted_disease, image=filepath)

    return render_template("index.html")

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
