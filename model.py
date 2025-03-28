from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# Load Pretrained Model & Processor
processor = AutoImageProcessor.from_pretrained("shreyasguha/22class_skindiseases_57acc")
model = AutoModelForImageClassification.from_pretrained("shreyasguha/22class_skindiseases_57acc")

# Class Labels (Modify if needed)
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

# Function to Predict Skin Disease
def predict_skin_disease(img_path):
    img = Image.open(img_path).convert("RGB")  # Open Image
    inputs = processor(images=img, return_tensors="pt")  # Preprocess Image

    with torch.no_grad():  # No need for gradient calculation
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return disease_labels[predicted_class]

# Example Usage
result = predict_skin_disease(r"C:\Users\prem thakkar\OneDrive\Desktop\sd-198\images\Acute_Eczema\040105VB.jpg")
print("Predicted Disease:", result)
