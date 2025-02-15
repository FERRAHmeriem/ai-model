import cv2
import os
import numpy as np
import pickle
from ultralytics import YOLO
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# 📌 Charger le modèle YOLO pour la détection d'objets
detector = YOLO("yolov8m.pt")

# 📌 Charger le modèle SVM, le label_map et le scaler
with open("svm_model.pkl", "rb") as file:
    loaded_svm = pickle.load(file)

with open("label_map.pkl", "rb") as file:
    label_map = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# 🔹 Récupérer la liste des catégories en fonction du modèle entraîné
categories = {v: k for k, v in label_map.items()}  # Inverser le dictionnaire

# 🔹 Charger ResNet50 pour l'extraction des features
feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_deep_features(img):
    """Extraire les features d'une image avec ResNet50."""
    img = cv2.resize(img, (224, 224))  # Redimensionner l'image
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = feature_extractor.predict(img_data)
    return features.flatten()

def calculate_sharpness(image):
    """Calculer la netteté d'une image avec la variance du Laplacien."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def classify_product(image):
    """Classifier un produit détecté en plusieurs catégories avec SVM."""
    try:
        # Extraire les features avec ResNet50
        features = extract_deep_features(image)
        features_scaled = scaler.transform([features])  # Normaliser les features
        category_index = loaded_svm.predict(features_scaled)[0]  # Index de la classe
        return categories.get(category_index, "Autres")
    except Exception as e:
        print(f"⚠️ Erreur lors de la classification: {e}")
        return "Autres"

def process_image(image_path):
    """Détecter, classifier et compter les produits par catégorie."""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Fichier image invalide"}

    # Détection des objets avec YOLO
    results = detector(image, conf=0.09)
    sharpness_threshold = 100  
    category_counts = {}  # Dictionnaire pour stocker les comptes

    for result in results:
        if not result.boxes:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_product = image[y1:y2, x1:x2]
            class_id = int(box.cls[0])  

            if class_id == 39:  # Vérifier si l'objet détecté est une bouteille
                sharpness = calculate_sharpness(cropped_product)
                if sharpness >= sharpness_threshold:
                    # Classification avec SVM
                    category = classify_product(cropped_product)
                    
                    # Mise à jour du dictionnaire de comptage
                    category_counts[category] = category_counts.get(category, 0) + 1

    return category_counts

# 📌 Exemple d'utilisation
if __name__ == "__main__":
    image_path = "3.png"
    result = process_image(image_path)
    print(result)
