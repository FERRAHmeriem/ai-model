import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


DATASET_PATH = "./final_data"

feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_deep_features(img_path):
    """Extract features from an image using ResNet50."""
    img = image.load_img(img_path, target_size=(224, 224))  
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = feature_extractor.predict(img_data)
    return features.flatten()

X, y = [], []
categories = os.listdir(DATASET_PATH)
label_map = {category: idx for idx, category in enumerate(categories)}

for category in categories:
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(category_path): 
        continue

    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        # 🔹 Ensure file exists
        if not os.path.exists(img_path):
            continue
        
        
        try:
            features = extract_deep_features(img_path)
            X.append(features)
            y.append(label_map[category])
        except Exception as e:
            continue


X = np.array(X)
y = np.array(y)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']}
svm = GridSearchCV(SVC(), param_grid, cv=5)
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Best SVM Parameters: {svm.best_params_}")
print(f"Final SVM Accuracy: {accuracy * 100:.2f}%")


with open("label_map.pkl", "wb") as file:
    pickle.dump(label_map, file)


with open("svm_model.pkl", "wb") as file:
    pickle.dump(svm, file)


with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
