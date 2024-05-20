# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:45:42 2024

@author: admin
"""

import os
import numpy as np
from skimage import io, color, img_as_float, transform
from sklearn.model_selection import train_test_split

data_path = 'C:/Users/admin/Downloads/dataset_full/original'

def count_classes_and_elements(folder):
    class_counts = {}
    class_names = os.listdir(folder)

    for class_name in class_names:
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            class_counts[class_name] = num_images

    return class_counts

class_counts = count_classes_and_elements(data_path)
print("Class Counts:")
for class_name, count in class_counts.items():
    print(f"Class '{class_name}' has {count} images")
    
"""As we can see that there are total 6 classes and the data set 
is imbalanced as all the other classes except Forest contains 500 images so 
we need to perform data augumentation techninque to  create more data to balance 
dataset or we can perform downsampling.... I am going with downsampling because 
of resources(memory, speed) problem
"""

from sklearn.utils import resample

def load_images_from_folder(folder, target_size=(128, 128), target_count=None, random_state=None):
    images = []
    labels = []
    class_names = os.listdir(folder)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            filenames = [filename for filename in os.listdir(class_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if target_count is not None:
                filenames = resample(filenames, n_samples=target_count, random_state=random_state)

            for filename in filenames:
                img_path = os.path.join(class_path, filename)
                if os.path.isfile(img_path):
                    image = io.imread(img_path)
                    image = transform.resize(image, target_size, anti_aliasing=True)
                    image = img_as_float(image)
                    images.append(image)
                    labels.append(label)

    return np.array(images), np.array(labels)

# Example usage
X, y = load_images_from_folder(data_path, target_count=500, random_state=42)
print(X.shape)
print(y.shape)

# Convert images to grayscale
"""
Converting images to grayscale simplifies data, reduces computational load,
 focuses on intensity information, removes redundant color information, and 
 improves the generalization of machine learning models.
"""
def preprocess_images(images):
    gray_images = [color.rgb2gray(image) for image in images]
    return np.array(gray_images)

X_gray = preprocess_images(X)

X_train_gray, X_test_gray, y_train, y_test = train_test_split(X_gray, y, test_size=0.2, random_state=42)
import matplotlib.pyplot as plt
from skimage import exposure, filters, feature, measure
# Function to extract features from a single image and display transformations
def extract_features(image, display=False):
    features = []

    if display:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        ax = axes.ravel()

    # Original image
    if display:
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Original Image")
        ax[0].axis('off')

    # Low-level Vision: Histogram
    hist, _ = np.histogram(image, bins=256, range=(0, 1))
    features.extend(hist)

    # Low-level Vision: Histogram equalization
    equalized_image = exposure.equalize_hist(image)
    if display:
        ax[1].imshow(equalized_image, cmap='gray')
        ax[1].set_title("Histogram Equalization")
        ax[1].axis('off')
    hist_eq, _ = np.histogram(equalized_image, bins=256, range=(0, 1))
    features.extend(hist_eq)

    # Low-level Vision: Edge Detection using Sobel
    edges_sobel = filters.sobel(image)
    if display:
        ax[2].imshow(edges_sobel, cmap='gray')
        ax[2].set_title("Sobel Edge Detection")
        ax[2].axis('off')
    hist_sobel, _ = np.histogram(edges_sobel, bins=256, range=(0, 1))
    features.extend(hist_sobel)

    # Mid-level Vision: Canny Edge Detection
    edges_canny = feature.canny(image)
    if display:
        ax[3].imshow(edges_canny, cmap='gray')
        ax[3].set_title("Canny Edge Detection")
        ax[3].axis('off')
    hist_canny, _ = np.histogram(edges_canny, bins=256, range=(0, 1))
    features.extend(hist_canny)

    # Connected Components
    labeled_image = measure.label(edges_canny)
    regions = measure.regionprops(labeled_image)
    if display:
        ax[4].imshow(labeled_image, cmap='nipy_spectral')
        ax[4].set_title("Connected Components")
        ax[4].axis('off')
    features.extend([len(regions)])

    if display:
        plt.show()

    return np.array(features)


_ = extract_features(X_train_gray[0], display=True)

X_train_features = np.array([extract_features(image) for image in X_train_gray])
X_test_features = np.array([extract_features(image) for image in X_test_gray])

print(f"Shape of X_train_features: {X_train_features.shape}")
print(f"Shape of X_test_features: {X_test_features.shape}")


from sklearn.decomposition import PCA
import joblib

# Applying PCA for dimensionality reduction
pca = PCA(n_components=50)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train_features)
X_test_pca = pca.transform(X_test_features)

joblib.dump(pca, 'C:/Users/admin/Downloads/dataset_full//pca.pkl')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Train Random Forest classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train_pca, y_train)

joblib.dump(clf_rf, 'C:/Users/admin/Downloads/dataset_full/random_forest_model.pkl')


y_pred_rf = clf_rf.predict(X_test_pca)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_rf}")
print(f"Precision: {precision_rf}")
print(f"Recall: {recall_rf}")
print(f"F1 Score: {f1_rf}")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Train k-NN classifier
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train_pca, y_train)

y_pred_knn = clf_knn.predict(X_test_pca)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

print("k-NN Model Evaluation:")
print(f"Accuracy: {accuracy_knn}")
print(f"Precision: {precision_knn}")
print(f"Recall: {recall_knn}")
print(f"F1 Score: {f1_knn}")


