# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:05:28 2024

@author: admin
"""


from flask import Flask, request, jsonify
from joblib import load
from skimage.transform import resize
import numpy as np
from PIL import Image
from skimage import color, exposure, filters, feature, measure

app = Flask(__name__)
model = load('C:/Users/admin/Downloads/dataset_full/random_forest_model.pkl')
pca = load('C:/Users/admin/Downloads/dataset_full/pca.pkl')  # Load PCA object


class_names = ['Building', 'Forest', 'Glacier', 'Mountains', 'Sea', 'Street']
def preprocess_image(image):
    # Resize image to 128x128
    image_resized = resize(image, (128, 128))

    # Convert image to numpy array
    img_array = np.array(image_resized)

    # Convert to grayscale
    gray_image = color.rgb2gray(img_array)

    # Low-level Vision: Histogram
    hist, _ = np.histogram(gray_image, bins=256, range=(0, 1))

    # Low-level Vision: Histogram equalization
    equalized_image = exposure.equalize_hist(gray_image)
    hist_eq, _ = np.histogram(equalized_image, bins=256, range=(0, 1))

    # Low-level Vision: Edge Detection using Sobel
    edges_sobel = filters.sobel(gray_image)
    hist_sobel, _ = np.histogram(edges_sobel, bins=256, range=(0, 1))

    # Mid-level Vision: Canny Edge Detection
    edges_canny = feature.canny(gray_image)
    hist_canny, _ = np.histogram(edges_canny, bins=256, range=(0, 1))

    # Connected Components
    labeled_image = measure.label(edges_canny)
    regions = measure.regionprops(labeled_image)
    
    features = np.concatenate([hist, hist_eq, hist_sobel, hist_canny, [len(regions)]])
    return features




@app.route('/')
def upload_form():
    return render_template('upload.html')
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        img = Image.open(file)
        img = np.array(img)  # Convert PIL image to numpy array
        features = preprocess_image(img)
        features_pca = pca.transform(features.reshape(1, -1))  # Apply PCA transformation
        prediction_proba = model.predict_proba(features_pca)[0]  # Get prediction probabilities
        predicted_class_index = np.argmax(prediction_proba)  # Get the index of the highest probability
        predicted_class_name = class_names[predicted_class_index]  # Get the corresponding class name

        return jsonify({
            'prediction': prediction_proba.tolist(),
            'predicted_class': predicted_class_name
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)