from flask import Flask, request, jsonify
import keras_ocr
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is provided
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    # Read the image file
    image_file = request.files['image'].read()
    image = Image.open(io.BytesIO(image_file)).convert('RGB')

    # Convert the image to a numpy array and process it with keras-ocr
    image_array = np.array(image)
    prediction_groups = pipeline.recognize([image_array])

    # Extract predicted text
    predictions = prediction_groups[0]
    predicted_text = [word for word, box in predictions]
    
    # Return the predictions as JSON
    return jsonify({"predicted_text": predicted_text})

if __name__ == '__main__':
    app.run(debug=True)
