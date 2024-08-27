from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
import io
from metrics import dice_loss, dice_coef

app = Flask(__name__)
CORS(app)

# UNET Configuration
image_size = (256, 256)

# Load the model
model_path = os.path.join("model.keras")
model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})

def preprocess_image(image):
    # Preprocess the image
    image = cv2.resize(image, image_size)  # Resize image to (256, 256)
    x = image / 255.0  # Normalize image
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files['image']

        # Check if the image is valid
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # Preprocess the image
        x = preprocess_image(image)

        # Prediction
        pred = model.predict(x, verbose=0)[0]

        # Ensure the prediction is in the correct shape and format if needed
        pred = np.squeeze(pred)  # Remove batch dimension if necessary
        pred = cv2.resize(pred, image_size)  # Resize prediction to match input size if necessary

        # Convert the result image to bytes
        _, buffer = cv2.imencode('.png', pred * 255)
        result_image_bytes = buffer.tobytes()

        return send_file(io.BytesIO(result_image_bytes), mimetype='image/png')

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Enable debug mode for debugger pin
    app.run(port=5002, debug=True)