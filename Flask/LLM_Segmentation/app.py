from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
from patchify import patchify
import matplotlib.pyplot as plt
import io
from metrics import dice_loss,dice_coef

app = Flask(__name__)
CORS(app)

# UNETR Configuration
cf = {
    "image_size": 256,
    "num_channels": 3,
    "num_layers": 12,
    "hidden_dim": 128,
    "mlp_dim": 32,
    "num_heads": 6,
    "dropout_rate": 0.1,
    "patch_size": 16,
    "num_patches": (256 ** 2) // (16 ** 2),
    "flat_patches_shape": ((256 ** 2) // (16 ** 2), 16 * 16 * 3),
}

# Load the model
model_path = os.path.join("model.keras")
model = tf.keras.models.load_model(model_path, custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    # Check if the image is valid
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # Preprocess the image
    image = cv2.resize(image, (cf["image_size"], cf["image_size"]))
    x = image / 255.0

    patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
    patches = patchify(x, patch_shape, cf["patch_size"])
    patches = np.reshape(patches, cf["flat_patches_shape"])
    patches = patches.astype(np.float32)
    patches = np.expand_dims(patches, axis=0)

    # Prediction
    pred = model.predict(patches, verbose=0)[0]
    pred = np.concatenate([pred, pred, pred], axis=-1)

    # Save final mask
    save_image_path = "prediction.png"
    cv2.imwrite(save_image_path, pred * 255)

    # Convert the result image to bytes
    _, buffer = cv2.imencode('.png', pred * 255)
    result_image = buffer.tobytes()

    # Return the image as a response
    return result_image, 200, {'Content-Type': 'image/png'}

if __name__ == '__main__':
    app.run(port=5001)
