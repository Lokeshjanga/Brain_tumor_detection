from flask import Flask, request, jsonify
from keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your model
saved_model = load_model("vggl (2).keras")

# Path for uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict(img):
    img = img.resize((224, 224))  # Resize the image to match the model input size
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    output = saved_model.predict(img_array)
    if output[0][0] > output[0][1]:
        return "No"
    else:
        return "Yes"
@app.route('/',methods=['GET'])
def home():
    return "hello welcome"

@app.route('/predict', methods=['POST'])
def upload_file():
    print("Request received")  # Debugging line
    # Check if the post request has the file part
    image_data = request.files['image']

    image = Image.open(image_data).convert('RGB')

    




    
    if image:
        try:
            
            prediction = predict(image)
            return jsonify({'prediction': prediction})
        except Exception as e:
            print(f"Error processing the file: {e}")  # Debugging line
            return jsonify({'error': 'Error processing the file'}), 400
    else:
        print("File type not allowed")  # Debugging line
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg, gif'}), 400

if __name__ == '__main__':
    app.run(debug=True,port=5000)
