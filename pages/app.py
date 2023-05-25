import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load the pre-trained emotion detection model
model = load_model('emotion_detection_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Set the upload folder for Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Process uploaded file
@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']

    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Save the file to the upload folder
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Process the uploaded image
        img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename), target_size=(48, 48), grayscale=True)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Perform emotion detection inference
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_emotion = emotion_labels[predicted_index]

        # Remove the uploaded file
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return f'Emotion detected: {predicted_emotion}'

    return 'Invalid file!', 400

if __name__ == '__main__':
    app.run(debug=True)
