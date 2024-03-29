from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the 'uploads' folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model("./model/bestmodel.keras")

# Define preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define prediction function
def predict(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(preprocessed_image)
    return prediction[0][0] > 0.5

# Define route for uploading image
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = "Tumor" if predict(file_path) else "No Tumor"
            return render_template('index.html', message='File uploaded successfully', prediction=prediction, filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
