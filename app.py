# app.py
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to the input size required by the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Preprocess the image to the format required by the model
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

@app.route('/')
def index():
    return 'Welcome to the Image Classification Flask App!'

@app.route('/favicon.ico')
def favicon():
    # Handle favicon request (return an empty response)
    return '', 204

@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the uploaded image file from the request
    uploaded_image = request.files['image']
    
    # Read the image file and convert it to a PIL Image object
    image = Image.open(BytesIO(uploaded_image.read()))
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Expand the dimensions of the image array to match the input shape required by the model
    input_image = np.expand_dims(preprocessed_image, axis=0)
    
    # Use the pre-trained model to predict the class probabilities
    predictions = model.predict(input_image)
    
    # Decode the predictions to get the class labels
    decoded_predictions = tf.keras.applications.imagenet_utils.decode_predictions(predictions, top=1)[0]
    
    # Extract the class label and confidence score from the decoded predictions
    class_label = decoded_predictions[0][1]
    confidence_score = float(decoded_predictions[0][2])
    
    # Prepare the result to be returned
    result = {'class': class_label, 'confidence': confidence_score}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
