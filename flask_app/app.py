from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras import backend as K

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
SEGMENTED_FOLDER = 'segmented'

# Create the directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

def combined_loss(y_true, y_pred):
    """Custom loss function for the model."""
    return K.mean(K.square(y_true - y_pred))  # Example: Mean Squared Error

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate the Dice Coefficient for model evaluation."""
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

# Load the trained UNet model with the custom loss function and metric
model = load_model('atttenunet_brain_tumor_segmentation.keras', custom_objects={
    'combined_loss': combined_loss,
    'dice_coefficient': dice_coefficient
})

def predict_and_segment(image_path):
    """Predict and generate segmented mask for the input image."""
    target_size = (256, 256)  # This should match your model's expected input size
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)  # Resize to 256x256
    img_preprocessed = img_resized.astype('float32') / 255.0
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)  # Add batch dimension

    # Predict the mask
    predicted_mask = model.predict(img_preprocessed)[0]
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255  # Binarize mask

    # Resize mask back to original image size
    predicted_mask_resized = cv2.resize(predicted_mask, (img.shape[1], img.shape[0]))

    # Overlay the mask onto the original image
    segmented_img = img.copy()
    segmented_img[predicted_mask_resized > 0] = [0, 0, 255]  # Highlight tumor in red
    
    return predicted_mask_resized, segmented_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded image
        filename = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        file.save(filename)

        # Predict and generate segmented image
        mask, segmented_img = predict_and_segment(filename)

        # Save the segmented image
        segmented_filename = os.path.join(SEGMENTED_FOLDER, 'segmented_image.jpg')
        cv2.imwrite(segmented_filename, segmented_img)

        # Save the mask image (optional)
        mask_filename = os.path.join(SEGMENTED_FOLDER, 'mask_image.jpg')
        cv2.imwrite(mask_filename, mask)

        # Return the segmented image path
        return jsonify({'segmented_img': 'segmented_image.jpg', 'mask_img': 'mask_image.jpg'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/segmented/<filename>')
def segmented_file(filename):
    return send_from_directory(SEGMENTED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
