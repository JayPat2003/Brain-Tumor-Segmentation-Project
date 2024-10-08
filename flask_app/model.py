import tensorflow as tf
from PIL import Image
import numpy as np

class InceptionV3BrainTumorModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_path):
        # Load the image
        image = Image.open(image_path)

        # Resize the image to the model's input size
        image = image.resize((299, 299))

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Preprocess the image
        image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)

        # Make a prediction
        prediction = self.model.predict(image_array)

        # Convert the prediction probabilities to a class prediction
        class_prediction = np.argmax(prediction)

        # Return the class prediction
        return class_prediction

