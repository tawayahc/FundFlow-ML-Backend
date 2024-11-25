import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image

__version__ = "0.1.0"

class AnomalyDetectionModelLoader:
    def __init__(self, target_size=(256, 256)):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, f"anomaly_detection_{__version__}.h5")
        self.target_size = target_size
        self.model = None

    def load_model(self):
        try:
            self.model = load_model(self.model_path, custom_objects={"mse": MeanSquaredError()})
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    def preprocess_image(self, image):
        try:
            image = image.convert("RGB")

            # Resize with padding
            old_size = image.size
            ratio = float(min(self.target_size)) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
            delta_w = self.target_size[0] - new_size[0]
            delta_h = self.target_size[1] - new_size[1]
            padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

            padded_image = Image.new("RGB", self.target_size, (0, 0, 0))  # Add padding to the resized image
            padded_image.paste(resized_image, (padding[0], padding[1]))

            # Normalize the pixel values to range [0, 1]
            cleaned_image = np.array(padded_image) / 255.0
            cleaned_image = np.expand_dims(cleaned_image, axis=0)  # Add batch dimension

            return cleaned_image
        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {e}")

    def predict_anomaly(self, images, threshold=0.015093279607587579):
        if self.model is None:
            raise ValueError("Model is not loaded. Call `load_model` first.")
        
        try:
            normal_images = []
            for image in images:
                processed_image = self.preprocess_image(image)
                reconstructed_image = self.model.predict(processed_image)
                reconstruction_error = np.mean((processed_image - reconstructed_image) ** 2)
                
                if reconstruction_error <= threshold:
                    normal_images.append(image)

            return normal_images
        except Exception as e:
            raise ValueError(f"Prediction failed: {e}")
