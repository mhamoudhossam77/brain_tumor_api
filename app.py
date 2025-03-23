from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# تحميل الموديل
model = load_model("brain_tumor.h5")

# إعداد الصورة حسب متطلبات الموديل
IMG_SIZE = (128, 128)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
        try:
            # Open image and ensure it's in RGB mode (3 channels)
            img = Image.open(file.stream).convert('RGB').resize(IMG_SIZE)

            # Convert the image to an array and normalize it
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            return jsonify({
                'prediction': predicted_class,
                'confidence': confidence
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
