from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)

# Load model and labels globally
model = tf.keras.models.load_model('best_model.h5')
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    try:
        # Load image stream safely at 150x150 pixels
        img = tf.keras.utils.load_img(io.BytesIO(file.read()), target_size=(150, 150))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)