from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)

# Core Model Integration
model = tf.keras.models.load_model('best_model.h5')
with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# HTML Template for Browser Testing
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Intel Image Classification Testing Server</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; border: 1px solid #ccc; border-radius: 10px; }
        h2 { color: #333; text-align: center; }
        input[type=file] { margin: 20px 0; width: 100%; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; width: 100%; font-size: 16px; }
        button:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <h2>AI Scene Classifier</h2>
    <p>Select a scenic image to test the model on localhost:</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Upload and Predict</button>
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    # Renders the upload form visually when visiting http://localhost:5000
    return render_template_string(HTML_FORM)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file element detected in network request'}), 400
        
    file = request.files['file']
    try:
        # Standardize matrix inputs to match CNN configurations (150x150)
        img = tf.keras.utils.load_img(io.BytesIO(file.read()), target_size=(150, 150))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Array classification through loaded network paths
        predictions = model.predict(img_array)
        predicted_class = labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        return jsonify({
            'status': 'success',
            'prediction': predicted_class,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': f'Inference Failure: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)