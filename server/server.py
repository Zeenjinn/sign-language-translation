# ✅ server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model('sign_model_fixed.h5')
encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('sequence')
    if data is None or len(data) != 30:
        return jsonify({'error': 'Invalid input'}), 400

    input_data = np.expand_dims(np.array(data), axis=0)
    prediction = model.predict(input_data)[0]
    confidence = float(np.max(prediction))
    label = encoder.inverse_transform([np.argmax(prediction)])[0]

    if confidence > 0.8:
        print(f'Prediction: {label}, Confidence: {confidence:.2f}')
        return jsonify({'result': label, 'confidence': confidence})
    else:
        print(f'\u274c Low confidence ({confidence:.2f}) - 무시됨')
        return jsonify({'result': None, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
