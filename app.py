from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plant_detection.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class PlantAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), default='Anonyme')
    plant_name = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

MODEL_PATH = "model/model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    'aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut',
    'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger', 'guava',
    'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy', 'papaya',
    'peper chili', 'pineapple', 'pomelo', 'shallot', 'soybeans', 'spinach',
    'sweet potatoes', 'tobacco', 'waterapple', 'watermelon'
]

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/history')
def history():
    analyses = PlantAnalysis.query.order_by(PlantAnalysis.timestamp.desc()).all()
    return render_template('history.html', analyses=analyses)

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucune image envoyée'}), 400

    try:
        username = request.form.get('username', 'Anonyme')
        file = request.files['file']
        
        img = Image.open(file).convert('RGB').resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        class_index = np.argmax(predictions)
        result = class_names[class_index]

        new_entry = PlantAnalysis(
            username=username,
            plant_name=result,
            confidence=confidence,
            filename=file.filename
        )
        db.session.add(new_entry)
        db.session.commit()

        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'username': username,
            'image_url': f'/static/uploads/{file.filename}'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)