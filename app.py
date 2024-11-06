import os
import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Periksa apakah file model ada dan muat model
model_path = 'model/Random_Forest.pkl'
if os.path.exists(model_path):
    print("Model file ditemukan.")
    model = joblib.load(model_path)  # Muat model
    print("Model berhasil dimuat.")
else:
    print("Model file tidak ditemukan.")
    model = None

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model tidak tersedia.", 500
    
    # Mengambil input dari pengguna
    try:
        a = int(request.form['a'])
        b = int(request.form['b'])
        c = int(request.form['c'])
        
        # Membuat array numpy untuk prediksi
        input_features = np.array([[a, b, c]])
        
        # Melakukan prediksi
        prediction = model.predict(input_features)
        return render_template('index.html', prediction=prediction[0])
    
    except ValueError as e:
        return f"Input tidak valid: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)
