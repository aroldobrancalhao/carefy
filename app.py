from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Caminho relativo para o modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def home():
    return "Bem-vindo ao serviço de previsão!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = data['features']
        prediction = model.predict([features])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
