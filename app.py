from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregar o modelo treinado
model = joblib.load("/home/aroldo/market4u/Carefy/model.pkl")

@app.route('/')
def home():
    return "API está funcionando!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sepal_length = data['sepal_length']
    sepal_width = data['sepal_width']
    petal_length = data['petal_length']
    petal_width = data['petal_width']
    
    # Realizar a previsão
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return jsonify({"species": prediction[0]})

if __name__ == '__main__':
    app.run(port=5001)
