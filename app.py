from flask import Flask, request, render_template
import joblib
import os

app = Flask(__name__)

# Caminho para o modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recebe os dados do formulário
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Verifica se as características têm o tamanho correto
        features = [sepal_length, sepal_width, petal_length, petal_width]
        
        # Faz a previsão
        prediction = model.predict([features])
        
        # Retorna a previsão como uma resposta
        return render_template('result.html', prediction=prediction[0])
    except Exception as e:
        return f'Erro: {str(e)}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
