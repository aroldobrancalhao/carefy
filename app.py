from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Carregue o modelo
model = joblib.load("model.pkl")

# Dicionário de mapeamento de classes para nomes das plantas
class_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = request.form['sepal_length'].replace(',', '.')
        sepal_width = request.form['sepal_width'].replace(',', '.')
        petal_length = request.form['petal_length'].replace(',', '.')
        petal_width = request.form['petal_width'].replace(',', '.')

        sepal_length = float(sepal_length)
        sepal_width = float(sepal_width)
        petal_length = float(petal_length)
        petal_width = float(petal_width)
        
        features = [sepal_length, sepal_width, petal_length, petal_width]
        prediction_index = model.predict([features])[0]
        prediction_name = class_names.get(prediction_index, 'Desconhecida')
        
        return render_template('result.html', prediction=prediction_name)
    except Exception as e:
        return f'Erro: {str(e)}'

if __name__ == "__main__":
    app.run(debug=True)
