from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(request.form[field]) for field in ['Pregnancies', 'Glucose', 'BloodPressure',
                                                      'SkinThickness', 'Insulin', 'BMI',
                                                      'DiabetesPedigreeFunction', 'Age']]
    prediction = model.predict([data])[0]
    result_text = "The model predicts the patient is  diabetic." if prediction == 1 else "The model predicts the patient is not diabetic."
    return render_template('result.html', result_text=result_text,prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
