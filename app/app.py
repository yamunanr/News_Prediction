import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import pickle
from Data.preprocess import text_preprocess, preprocess
from model import *

# Declare a Flask app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('../../ML_Flask/model2.pkl', 'rb'))
tfidfvect = pickle.load(open('../../ML_Flask/tfidfvect2.pkl', 'rb'))


# Build functionalities
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


def predict(text):
    # review = re.sub('[^a-zA-Z]', ' ', text)
    #review = text_preprocess(text)
    #review = preprocess(review)
    review_vect = tfidfvect.transform([text]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'TRUE'
    return prediction


@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index.html', text=text, result=prediction)


@app.route('/predict/', methods=['GET', 'POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
