import numpy as np
import networkx as nx
import text_summarizer as ts

from flask import Flask, request, jsonify, render_template
from Levenshtein import jaro
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance

# refrence
# https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarizer', methods = ['GET', 'POST'])
def summarizer():
    return render_template('summarizer.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    '''
    This will output the summary of a text given the input
    '''
    body = str(request.form['Body'])
    sentence_count = int(request.form['sentence_count'])
    result = generate_summary(data = body, top_n = sentence_count)
    return render_template('summarizer.html', prediction_text='{}'.format(result))

@app.route('/ner', methods = ['GET', 'POST'])
def ner():
    return render_template('ner.html')

@app.route('/predict_ner', methods = ['GET', 'POST'])
def predict_ner():
    '''
    This will return the similarity between two strings
    '''

    str1 = str(request.form['string1'])
    str2 = str(request.form['string2'])
    result = jaro(str1, str2)
    return render_template('ner.html', prediction_text = '{}'.format(result))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls through request
#     '''
#
#     data = request.get_json(force=True)
#     output = model(data, min_length = 60)
#
#     return jsonify(output)

if __name__ == "__main__":

    app.run(debug=True)
