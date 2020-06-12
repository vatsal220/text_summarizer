from flask import Flask, request, jsonify, render_template
import torch
model = torch.load('summarizer_model.pt')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    '''
    This will output the summary of a text given the input
    '''
    
    body = str(request.form['Body'])

    result = model(body, min_length=60)
    output = ''.join(result)

    return render_template('index.html', prediction_text='The Summarized Text : {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''

    data = request.get_json(force=True)
    output = model(data, min_length = 60)
    
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)

    
    
