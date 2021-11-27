import flask
from flask import request
from flask.templating import render_template
import numpy as np
import pickle
from io import BytesIO
import base64

app = flask.Flask(__name__, template_folder='templates')

model = pickle.load(open('model/model_linreg.pkl', 'rb'))

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI IKG
    '''
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return flask.render_template('main.html', prediction_text='The cab price is ${}'.format(output))

@app.route('/')
def main():
    return(flask.render_template('main.html'))

if __name__ == '__main__':
    app.run(debug=True)

    