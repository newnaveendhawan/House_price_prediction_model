# flask, scikit-learn, pandas, pickle-mixin, flask_cors
import os
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
import pickle
import numpy as np

app = Flask(__name__, static_folder='static')
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))
    bath = int(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    input_data = pd.DataFrame([[location, sqft, bath, bhk]], 
                              columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0] * 1e5
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
