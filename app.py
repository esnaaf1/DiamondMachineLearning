# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from pickle import load
import json

# Initialize the flask App
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load the model from its pickle file. (This pickle
# file was originally saved by the code that trained
# the model. See mlmodel.py)
loaded_classifier = load(open('classifier.pkl', 'rb'))


# Define the index route
@app.route('/')
def home():
	return render_template('diamonds.html')

# Define a route that runs when the user clicks the Predict button in the web-app


@app.route('/predict', methods=['POST'])
def predict():

	args_from_json = json.loads(request.data)

	row = [
		args_from_json["shape"],
		args_from_json["price"],
		args_from_json["carat"],
		args_from_json["cut"],
		args_from_json["color"],
		args_from_json["clarity"],
		args_from_json["report"],
	]
	matrix = [row]

	return loaded_classifier.predict(matrix)[0]
	
	# Create a list of the output labels.
	# prediction_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

	# # Read the list of user-entered values from the website. Note that these
	# # will be strings.
	# features = [x for x in request.form.data()]

	# # Convert each value to a float.
	# float_features = [float(x) for x in features]

	# # Put the list of floats into another list, to make scikit-learn happy.
	# # (This is how scikit-learn wants the data formatted. We touched on this
	# # in class.)
	# final_features = [np.array(float_features)]
	# print(final_features)

	# # Preprocess the input using the ORIGINAL (unpickled) scaler.
	# # This scaler was fit to the TRAINING set when we trained the
	# # model, and we must use that same scaler for our prediction
	# # or we won't get accurate results.
	# final_features_scaled = scaler.transform(final_features)

	# # Use the scaled values to make the prediction.
	# prediction_encoded = randomforest.predict(final_features_scaled)
	# prediction = prediction_labels[prediction_encoded[0]]

	# # Render a template that shows the result.
	# prediction_text = f'Iris flower type is predicted to be :  {prediction}'
	# print (prediction_text)
	# return prediction
# Allow the Flask app to launch from the command line
if __name__ == "__main__":
	app.run(debug=True)
