from time import strftime

import flask
import logging
from logging.handlers import RotatingFileHandler
import os

import numpy as np
import pandas as pd

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


app = flask.Flask(__name__)
model = None
host = "0.0.0.0"
port = 8180


@app.route("/", methods=["GET"])
def general():
	return f"""Добро пожаловать! 'http://{host}:{port}/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	if flask.request.method == "POST":
		request_json = flask.request.get_json()
		description = request_json.get('description', '')
		company_profile = request_json.get('company_profile', '')
		benefits = request_json.get('benefits', '')

		logger.info(f'{dt} Data: description={description}, company_profile={company_profile}, benefits={benefits}')

		try:
#			preds = model.predict_proba(pd.DataFrame({"description": [description],
#												  "company_profile": [company_profile],
#												  "benefits": [benefits]}))

			pass
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = [1, 0.5, 1] # preds[:, 1][0]
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)



if __name__ == "__main__":
	print("* Загружаем модель и запускаем сервер ...")
	port = int(os.environ.get('PORT', port))
	app.run(host=host, debug=True, port=port)