from time import strftime

import flask
import logging
from logging.handlers import RotatingFileHandler
import os

import pandas as pd
import dill
dill._dill._reverse_typemap['ClassType'] = type

start_dt = strftime("[%Y-%b-%d %H:%M:%S]")

log_path = '/app/log'
model_path = '/app/app/models' # для работы из контейнера


handler = RotatingFileHandler(filename=log_path + '/app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

#model_name = 'RandomForestClassifier' # лучшая модель, но не хватает памяти на локальном virtualbox-е
model_name = 'LogisticRegression'
model_file = model_path + '/' + model_name + '.dill'
model = None

def load_model():
	global model
	try:
		with open(model_file, 'rb') as f:
			model = dill.load(f)
	except IOError as e:
		logger.error(f"{start_dt} Error loading '{model_file}': {str(e)}")
		exit(1)
	logger.info(f"{start_dt} Loaded model: " + str(model))



app = flask.Flask(__name__)
model = None
host = "0.0.0.0"
port = 8180


@app.route("/", methods=["GET"])
def general():
	return f"""Welcome! 'http://{host}:{port}/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	if flask.request.method == "POST":
		request_json = flask.request.get_json()
		text = request_json.get('text', '')
		keyword = request_json.get('keyword', '')
		location = request_json.get('location', '')
		try:
			logger.info(f"{dt} Data: text='{text}', location='{location}', keyword='{keyword}'")
			preds = model.predict_proba(pd.DataFrame({"text": [text], "keyword": [keyword], "location": [location]}))
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e) + ' model = ' + str(model)
			data['success'] = False
			return flask.jsonify(data)
		data["predictions"] = preds[:, 1][0]
		logger.info(f"{dt} Data: text='{text}', location='{location}', keyword='{keyword}' - predicted {data['predictions']}")
		data["success"] = True
	return flask.jsonify(data)


if __name__ == "__main__":
	logger.info(f"{strftime('[%Y-%b-%d %H:%M:%S]')} Loading model and starting server ...")
	load_model() # если сюда не перенести вызов загрузки, в predict оказывается model==None
	port = int(os.environ.get('PORT', port))
	app.run(host=host, port=port, debug=False)