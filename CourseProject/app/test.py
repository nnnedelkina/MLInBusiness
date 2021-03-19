import urllib.request
import json
import pandas as pd
import sys
import os

rest_host = "127.0.0.1"
rest_port = 8180

def get_predictions(url, **args):
    req = urllib.request.Request(url)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(args)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    response = urllib.request.urlopen(req, jsondataasbytes)
    if response.status != 200:
        raise Exception(f"Server returned status {response.status}")
    js = json.loads(response.read())
    if not js['success']:
        raise Exception(f"Error: {js['predictions']}")
    return js['predictions']


if __name__ == "__main__":
    data_path = os.path.dirname(sys.argv[0]) + '/data'

    X_test = pd.read_csv(data_path + "/X_test.csv").fillna('') # иначе пустые значения превращаются в nan
    y_test = pd.read_csv(data_path + "/y_test.csv")
    X_test['target'] = y_test
    n_rows = X_test.shape[0]
    for i, row in X_test.iterrows():
        print(f"Запрос №{i}: text='{row['text']}', keyword='{row['keyword']}', location='{row['location']}'")
        preds = get_predictions(f"http://{rest_host}:{rest_port}/predict",
                    text=row['text'], keyword=row['keyword'], location=row['location'])
        print(f"Предсказанное: {preds}, истинное: {row['target']}")
