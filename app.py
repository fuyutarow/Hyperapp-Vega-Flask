import os
import sys
import time
import json
import threading
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras.models import load_model

from flask import Flask, render_template, make_response, jsonify, send_from_directory, redirect
from flask_cors import CORS, cross_origin

from tuner import load_data

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


def crossdomain(origin=None,
                methods=None,
                headers=None,
                max_age=21600,
                attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):

        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)

    return decorator


class Master:

    def __init__(self, model_file):
        self.model = load_model(model_file)

    def predict(self, file):
        img = load_data.arr_fromf(file, rescale=1. / 255, resize=96)
        xs = np.array([img])
        y_pred = self.model.predict(xs)
        y_pred = np.ravel(y_pred)
        return y_pred

    def dump_fig(self, filename):
        img_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', img_file)
        pred = self.predict(img_file)
        df = pd.DataFrame(pred)
        df = df.T
        df.columns = [
            'red_finch',
            'red_parrot',
            'white_finch',
            'white_parrot',
            'yellow_finch',
            'yellow_parrot',
        ]
        plt.style.use('ggplot')
        plt.figure()
        plt.bar(
            range(1, 1 + df.size),
            height=np.ravel(df.values).tolist(),
            tick_label=df.columns.values.tolist())
        timestamp = int(time.time())
        savefile = f'uploads/demo.{timestamp}.png'
        plt.savefig(savefile)
        return savefile

    def vega(self, file):
        y_pred = self.predict(file)

        i2l = {
            0: 'red_finch',
            1: 'red_parrot',
            2: 'white_finch',
            3: 'white_parrot',
            4: 'yellow_finch',
            5: 'yellow_parrot',
        }
        l2v = {i2l[i]: float(v * 100) for i, v in enumerate(y_pred.tolist())}
        values = [{'label': str(label), 'probability': value} for label, value in list(l2v.items())]
        vega_format = {
            "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
            "description": "A simple bar chart with embedded data.",
            "data": {
                "values": values
            },
            "mark": "bar",
            "encoding": {
                "x": {
                    "field": "label",
                    "type": "ordinal"
                },
                "y": {
                    "field": "probability",
                    "type": "quantitative"
                }
            }
        }

        return vega_format

    def set_file(self, filename):
        self.last_file = filename


master = Master("tuner.1517298006.model.hdf5")

app = Flask(__name__, static_folder='birds-dataset')
app.config['ROOT_DIR'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['ROOT_DIR'], 'uploads')
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif', 'JPG'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/viz/<filename>')
@crossdomain(origin='http://localhost:1234')
def viz(filename):
    save_file  = master.dump_fig(filename)
    return render_template('viz.html', img=f'/uploads/{filename}', fig=f'/{save_file}' )


@app.route('/upload', methods=['POST'])
@crossdomain(origin='http://localhost:1234')
def upload():
    uploaded_files = request.files.getlist("file[]")
    res = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
    master.set_file(filename)
    return redirect('/viz/' + filename)
    #return redirect('http://localhost:1234/vega')


@app.route('/uploads/<filename>')
@crossdomain(origin='http://localhost:1234')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/test')
@crossdomain(origin='http://localhost:1234')
def test():
    img_file = 'birds-dataset/red_finch/100.jpg'
    vega = master.vega(img_file)
    vega_data = master.vega(img_file)
    return make_response(jsonify(vega_data))


@app.route('/last-vega')
@crossdomain(origin='http://localhost:1234')
def last_vega():
    filename = master.last_file
    img_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(img_file)
    res = {
        'vega': master.vega(img_file),
        'filename': filename,
    }
    return make_response(jsonify(res))


@app.route('/vega/<filename>')
@crossdomain(origin='http://localhost:1234')
def vega(filename):
    img_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    vega_data = master.vega(img_file)
    return make_response(jsonify(vega_data))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', '-p', type=int, default=4567)
    args = parser.parse_args()

    app.run(port=args.port)
