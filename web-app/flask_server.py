import flask
from flask import Flask, request
from flask_cors import CORS
import json
import pickle
import subprocess
# logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
cors = CORS(app)

@app.route("/run_model", methods=['POST'])
def run_model():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save('storage/' + uploaded_file.filename)

    proc = subprocess.Popen(['python3', 'inference.py', 'web-app/storage/' + uploaded_file.filename], cwd='../')
    proc.wait()
    chord_array = pickle.load(open("../audios/timestamps.pickle", 'rb'))
    return json.dumps(chord_array), "200"

@app.route("/fetch_cached_audio", methods=['GET'])
def fetch_cached_audio():
    return flask.send_file('../audios/sample.wav'), "200"

app.run(threaded=False)
