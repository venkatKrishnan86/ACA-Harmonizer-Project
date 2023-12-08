import flask
from flask import Flask, request
from flask_cors import CORS
import json
import pickle
import subprocess
from werkzeug.wrappers import Response
from requests_toolbelt.multipart.encoder import MultipartEncoder
# logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
cors = CORS(app)

@app.route("/run_model", methods=['POST'])
def run_model():
    print ("run_model")
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save('storage/' + uploaded_file.filename)

    proc = subprocess.Popen(['python3', 'inference.py', 'web-app/storage/' + uploaded_file.filename], cwd='../')
    proc.wait()
    
    chord_array = pickle.load(open("../audios/timestamps.pickle", 'rb'))
    print ("Processed")
    return json.dumps(chord_array), "200"

    # data = MultipartEncoder(
    #     fields={
    #         'file': ('filename', open('../audios/sample.wav', 'rb')),
    #         'data': ('', json.dumps(chord_array), 'application/json')
    #     }
    # )
    

@app.route("/audio", methods=['GET'])
def fetch_cached_audio():
    return flask.send_file('../audios/sample.wav'), "200"

app.run(threaded=False)
