from flask import Flask, request
from flask_cors import CORS
import model
import json
# logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
cors = CORS(app)

@app.route("/audio_endpoint", methods=['POST'])
def audio_endpoint():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save('storage/' + uploaded_file.filename)

    results = model.run_model('storage/' + uploaded_file.filename)

    print(results)

    return json.dumps(results), "200"

app.run(threaded=False)
