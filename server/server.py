from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/get_data")
def get_data():
    results = "yoooo"
    return results

if __name__== "__main__":
    app.run(debug=True)
