from flask import Flask, request, jsonify
from engine import NERProcessingEngine

app = Flask(__name__)
try:
    engine = NERProcessingEngine()
except Exception as e:
    print(e)


@app.route("/")
def home():
    return "Welcome to the Entity Extraction API!"


@app.route("/endpoint", methods=["POST"])
def extract_entities():
    # try:
    #     data = request.get_json()
    # except Exception as e:
    #     return jsonify({"error": "Invalid JSON"}), 400

    if request.content_type == "application/octet-stream":
        data = request.data
        text = data.decode("utf-8")
    elif request.content_type == "text/markdown":
        data = request.data
        text = data.decode("utf-8")
    else:
        data = request.get_json()
        text = data.get("text", "")
    # text = data.get("text", text)
    engine.pipeline(text)

    entities = {"entities": engine.entities}  # Replace with actual entity extraction
    return jsonify(entities)


if __name__ == "__main__":
    app.run(debug=True)
