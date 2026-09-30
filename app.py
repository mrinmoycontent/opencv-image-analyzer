from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

face_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_frontalface_default.xml")
)


def image_to_base64(image):
    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "OpenCV Image Analyzer API"
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    image_bytes = file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({
            "error": "Could not read image"
        }), 400

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    faces_detected = len(faces)

    # Draw face boxes
    for (x, y, w, h) in faces:

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    encoded_image = image_to_base64(image)

    return jsonify({
        "faces_detected": faces_detected,
        "eyes_detected": 0,
        "image": encoded_image
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )
