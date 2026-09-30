
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "OpenCV Image Analyzer API is running"
    })

@app.route("/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    image_bytes = np.frombuffer(
        file.read(),
        np.uint8
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    total_eyes = 0
    result = image.copy()

    for (x, y, w, h) in faces:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        face_gray = gray[y:y+h, x:x+w]
        face_color = result[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        total_eyes += len(eyes)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

    _, buffer = cv2.imencode(".png", result)

    result_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return jsonify({
        "faces_detected": len(faces),
        "eyes_detected": total_eyes,
        "image": result_base64
    })
