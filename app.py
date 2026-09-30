from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
PUBLIC_DIR = BASE_DIR / "public"

# Load cascade files directly from the project
face_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_frontalface_default.xml")
)

eye_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_eye.xml")
)

# Make sure the cascade files loaded correctly
if face_cascade.empty():
    raise RuntimeError("Face cascade file could not be loaded")

if eye_cascade.empty():
    raise RuntimeError("Eye cascade file could not be loaded")


@app.route("/")
def home():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
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

    result = image.copy()
    total_eyes = 0

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

    success, buffer = cv2.imencode(
        ".png",
        result
    )

    if not success:
        return jsonify({
            "error": "Could not process image"
        }), 500

    result_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return jsonify({
        "faces_detected": len(faces),
        "eyes_detected": total_eyes,
        "image": result_base64
    })
