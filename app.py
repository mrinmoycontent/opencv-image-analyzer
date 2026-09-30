from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

# OpenCV built-in cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

if face_cascade.empty():
    raise RuntimeError("Face cascade could not be loaded")

if eye_cascade.empty():
    raise RuntimeError("Eye cascade could not be loaded")


def image_to_base64(image):

    success, buffer = cv2.imencode(
        ".jpg",
        image
    )

    if not success:
        return None

    return base64.b64encode(
        buffer
    ).decode("utf-8")


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

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(25, 25)
    )

    faces_detected = len(faces)

    eyes_detected = 0

    # Process every detected face
    for (x, y, w, h) in faces:

        # Face rectangle
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ------------------------------------------------
        # IMPORTANT:
        # Search only the upper part of the face.
        # This prevents nose/mouth/cheek detections.
        # ------------------------------------------------

        upper_height = int(h * 0.58)

        face_gray = gray[
            y:y + upper_height,
            x:x + w
        ]

        if face_gray.size == 0:
            continue

        # Enlarge small faces
        scale = 4

        enlarged = cv2.resize(
            face_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Detect eyes
        detected_eyes = eye_cascade.detectMultiScale(
            enlarged,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(34, 34),
            maxSize=(120, 80)
        )

        # Sort from left to right
        detected_eyes = sorted(
            detected_eyes,
            key=lambda e: e[0]
        )

        # Keep at most two eye detections
        detected_eyes = detected_eyes[:2]

        for (ex, ey, ew, eh) in detected_eyes:

            # Convert enlarged coordinates
            # back to original image coordinates

            original_x = (
                x +
                int(ex / scale)
            )

            original_y = (
                y +
                int(ey / scale)
            )

            original_w = int(
                ew / scale
            )

            original_h = int(
                eh / scale
            )

            # Draw blue eye box
            cv2.rectangle(
                image,
                (
                    original_x,
                    original_y
                ),
                (
                    original_x +
                    original_w,
                    original_y +
                    original_h
                ),
                (255, 0, 0),
                2
            )

            eyes_detected += 1

    encoded_image = image_to_base64(
        image
    )

    if encoded_image is None:

        return jsonify({
            "error": "Could not encode processed image"
        }), 500

    return jsonify({

        "faces_detected":
            faces_detected,

        "eyes_detected":
            eyes_detected,

        "image":
            encoded_image
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
