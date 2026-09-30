from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(
    BASE_DIR / "face_detection_yunet_2023mar.onnx"
)


# --------------------------------------------------
# YuNet Face Detector
# --------------------------------------------------

face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

if face_detector is None:
    raise RuntimeError(
        "YuNet face detector could not be loaded"
    )


# --------------------------------------------------
# Convert image to Base64
# --------------------------------------------------

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


# --------------------------------------------------
# Frontend
# --------------------------------------------------

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR / "public",
        "index.html"
    )


# --------------------------------------------------
# Analyze Image
# --------------------------------------------------

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

    height, width = image.shape[:2]

    # Set actual image size for YuNet

    face_detector.setInputSize(
        (width, height)
    )

    # Detect faces

    _, detections = face_detector.detect(
        image
    )

    faces_detected = 0
    eyes_detected = 0

    # --------------------------------------------------
    # Process detections
    # --------------------------------------------------

    if detections is not None:

        for detection in detections:

            # YuNet output:
            #
            # 0  = face X
            # 1  = face Y
            # 2  = face width
            # 3  = face height
            #
            # 4  = right eye X
            # 5  = right eye Y
            #
            # 6  = left eye X
            # 7  = left eye Y
            #
            # 8  = nose X
            # 9  = nose Y
            #
            # 10 = right mouth X
            # 11 = right mouth Y
            #
            # 12 = left mouth X
            # 13 = left mouth Y
            #
            # 14 = confidence

            confidence = float(
                detection[14]
            )

            if confidence < 0.6:
                continue

            faces_detected += 1

            # --------------------------------------------------
            # Face box
            # --------------------------------------------------

            x = int(detection[0])
            y = int(detection[1])
            w = int(detection[2])
            h = int(detection[3])

            x = max(
                0,
                x
            )

            y = max(
                0,
                y
            )

            w = min(
                w,
                width - x
            )

            h = min(
                h,
                height - y
            )

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # --------------------------------------------------
            # Eye landmarks
            # --------------------------------------------------

            right_eye_x = int(
                detection[4]
            )

            right_eye_y = int(
                detection[5]
            )

            left_eye_x = int(
                detection[6]
            )

            left_eye_y = int(
                detection[7]
            )

            # --------------------------------------------------
            # Eye box size
            # --------------------------------------------------

            eye_width = max(
                12,
                int(w * 0.14)
            )

            eye_height = max(
                8,
                int(h * 0.10)
            )

            half_width = eye_width // 2
            half_height = eye_height // 2

            # --------------------------------------------------
            # Right eye box
            # --------------------------------------------------

            rx1 = max(
                0,
                right_eye_x - half_width
            )

            ry1 = max(
                0,
                right_eye_y - half_height
            )

            rx2 = min(
                width - 1,
                right_eye_x + half_width
            )

            ry2 = min(
                height - 1,
                right_eye_y + half_height
            )

            cv2.rectangle(
                image,
                (rx1, ry1),
                (rx2, ry2),
                (255, 0, 0),
                2
            )

            # --------------------------------------------------
            # Left eye box
            # --------------------------------------------------

            lx1 = max(
                0,
                left_eye_x - half_width
            )

            ly1 = max(
                0,
                left_eye_y - half_height
            )

            lx2 = min(
                width - 1,
                left_eye_x + half_width
            )

            ly2 = min(
                height - 1,
                left_eye_y + half_height
            )

            cv2.rectangle(
                image,
                (lx1, ly1),
                (lx2, ly2),
                (255, 0, 0),
                2
            )

            eyes_detected += 2

    # --------------------------------------------------
    # Encode processed image
    # --------------------------------------------------

    encoded_image = image_to_base64(
        image
    )

    if encoded_image is None:

        return jsonify({
            "error": "Could not encode image"
        }), 500

    # --------------------------------------------------
    # Return result
    # --------------------------------------------------

    return jsonify({
        "faces_detected": faces_detected,
        "eyes_detected": eyes_detected,
        "image": encoded_image
    })


# --------------------------------------------------
# Local Flask Server
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
