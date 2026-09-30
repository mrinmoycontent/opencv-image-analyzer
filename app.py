from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(
    BASE_DIR / "face_detection_yunet_2023mar.onnx"
)

# YuNet face detector
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
        "message": "OpenCV YuNet Image Analyzer API"
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

    height, width = image.shape[:2]

    # Tell YuNet the actual image size
    face_detector.setInputSize(
        (width, height)
    )

    _, detections = face_detector.detect(
        image
    )

    faces_detected = 0
    eyes_detected = 0

    if detections is not None:

        for detection in detections:

            confidence = float(
    detection[4]
)
            )

            if confidence < 0.6:
                continue

            faces_detected += 1

            # --------------------------------
            # Face bounding box
            # --------------------------------

            x = int(detection[0])
            y = int(detection[1])
            w = int(detection[2])
            h = int(detection[3])

            # Keep inside image
            x = max(0, x)
            y = max(0, y)

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

            # --------------------------------
            # YuNet eye landmarks
            #
            # detection:
            #
            # 0-3   face box
            # 4     confidence
            # 5-6   right eye
            # 7-8   left eye
            # 9-10  nose
            # 11-12 right mouth
            # 13-14 left mouth
            # --------------------------------

            right_eye_x = int(
                detection[5]
            )

            right_eye_y = int(
                detection[6]
            )

            left_eye_x = int(
                detection[7]
            )

            left_eye_y = int(
                detection[8]
            )

            # Eye box dimensions relative
            # to detected face size
            eye_width = max(
                12,
                int(w * 0.16)
            )

            eye_height = max(
                8,
                int(h * 0.10)
            )

            half_w = eye_width // 2
            half_h = eye_height // 2

            # --------------------------------
            # Right eye box
            # --------------------------------

            rx1 = max(
                0,
                right_eye_x - half_w
            )

            ry1 = max(
                0,
                right_eye_y - half_h
            )

            rx2 = min(
                width,
                right_eye_x + half_w
            )

            ry2 = min(
                height,
                right_eye_y + half_h
            )

            cv2.rectangle(
                image,
                (rx1, ry1),
                (rx2, ry2),
                (255, 0, 0),
                2
            )

            # --------------------------------
            # Left eye box
            # --------------------------------

            lx1 = max(
                0,
                left_eye_x - half_w
            )

            ly1 = max(
                0,
                left_eye_y - half_h
            )

            lx2 = min(
                width,
                left_eye_x + half_w
            )

            ly2 = min(
                height,
                left_eye_y + half_h
            )

            cv2.rectangle(
                image,
                (lx1, ly1),
                (lx2, ly2),
                (255, 0, 0),
                2
            )

            eyes_detected += 2

    encoded_image = image_to_base64(
        image
    )

    if encoded_image is None:

        return jsonify({
            "error": "Could not encode image"
        }), 500

    return jsonify({
        "faces_detected": faces_detected,
        "eyes_detected": eyes_detected,
        "image": encoded_image
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )
