from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from pathlib import Path
import mediapipe as mp

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent

# Face detector
face_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_frontalface_default.xml")
)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3
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

    image_array = np.frombuffer(image_bytes, np.uint8)

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
    eyes_detected = 0

    # Draw face boxes
    for (x, y, w, h) in faces:

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Add padding around face
        padding = int(0.15 * max(w, h))

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Enlarge small face regions
        scale = max(2.5, 300 / max(face_crop.shape[:2]))

        enlarged = cv2.resize(
            face_crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        rgb_face = cv2.cvtColor(
            enlarged,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_face)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        face_width = enlarged.shape[1]
        face_height = enlarged.shape[0]

        # MediaPipe eye landmark groups
        left_eye_points = [
            33, 133, 159, 145
        ]

        right_eye_points = [
            362, 263, 386, 374
        ]

        for eye_points in [
            left_eye_points,
            right_eye_points
        ]:

            points = []

            for index in eye_points:

                landmark = landmarks[index]

                px = int(
                    landmark.x * face_width
                )

                py = int(
                    landmark.y * face_height
                )

                points.append((px, py))

            if not points:
                continue

            min_x = min(p[0] for p in points)
            max_x = max(p[0] for p in points)

            min_y = min(p[1] for p in points)
            max_y = max(p[1] for p in points)

            # Expand eye box slightly
            eye_padding_x = max(
                8,
                int((max_x - min_x) * 0.7)
            )

            eye_padding_y = max(
                6,
                int((max_y - min_y) * 1.2)
            )

            min_x -= eye_padding_x
            max_x += eye_padding_x

            min_y -= eye_padding_y
            max_y += eye_padding_y

            # Convert enlarged coordinates
            # back to original image coordinates
            original_x1 = int(
                x1 + min_x / scale
            )

            original_y1 = int(
                y1 + min_y / scale
            )

            original_x2 = int(
                x1 + max_x / scale
            )

            original_y2 = int(
                y1 + max_y / scale
            )

            # Keep coordinates inside image
            original_x1 = max(
                0,
                original_x1
            )

            original_y1 = max(
                0,
                original_y1
            )

            original_x2 = min(
                image.shape[1],
                original_x2
            )

            original_y2 = min(
                image.shape[0],
                original_y2
            )

            # Draw eye box
            cv2.rectangle(
                image,
                (original_x1, original_y1),
                (original_x2, original_y2),
                (255, 0, 0),
                2
            )

            eyes_detected += 1

    encoded_image = image_to_base64(image)

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
