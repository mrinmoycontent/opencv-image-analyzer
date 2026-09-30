from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
from pathlib import Path
import mediapipe as mp

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
PUBLIC_DIR = BASE_DIR / "public"

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_frontalface_default.xml")
)

if face_cascade.empty():
    raise RuntimeError("Face cascade file could not be loaded")


# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


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

    # --------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------

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

    # --------------------------------------------------
    # EYE LANDMARK DETECTION
    # --------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mesh_result = face_mesh.process(rgb_image)

    eyes_detected = 0

    if mesh_result.multi_face_landmarks:

        for face_landmarks in mesh_result.multi_face_landmarks:

            height, width = image.shape[:2]

            # Left eye landmarks
            left_eye_indices = [
                33, 133, 159, 145
            ]

            # Right eye landmarks
            right_eye_indices = [
                362, 263, 386, 374
            ]

            # Draw left eye
            left_points = []

            for index in left_eye_indices:

                landmark = face_landmarks.landmark[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                left_points.append((x, y))

            if left_points:

                x_values = [p[0] for p in left_points]
                y_values = [p[1] for p in left_points]

                x1 = max(min(x_values) - 5, 0)
                y1 = max(min(y_values) - 5, 0)
                x2 = min(max(x_values) + 5, width - 1)
                y2 = min(max(y_values) + 5, height - 1)

                cv2.rectangle(
                    result,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                eyes_detected += 1

            # Draw right eye
            right_points = []

            for index in right_eye_indices:

                landmark = face_landmarks.landmark[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                right_points.append((x, y))

            if right_points:

                x_values = [p[0] for p in right_points]
                y_values = [p[1] for p in right_points]

                x1 = max(min(x_values) - 5, 0)
                y1 = max(min(y_values) - 5, 0)
                x2 = min(max(x_values) + 5, width - 1)
                y2 = min(max(y_values) + 5, height - 1)

                cv2.rectangle(
                    result,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                eyes_detected += 1

    # --------------------------------------------------
    # ENCODE RESULT
    # --------------------------------------------------

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
        "eyes_detected": eyes_detected,
        "image": result_base64
    })
