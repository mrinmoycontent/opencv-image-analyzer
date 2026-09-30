from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import base64
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
PUBLIC_DIR = BASE_DIR / "public"

# --------------------------------------------------
# LOAD FACE CASCADE
# --------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_frontalface_default.xml")
)

if face_cascade.empty():
    raise RuntimeError(
        "Face cascade file could not be loaded"
    )


# --------------------------------------------------
# LOAD EYE CASCADE
# --------------------------------------------------

eye_cascade = cv2.CascadeClassifier(
    str(BASE_DIR / "haarcascade_eye.xml")
)

if eye_cascade.empty():
    raise RuntimeError(
        "Eye cascade file could not be loaded"
    )


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(
        PUBLIC_DIR,
        "index.html"
    )


# --------------------------------------------------
# IMAGE ANALYSIS
# --------------------------------------------------

@app.route("/api/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

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
        return jsonify({
            "error": "Invalid image"
        }), 400

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    result = image.copy()

    total_eyes = 0

    # --------------------------------------------------
    # PROCESS EACH FACE
    # --------------------------------------------------

    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        # Crop the face
        face_gray = gray[
            y:y + h,
            x:x + w
        ]

        face_color = result[
            y:y + h,
            x:x + w
        ]

        if face_gray.size == 0:
            continue

        # --------------------------------------------------
        # ENLARGE SMALL FACE
        # --------------------------------------------------

        scale = 3

        enlarged_face = cv2.resize(
            face_gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # --------------------------------------------------
        # EYE DETECTION
        # --------------------------------------------------

        eyes = eye_cascade.detectMultiScale(
            enlarged_face,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(15, 15),
            maxSize=(
                int(enlarged_face.shape[1] * 0.45),
                int(enlarged_face.shape[0] * 0.35)
            )
        )

        # --------------------------------------------------
        # REMOVE DUPLICATE / OVERLAPPING DETECTIONS
        # --------------------------------------------------

        detected_eyes = []

        for (ex, ey, ew, eh) in eyes:

            # Convert enlarged coordinates
            # back to original face coordinates
            ex = int(ex / scale)
            ey = int(ey / scale)
            ew = int(ew / scale)
            eh = int(eh / scale)

            # Ignore detections outside face
            if ex < 0 or ey < 0:
                continue

            if ex + ew > w:
                continue

            if ey + eh > h:
                continue

            # Eye center
            center_x = ex + ew // 2
            center_y = ey + eh // 2

            # Eyes should normally be in the
            # upper half of the face
            if center_y > int(h * 0.65):
                continue

            # Avoid duplicate detections
            duplicate = False

            for (
                old_x,
                old_y,
                old_w,
                old_h
            ) in detected_eyes:

                old_center_x = (
                    old_x + old_w // 2
                )

                old_center_y = (
                    old_y + old_h // 2
                )

                distance_x = abs(
                    center_x - old_center_x
                )

                distance_y = abs(
                    center_y - old_center_y
                )

                if (
                    distance_x < max(ew, old_w) * 0.5
                    and
                    distance_y < max(eh, old_h) * 0.5
                ):
                    duplicate = True
                    break

            if not duplicate:
                detected_eyes.append(
                    (ex, ey, ew, eh)
                )

        # --------------------------------------------------
        # KEEP AT MOST TWO EYES
        # --------------------------------------------------

        detected_eyes = sorted(
            detected_eyes,
            key=lambda eye: eye[0]
        )

        detected_eyes = detected_eyes[:2]

        # --------------------------------------------------
        # DRAW EYE BOXES
        # --------------------------------------------------

        for (
            ex,
            ey,
            ew,
            eh
        ) in detected_eyes:

            cv2.rectangle(
                face_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
                2
            )

        total_eyes += len(
            detected_eyes
        )

    # --------------------------------------------------
    # ENCODE RESULT IMAGE
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
        "eyes_detected": total_eyes,
        "image": result_base64
    })
