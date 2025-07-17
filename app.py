from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import os

app = Flask(__name__)

REFERENCE_FACE_DIR = "reference_faces"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def load_image_from_bytes(data):
    """Convert bytes to OpenCV image"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img

@app.route("/verify-interviewer", methods=["POST"])
def verify_interviewer():
    """
    API to verify the face when user clicks "Start Interview"
    Expected:
    - POST with:
        - "frame": image file
        - "candidate_id": string (e.g. "john_doe")
    """

    if "frame" not in request.files:
        return jsonify({"status": "error", "message": "No frame provided"})

    candidate_id = request.form.get("candidate_id")
    if not candidate_id:
        return jsonify({"status": "error", "message": "Missing candidate_id"})

    file = request.files["frame"]
    frame = load_image_from_bytes(file.read())

    temp_path = os.path.join(TEMP_DIR, "verify_frame.jpg")
    cv2.imwrite(temp_path, frame)

    ref_face_path = os.path.join(REFERENCE_FACE_DIR, f"{candidate_id}.jpg")

    if not os.path.exists(ref_face_path):
        return jsonify({"status": "error", "message": "Reference face not found"})

    try:
        result = DeepFace.verify(ref_face_path, temp_path, model_name="Facenet")

        if not result["verified"]:
            return jsonify({"status": "unauthorized", "message": "Face not verified"})

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 1:
            return jsonify({"status": "fraud", "message": "Multiple faces detected"})

        return jsonify({"status": "verified", "message": "Face verified successfully"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)