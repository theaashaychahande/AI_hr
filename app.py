from flask import Flask, request, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
REFERENCE_FACE_DIR = "reference_faces"
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(REFERENCE_FACE_DIR, exist_ok=True)

def load_image_from_bytes(data):
    """Convert bytes to OpenCV image"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img

def detect_multiple_faces(frame):
    """Detect if more than one face is present in the frame"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 1

@app.route("/")
def home():
    return jsonify({
        "message": "AI Interview Assistant Backend Running!",
        "available_routes": {
            "register_face": "/register-face (POST)",
            "verify_interviewer": "/verify-interviewer (POST)",
            "fraud_check": "/fraud-check (POST)",
            "fraud_alert": "/alert (POST)"
        },
        "status": "OK"
    })

@app.route("/register-face", methods=["POST"])
def register_face():
    """
    Register a candidate's face from a webcam frame.
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

    ref_face_path = os.path.join(REFERENCE_FACE_DIR, f"{candidate_id}.jpg")
    cv2.imwrite(ref_face_path, frame)

    return jsonify({"status": "registered", "message": f"Face for {candidate_id} saved"})

@app.route("/verify-interviewer", methods=["POST"])
def verify_interviewer():
    """
    Verify a candidate's face from a webcam frame.
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

        # Detect multiple faces
        if detect_multiple_faces(frame):
            return jsonify({"status": "fraud", "message": "Multiple faces detected"})

        return jsonify({"status": "verified", "message": "Face ID logged in"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/fraud-check", methods=["POST"])
def fraud_check():
    """
    API to check for fraud 
    Expected:
    - POST with:
        - "frame": image file
    """
    if "frame" not in request.files:
        return jsonify({"status": "error", "message": "No frame provided"})

    file = request.files["frame"]
    frame = load_image_from_bytes(file.read())

    if detect_multiple_faces(frame):
        return jsonify({
            "status": "fraud",
            "message": "Multiple faces detected"
        })

    return jsonify({
        "status": "safe",
        "message": "No fraud detected"
    })

@app.route("/alert", methods=["POST"])
def fraud_alert():
    """
    Endpoint to receive fraud alerts from frontend (e.g., tab switch)
    """
    data = request.json
    reason = data.get("reason", "Unknown")
    print(f" Fraud Alert: {reason}")
    return jsonify({"status": "alert_received"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)