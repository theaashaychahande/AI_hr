import cv2

def detect_multiple_faces(frame):
    """
    Detects if more than one face is present in the frame.
    Returns:
        - True if multiple faces detected
        - False otherwise
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

   
    return len(faces) > 1