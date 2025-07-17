import cv2
from deepface import DeepFace

reference_img_path = "reference_faces/Aashay.jpg"
reference_img = cv2.imread(reference_img_path)
cap = cv2.VideoCapture(0)
print("Webcam running... Press 'c' to capture frame, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to open webcam")
        break

    cv2.imshow("Webcam - Press 'c' to capture", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        print("Capturing frame for verification...")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()


temp_path = "temp_verify.jpg"
cv2.imwrite(temp_path, frame)

try:
    result = DeepFace.verify(reference_img_path, temp_path, model_name="Facenet")
    
    if result["verified"]:
        print("✅ Face Verified")
    else:
        print("❌ Not Verified")
        
except Exception as e:
    print("Error during verification:", e)