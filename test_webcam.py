import cv2
import requests

API_REGISTER = "http://localhost:5000/register-face"
API_VERIFY = "http://localhost:5000/verify-interviewer"

def send_frame(api_url, frame, candidate_id):
    _, img_encoded = cv2.imencode('.jpg', frame)
    files = {'frame': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
    data = {'candidate_id': candidate_id}

    response = requests.post(api_url, data=data, files=files)
    print("Response:", response.json())

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("[INFO] Webcam is running. Press 'c' to capture a frame.")

    candidate_id = input("Enter Candidate ID (e.g. john_doe): ").strip()
    if not candidate_id:
        print("Error: Candidate ID is required.")
        return

    mode = input("Choose mode: (r) Register Face / (v) Verify Face: ").strip().lower()

    if mode not in ['r', 'v']:
        print("Invalid mode.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam.")
            break

        cv2.imshow('Webcam Feed', frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            print("[INFO] Capturing frame...")
            if mode == 'r':
                print(f"[INFO] Registering face for {candidate_id}")
                send_frame(API_REGISTER, frame, candidate_id)
            elif mode == 'v':
                print(f"[INFO] Verifying face for {candidate_id}")
                send_frame(API_VERIFY, frame, candidate_id)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()