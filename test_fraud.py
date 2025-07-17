import requests
url = "http://localhost:5000/fraud-check"
image_path = "test_images/multiple_faces.jpg"

with open(image_path, "rb") as f:
    files = {"frame": f}
    response = requests.post(url, files=files)

print("AI Response:", response.json())