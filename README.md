# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Program:
```

import cv2
import matplotlib.pyplot as plt
import os
import glob
import urllib.request

# Function to download cascade classifiers if needed
def download_cascade(filename):
    """Download cascade classifier from OpenCV repository"""
    url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}'
    try:
        print(f"Downloading {filename} from OpenCV repository...")
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

# Define specific images for different purposes
image_paths = {
    'humans': ['human1.png', 'human2.png'],
    'with_glasses': ['glass.png'],
    'group': ['group.png', 'group.jpg']
}

# Check which images exist
available_images = {}
for category, files in image_paths.items():
    available_images[category] = [f for f in files if os.path.exists(f)]

print("Available images:")
for category, files in available_images.items():
    print(f"  {category}: {files}")

# Load images
def load_image(filepath):
    """Load image and check if successful"""
    img = cv2.imread(filepath, 0)
    if img is None:
        print(f"Warning: Could not load {filepath}")
        return None
    return img

# Load specific images
human_images = [load_image(f) for f in available_images['humans']]
human_images = [img for img in human_images if img is not None]

glass_images = [load_image(f) for f in available_images['with_glasses']]
glass_images = [img for img in glass_images if img is not None]

group_images = [load_image(f) for f in available_images['group']]
group_images = [img for img in group_images if img is not None]

print(f"Loaded {len(human_images)} human image(s)")
print(f"Loaded {len(glass_images)} image(s) with glasses")
print(f"Loaded {len(group_images)} group image(s)")

# Display loaded images
if human_images:
    for i, img in enumerate(human_images):
        plt.figure(figsize=(10, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"Human Image {i+1}")
        plt.show()

if glass_images:
    for i, img in enumerate(glass_images):
        plt.figure(figsize=(10, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"Human with Glasses Image {i+1}")
        plt.show()

if group_images:
    for i, img in enumerate(group_images):
        plt.figure(figsize=(10, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"Group Image {i+1}")
        plt.show()

# Load cascade classifiers - download if needed
face_cascade_file = 'haarcascade_frontalface_default.xml'
eye_cascade_file = 'haarcascade_eye.xml'

# Try to download if files don't exist
if not os.path.exists(face_cascade_file):
    download_cascade(face_cascade_file)
if not os.path.exists(eye_cascade_file):
    download_cascade(eye_cascade_file)

# Load the cascade classifiers
face_cascade = cv2.CascadeClassifier(face_cascade_file)
eye_cascade = cv2.CascadeClassifier(eye_cascade_file)

if face_cascade.empty():
    print("Warning: Face cascade classifier is empty")
if eye_cascade.empty():
    print("Warning: Eye cascade classifier is empty")

def detect_face(img, scaleFactor=1.1, minNeighbors=5):
    """Detect faces in image"""
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    print(f"  Detected {len(face_rects)} face(s)")
    return face_img

def detect_eyes(img):
    """Detect eyes in image"""
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    print(f"  Detected {len(eyes)} eye(s)")
    return face_img

# Process human images
print("\n=== PROCESSING HUMAN IMAGES ===")
for i, img in enumerate(human_images):
    print(f"Human Image {i+1}:")
    result_faces = detect_face(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(result_faces, cmap='gray')
    plt.title(f"Faces Detected in Human Image {i+1}")
    plt.show()
    
    result_eyes = detect_eyes(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(result_eyes, cmap='gray')
    plt.title(f"Eyes Detected in Human Image {i+1}")
    plt.show()

# Process images with glasses
print("\n=== PROCESSING HUMAN WITH GLASSES IMAGES ===")
for i, img in enumerate(glass_images):
    print(f"Human with Glasses Image {i+1}:")
    result_faces = detect_face(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(result_faces, cmap='gray')
    plt.title(f"Faces Detected in With Glasses Image {i+1}")
    plt.show()
    
    result_eyes = detect_eyes(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(result_eyes, cmap='gray')
    plt.title(f"Eyes Detected in With Glasses Image {i+1}")
    plt.show()

# Process group images
print("\n=== PROCESSING GROUP IMAGES ===")
for i, img in enumerate(group_images):
    print(f"Group Image {i+1}:")
    result_faces = detect_face(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(result_faces, cmap='gray')
    plt.title(f"Faces Detected in Group Image {i+1}")
    plt.show()
    
    result_eyes = detect_eyes(img)
    plt.figure(figsize=(10, 6))
    plt.imshow(result_eyes, cmap='gray')
    plt.title(f"Eyes Detected in Group Image {i+1}")
    plt.show()

print("\n=== DETECTION COMPLETE ===")
plt.show()
```
## output:
<img width="1232" height="822" alt="Screenshot 2026-08-29 191935" src="https://github.com/user-attachments/assets/a33487f9-30d0-4d17-abda-b2628197ff35" />

<img width="1245" height="837" alt="Screenshot 2026-08-29 192001" src="https://github.com/user-attachments/assets/ea758d57-96fa-45bd-b5c2-228589858b85" />

<img width="1255" height="845" alt="Screenshot 2026-08-29 192027" src="https://github.com/user-attachments/assets/f1168334-d83e-4708-8bf5-018ccbf3bde8" />

<img width="1265" height="851" alt="Screenshot 2026-08-29 192042" src="https://github.com/user-attachments/assets/0b1f8109-c668-4d7f-9c64-b3f7e435ff27" />

<img width="1267" height="853" alt="Screenshot 2026-08-29 192052" src="https://github.com/user-attachments/assets/06d54666-0a56-42f3-93de-298a9cd9a0ba" />
