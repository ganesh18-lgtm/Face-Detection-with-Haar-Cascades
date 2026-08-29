
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