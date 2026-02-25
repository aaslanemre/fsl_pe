import json
import cv2
import os
import matplotlib.pyplot as plt

# Update these paths to match your Uni PC setup
JSON_PATH = 'data/mpii/annotations/mpii_train.json'
IMG_DIR = 'data/mpii/images/'

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Grab the first entry that actually has joints
sample = data[0]
img_name = sample['img_paths']
joints = sample['joint_self'] # Usually a list of [x, y, visibility]

# Load image
img_path = os.path.join(IMG_DIR, img_name)
image = cv2.imread(img_path)

if image is None:
    print(f"❌ Error: Could not find image at {img_path}")
else:
    print(f"✅ Success: Found {img_name}")
    # Draw joints
    for joint in joints:
        x, y, vis = int(joint[0]), int(joint[1]), joint[2]
        color = (0, 255, 0) if vis > 0 else (0, 0, 255) # Green if visible, Red if hidden
        cv2.circle(image, (x, y), 5, color, -1)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Testing: {img_name} (Red = Occluded)")
    plt.show()
