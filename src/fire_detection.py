import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide tkinter root window
Tk().withdraw()

# Open file picker
file_path = askopenfilename(title="Select Satellite Fire Image")

# Load image
image = cv2.imread(file_path)

if image is None:
    print("Error: Could not load the image.")
    exit()

# Convert to RGB for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Fire color range
lower_fire = np.array([0,150,150])
upper_fire = np.array([35,255,255])

# Create fire mask
mask = cv2.inRange(hsv, lower_fire, upper_fire)

# Clean noise
kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

# Find contours (fire regions)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

fire_count = 0
output = image_rgb.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)

    # Ignore very small detections
    if area > 500:
        fire_count += 1

        x,y,w,h = cv2.boundingRect(cnt)

        # Draw bounding box
        cv2.rectangle(output,(x,y),(x+w,y+h),(255,0,0),2)

        # Estimate intensity
        if area > 5000:
            intensity = "HIGH"
        elif area > 2000:
            intensity = "MEDIUM"
        else:
            intensity = "LOW"

        cv2.putText(output,
                    intensity,
                    (x,y-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,0,0),
                    2)

# Display results
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Fire Mask")
plt.imshow(mask,cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title(f"Detected Fire Regions: {fire_count}")
plt.imshow(output)
plt.axis("off")

plt.show()

print("🔥 Total Fire Regions Detected:", fire_count)