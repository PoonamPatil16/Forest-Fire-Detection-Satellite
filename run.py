from src.fire_detection import detect_fire
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide tkinter root window
Tk().withdraw()

# Open file picker
image_path = askopenfilename(title="Select Satellite Fire Image")

if image_path == "":
    print("No file selected.")
    exit()

# Run fire detection
result = detect_fire(image_path, "results/output.jpg")

# Show result
cv2.imshow("Fire Detection Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
