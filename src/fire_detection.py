import cv2
import numpy as np

def detect_fire(image_path, save_path=None):

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_fire = np.array([0,150,150])
    upper_fire = np.array([35,255,255])

    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 500:
            fire_count += 1
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

    print("🔥 Fire Regions Detected:", fire_count)

    if save_path:
        cv2.imwrite(save_path, image)

    return image
