import cv2
import numpy as np

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Define the lower and upper bounds for white color in HSV
lower_white = np.array([0, 0, 200])  # Low saturation, high value (brightness)
upper_white = np.array([255, 60, 255])  # High hue, low saturation, high brightness

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for white color
    white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Bitwise AND to extract the white regions from the original frame
    white_regions = cv2.bitwise_and(frame, frame, mask=white_mask)

    # Display the original frame, the white mask, and the detected white regions
    cv2.imshow('Original Frame', frame)
    cv2.imshow('White Mask', white_mask)
    cv2.imshow('White Regions', white_regions)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
