from roboflow import Roboflow
import cv2
rf = Roboflow(api_key="97LtEo52U94uyhBSrQgw")
project = rf.workspace().project("https://detect.roboflow.com/rc-car-detection-graduation/1")
model = project.version(1).model

# Open webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Process webcam frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Save the current frame as a temporary image
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Make a prediction
    predictions = model.predict(temp_image_path, confidence=40, overlap=30).json()

    # Visualize predictions on the frame
    for prediction in predictions['predictions']:
        x = int(prediction['x'] - prediction['width'] / 2)
        y = int(prediction['y'] - prediction['height'] / 2)
        width = int(prediction['width'])
        height = int(prediction['height'])
        label = prediction['class']
        confidence = prediction['confidence']

        # Draw a rectangle around the detected object
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Webcam Predictions", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()