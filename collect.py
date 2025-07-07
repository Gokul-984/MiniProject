import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Define allowed labels for data collection (adjust as needed)
allowed_labels = { "M", "N", "X", "U"}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Define dataset storage
dataset_path = "landmark_dataset.csv"
data = []

# Start capturing video
cap = cv2.VideoCapture(0)
current_label = ""

# Function to compute centered skeleton and draw on white canvas
def get_centered_skeleton(hand_landmarks, frame_width, frame_height, canvas_size=400, scale_margin=0.7):
    pts = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        pts.append((x, y))
    
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width == 0 or bbox_height == 0:
        return None, None

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    scale_factor = (canvas_size * scale_margin) / max(bbox_width, bbox_height)
    canvas_cx = canvas_size / 2
    canvas_cy = canvas_size / 2

    transformed = []
    for (x, y) in pts:
        x_new = int((x - cx) * scale_factor + canvas_cx)
        y_new = int((y - cy) * scale_factor + canvas_cy)
        transformed.append((x_new, y_new))
    
    # Create white canvas and draw skeleton
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(transformed) and end_idx < len(transformed):
            cv2.line(canvas, transformed[start_idx], transformed[end_idx], (0, 255, 0), thickness=2)
    for pt in transformed:
        cv2.circle(canvas, pt, radius=3, color=(0, 255, 0), thickness=-1)
    
    return canvas, transformed

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Process frame with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Display instructions on the frame
    instruction = f"Press allowed key {sorted(allowed_labels)} to set label | Current: {current_label} | Press 'c' to capture | Press 'q' to exit and save"
    cv2.putText(frame, instruction, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Prepare skeleton canvas
    skeleton_canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        canvas, _ = get_centered_skeleton(hand_landmarks, frame_w, frame_h)
        if canvas is not None:
            skeleton_canvas = canvas

    # Display the skeleton canvas for visual feedback
    cv2.imshow("Skeleton Canvas", skeleton_canvas)
    cv2.imshow("Hand Landmark Capture", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        # When 'c' is pressed, capture landmarks if current_label is allowed and a hand is detected.
        if current_label and current_label in allowed_labels and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                data.append([current_label] + landmarks)
                print(f"Captured landmarks for label: {current_label}")
    elif key != -1:
        pressed = chr(key).upper()
        if pressed in allowed_labels:
            current_label = pressed
        else:
            current_label = ""

# Save dataset if data was captured
if data:
    df = pd.DataFrame(data, columns=["label"] + [f"lm_{i}" for i in range(63)])
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")
else:
    print("No data captured.")

cap.release()
cv2.destroyAllWindows()
