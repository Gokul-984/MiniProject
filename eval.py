import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Configuration & Constants
# ---------------------------
MODEL_INPUT_SIZE = (400, 400)  # Expected size for the CNN models
SKELETON_SIZE = 400            # Size of the white canvas used for skeleton processing
scale_margin = 0.5             # Margin factor for scaling the hand in the canvas
TEST_DATASET_DIR = "test_dataset"  # Folder structure: test_dataset/<class_label>/image.jpg

# ---------------------------
# Helper Functions
# ---------------------------

def preprocess_image(img, target_size=MODEL_INPUT_SIZE):
    """
    Resize and normalize the image.
    """
    img_resized = cv2.resize(img, target_size)
    img_norm = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_norm, axis=0)

def get_centered_skeleton(hand_landmarks, frame_width, frame_height, canvas_size=SKELETON_SIZE):
    """
    Compute the transformed landmarks and return them as a list of (x, y) points.
    """
    pts = []
    for lm in hand_landmarks.landmark:
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        pts.append((x, y))
    
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if not xs or not ys:
        return None
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    if bbox_width == 0 or bbox_height == 0:
        return None
    
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
    
    return transformed

def draw_skeleton_on_canvas(canvas, points):
    """
    Draw the hand skeleton on the given canvas.
    """
    mp_hands = cv2.__version__  # dummy; not used. We use global MediaPipe connections.
    # For drawing, we need to use MediaPipe's HAND_CONNECTIONS:
    from mediapipe.framework.formats import landmark_pb2
    # However, for our purposes, we can hardcode the connections from MediaPipe Hands:
    HAND_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4),
                        (0,5), (5,6), (6,7), (7,8),
                        (5,9), (9,10), (10,11), (11,12),
                        (9,13), (13,14), (14,15), (15,16),
                        (13,17), (17,18), (18,19), (19,20)]
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            cv2.line(canvas, pt1, pt2, (0, 255, 0), thickness=2)
    for pt in points:
        cv2.circle(canvas, pt, radius=3, color=(0, 255, 0), thickness=-1)

# ---------------------------
# Load Models
# ---------------------------
# Load coarse model and mapping
coarse_model = tf.keras.models.load_model("coarse_model.h5")
with open("coarse_class_map.pkl", "rb") as f:
    coarse_class_map = pickle.load(f)
inv_coarse_map = {v: k for k, v in coarse_class_map.items()}

# Load fine models and their label maps
group_list = ["group_0", "group_1", "group_2", "group_3", "group_4", "group_5"]
group_models = {}
group_label_maps = {}
for group in group_list:
    model_path = f"{group}_model.h5"
    map_path = f"{group}_map.pkl"
    if os.path.exists(model_path) and os.path.exists(map_path):
        group_models[group] = tf.keras.models.load_model(model_path)
        with open(map_path, "rb") as f:
            group_label_maps[group] = pickle.load(f)
    else:
        group_models[group] = None
        group_label_maps[group] = {}
        print(f"Warning: Model or map for {group} not found.")

# Load landmark-based model (numeric)
try:
    with open("landmark_mlp.pkl", "rb") as f:
        landmark_model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    print("Landmark model files not found or error loading:", e)
    landmark_model = None

# ---------------------------
# Load Test Images
# ---------------------------
test_images = []
test_labels = []
for label in os.listdir(TEST_DATASET_DIR):
    label_folder = os.path.join(TEST_DATASET_DIR, label)
    if os.path.isdir(label_folder):
        for fname in os.listdir(label_folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(label_folder, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    # Optionally, you might want to resize using cv2.resize if needed.
                    img = cv2.resize(img, MODEL_INPUT_SIZE)
                    test_images.append(img)
                    test_labels.append(label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
print(f"Loaded {len(test_images)} test images.")

# ---------------------------
# Prediction Loop on Test Images
# ---------------------------
final_predictions = []
for image in test_images:
    # Preprocess image for CNN pipeline
    cnn_input = preprocess_image(image, target_size=MODEL_INPUT_SIZE)
    # Run coarse model
    coarse_pred = coarse_model.predict(cnn_input)
    coarse_idx = np.argmax(coarse_pred)
    coarse_group = inv_coarse_map.get(coarse_idx, None)
    
    if coarse_group and group_models.get(coarse_group) is not None:
        sub_model = group_models[coarse_group]
        sub_map = group_label_maps[coarse_group]
        fine_pred = sub_model.predict(cnn_input)
        fine_idx = np.argmax(fine_pred)
        inv_sub_map = {v: k for k, v in sub_map.items()}
        final_predictions.append(inv_sub_map.get(fine_idx, "Unknown"))
    else:
        final_predictions.append("Unknown")

# ---------------------------
# Evaluation Metrics
# ---------------------------
print("Classification Report:")
print(classification_report(test_labels, final_predictions))
print("Confusion Matrix:")
print(confusion_matrix(test_labels, final_predictions))
