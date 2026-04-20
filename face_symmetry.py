import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "face_landmarker.task"
image_path = "face.jpg"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.IMAGE,
    num_faces=1
)

mp_image = mp.Image.create_from_file(image_path)

with FaceLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

landmarks = result.face_landmarks[0]
pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32)

# Example: define a vertical midline from central landmarks you choose
# You should replace these indices with a validated set for your use case
mid_idx = [1, 4, 6, 9, 168]
mid_pts = pts[mid_idx, :2]
x_mid = mid_pts[:, 0].mean()

# Example left/right landmark pairs — replace with your own validated pairs
pairs = [
    (33, 263),   # outer eye corners-ish in MediaPipe mesh conventions
    (133, 362),  # inner eye corners-ish
    (61, 291),   # mouth corners
    (234, 454),  # cheek/jaw lateral points
]

errors = []
for li, ri in pairs:
    lx, ly = pts[li, :2]
    rx, ry = pts[ri, :2]

    mirrored_lx = 2 * x_mid - lx
    err = np.sqrt((mirrored_lx - rx) ** 2 + (ly - ry) ** 2)
    errors.append(err)

# normalize by eye distance
left_eye = pts[133, :2]
right_eye = pts[362, :2]
D = np.linalg.norm(left_eye - right_eye)

symmetry_score = 1.0 - (np.mean(errors) / D)
symmetry_score = float(np.clip(symmetry_score, 0.0, 1.0))
print("symmetry_score:", symmetry_score)
