import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
import matplotlib.pyplot as plt

# ==============================
# MediaPipe Setup
# ==============================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21

# ==============================
# Utility Functions
# ==============================

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(landmarks, eye_points):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]

    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    return (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)


def compute_mar(landmarks):
    upper = landmarks[13]
    lower = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    vertical = euclidean(upper, lower)
    horizontal = euclidean(left, right)

    return vertical / (horizontal + 1e-6)


def compute_head_pose(landmarks, img_w, img_h):
    try:
        image_points = np.array([
            landmarks[1],
            landmarks[152],
            landmarks[33],
            landmarks[263],
            landmarks[61],
            landmarks[291]
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26),
            (43.3, 32.7, -26),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        focal_length = img_w
        center = (img_w / 2, img_h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        return angles[0], angles[1], angles[2]

    except:
        return 0.0, 0.0, 0.0


# ==============================
# Dataset Builder
# ==============================

DATASET_PATH = "UTA-RLDD Face Cropped Video/len5"

rows = []

for subject in tqdm(os.listdir(DATASET_PATH)):

    subject_path = os.path.join(DATASET_PATH, subject)

    # Skip hidden/system files (macOS fix)
    if not os.path.isdir(subject_path):
        continue

    for label_folder in os.listdir(subject_path):

        label_path = os.path.join(subject_path, label_folder)

        if not os.path.isdir(label_path):
            continue

        if label_folder not in ["0", "10"]:
            continue

        label = 0 if label_folder == "0" else 1

        for video in os.listdir(label_path):

            video_path = os.path.join(label_path, video)

            if not os.path.isfile(video_path):
                continue

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                continue

            ear_list = []
            video_rows = []
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = face_mesh.process(rgb)

                if result.multi_face_landmarks:

                    mesh = result.multi_face_landmarks[0]

                    landmarks = [(lm.x * w, lm.y * h) for lm in mesh.landmark]

                    left_ear = compute_ear(landmarks, LEFT_EYE)
                    right_ear = compute_ear(landmarks, RIGHT_EYE)
                    ear = (left_ear + right_ear) / 2.0

                    mar = compute_mar(landmarks)
                    pitch, yaw, roll = compute_head_pose(landmarks, w, h)

                    # Skip invalid values
                    if np.isnan(ear) or np.isnan(mar):
                        frame_id += 1
                        continue

                    ear_list.append(ear)

                    video_rows.append([
                        subject,
                        video,
                        frame_id,
                        ear,
                        mar,
                        pitch,
                        yaw,
                        roll,
                        label
                    ])

                frame_id += 1

            cap.release()

            # Compute PERCLOS safely
            if len(ear_list) > 0:

                closed = sum(e < EAR_THRESHOLD for e in ear_list)
                perclos = closed / len(ear_list)

                for r in video_rows:
                    r.append(perclos)

                rows.extend(video_rows)


# ==============================
# Create DataFrame
# ==============================

columns = [
    "subject",
    "video",
    "frame",
    "EAR",
    "MAR",
    "pitch",
    "yaw",
    "roll",
    "label",
    "PERCLOS"
]

df = pd.DataFrame(rows, columns=columns)

# Clean infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df.to_csv("frame_level_dataset.csv", index=False)

# ==============================
# Validation & Analysis
# ==============================

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nLabel distribution:\n", df["label"].value_counts())

print("\nMean EAR by class:\n", df.groupby("label")["EAR"].mean())
print("\nMean PERCLOS by class:\n", df.groupby("label")["PERCLOS"].mean())

print("\nCorrelation Matrix:\n")
print(df.corr(numeric_only=True))