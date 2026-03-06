import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from collections import deque

# =============================
# LSTM MODEL
# =============================

class LSTMModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=6, hidden_size=64, batch_first=True)
        self.dropout1 = nn.Dropout(0.4)

        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.dropout2 = nn.Dropout(0.4)

        self.fc1 = nn.Linear(32,16)
        self.fc2 = nn.Linear(16,1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x,_ = self.lstm1(x)
        x = self.dropout1(x)

        x,_ = self.lstm2(x)
        x = self.dropout2(x)

        x = x[:,-1,:]

        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x.squeeze()

# =============================
# LOAD MODEL
# =============================

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = LSTMModel()
model.load_state_dict(torch.load("best_drowsiness_model.pth", map_location=device))
model.to(device)
model.eval()

print("Model Loaded")

# =============================
# MEDIAPIPE SETUP
# =============================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

EAR_THRESHOLD = 0.21

# =============================
# UTILITY FUNCTIONS
# =============================

def euclidean(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))

def compute_ear(landmarks,eye):

    p1,p2,p3,p4,p5,p6 = [landmarks[i] for i in eye]

    v1 = euclidean(p2,p6)
    v2 = euclidean(p3,p5)
    h = euclidean(p1,p4)

    return (v1+v2)/(2*h+1e-6)

def compute_mar(landmarks):

    upper = landmarks[13]
    lower = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    vertical = euclidean(upper,lower)
    horizontal = euclidean(left,right)

    return vertical/(horizontal+1e-6)

def compute_head_pose(landmarks,w,h):

    try:

        image_points = np.array([
            landmarks[1],
            landmarks[152],
            landmarks[33],
            landmarks[263],
            landmarks[61],
            landmarks[291]
        ],dtype="double")

        model_points = np.array([
            (0,0,0),
            (0,-63.6,-12.5),
            (-43.3,32.7,-26),
            (43.3,32.7,-26),
            (-28.9,-28.9,-24.1),
            (28.9,-28.9,-24.1)
        ])

        focal_length = w
        center = (w/2,h/2)

        camera_matrix = np.array([
            [focal_length,0,center[0]],
            [0,focal_length,center[1]],
            [0,0,1]
        ],dtype="double")

        dist_coeffs = np.zeros((4,1))

        success,rotation_vector,_ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if not success:
            return 0,0,0

        rmat,_ = cv2.Rodrigues(rotation_vector)
        angles,_,_,_,_,_ = cv2.RQDecomp3x3(rmat)

        return angles[0],angles[1],angles[2]

    except:
        return 0,0,0

# =============================
# BUFFERS
# =============================

sequence_buffer = deque(maxlen=30)
ear_window = deque(maxlen=30)

# =============================
# VIDEO INPUT
# =============================

cap = cv2.VideoCapture("ndrow.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

delay = int(1000/fps)

alert_frames = 0
drowsy_frames = 0

while True:

    ret,frame = cap.read()

    if not ret:
        break

    h,w = frame.shape[:2]

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        mesh = results.multi_face_landmarks[0]

        landmarks = [(lm.x*w,lm.y*h) for lm in mesh.landmark]

        left_ear = compute_ear(landmarks,LEFT_EYE)
        right_ear = compute_ear(landmarks,RIGHT_EYE)

        ear = (left_ear+right_ear)/2

        mar = compute_mar(landmarks)

        pitch,yaw,roll = compute_head_pose(landmarks,w,h)

        ear_window.append(ear)

        # Sliding window PERCLOS
        closed = sum(e < EAR_THRESHOLD for e in ear_window)
        perclos = closed/len(ear_window)

        features = [ear,mar,pitch,yaw,roll,perclos]

        sequence_buffer.append(features)

        if len(sequence_buffer) == 30:

            seq = np.array(sequence_buffer)

            seq = torch.tensor(seq,dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():

                prob = model(seq).item()

            label = "DROWSY" if prob > 0.5 else "ALERT"

            if label == "DROWSY":
                drowsy_frames += 1
                color = (0,0,255)
            else:
                alert_frames += 1
                color = (0,255,0)

            cv2.putText(frame,
                        f"{label} {prob:.2f}",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

    cv2.imshow("Driver Drowsiness Detection",frame)

    if cv2.waitKey(delay) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL RESULT")
print("Alert Frames:",alert_frames)
print("Drowsy Frames:",drowsy_frames)

if drowsy_frames > alert_frames:
    print("Driver State: DROWSY")
else:
    print("Driver State: ALERT")