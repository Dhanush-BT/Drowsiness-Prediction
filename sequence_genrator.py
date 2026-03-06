import pandas as pd
import numpy as np
from tqdm import tqdm

SEQ_LEN = 30

print("Loading dataset...")
df = pd.read_csv("frame_level_dataset.csv")

features = ["EAR","MAR","pitch","yaw","roll","PERCLOS"]

X = []
y = []

groups = df.groupby(["subject","video"])

for (_, _), group in tqdm(groups):

    group = group.sort_values("frame")

    data = group[features].values
    label = group["label"].iloc[0]

    # PAD SHORT VIDEOS
    if len(data) < SEQ_LEN:
        pad = np.repeat(data[-1][None,:], SEQ_LEN-len(data), axis=0)
        data = np.vstack((data,pad))

    for i in range(len(data) - SEQ_LEN + 1):
        seq = data[i:i+SEQ_LEN]

        X.append(seq)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X shape:",X.shape)
print("y shape:",y.shape)

np.save("X_sequences.npy",X)
np.save("y_sequences.npy",y)

print("Saved sequences successfully.")

# Dataset sanity check
print("\nFeature ranges:")

print("EAR:",X[:,:,0].min(),X[:,:,0].max())
print("MAR:",X[:,:,1].min(),X[:,:,1].max())
print("pitch:",X[:,:,2].min(),X[:,:,2].max())
print("yaw:",X[:,:,3].min(),X[:,:,3].max())
print("roll:",X[:,:,4].min(),X[:,:,4].max())
print("PERCLOS:",X[:,:,5].min(),X[:,:,5].max())