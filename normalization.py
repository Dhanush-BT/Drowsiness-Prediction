import numpy as np

X = np.load("X_sequences.npy")
y = np.load("y_sequences.npy")

print("Before normalization:", X.shape)

# normalize columns
X[:,:,0] = X[:,:,0] / 1.0      # EAR
X[:,:,1] = X[:,:,1] / 1.5      # MAR
X[:,:,2] = X[:,:,2] / 180.0    # pitch
X[:,:,3] = X[:,:,3] / 90.0     # yaw
X[:,:,4] = X[:,:,4] / 180.0    # roll
# PERCLOS already 0–1

np.save("X_norm.npy",X)
np.save("y_norm.npy",y)

print("Normalization done")