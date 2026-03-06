import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("frame_level_dataset.csv")

print("Dataset shape:", df.shape)

# Features
features = ["EAR","MAR","pitch","yaw","roll","PERCLOS"]

sns.set_style("whitegrid")

# ==============================
# 1 Feature Distribution
# ==============================

for feature in features:
    plt.figure()
    sns.histplot(df[feature], bins=100, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()


# ==============================
# 2 Feature vs Class
# ==============================

for feature in features:
    plt.figure()
    sns.boxplot(x="label", y=feature, data=df)
    plt.title(f"{feature} vs Drowsiness")
    plt.show()


# ==============================
# 3 Correlation Matrix
# ==============================

plt.figure(figsize=(10,8))
corr = df[features + ["label"]].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()


# ==============================
# 4 EAR Time Behaviour
# ==============================

sample_video = df["video"].unique()[0]

temp = df[df["video"] == sample_video]

plt.figure(figsize=(12,5))
plt.plot(temp["frame"], temp["EAR"])
plt.title("EAR over Time (Sample Video)")
plt.xlabel("Frame")
plt.ylabel("EAR")
plt.show()


# ==============================
# 5 MAR vs EAR Relationship
# ==============================

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["EAR"],
    y=df["MAR"],
    hue=df["label"],
    alpha=0.3
)

plt.title("EAR vs MAR")
plt.show()


# ==============================
# 6 PERCLOS vs Class
# ==============================

plt.figure()
sns.violinplot(x="label", y="PERCLOS", data=df)
plt.title("PERCLOS Distribution")
plt.show()


# ==============================
# 7 Class Distribution
# ==============================

plt.figure()

df["label"].value_counts().plot(
    kind="bar"
)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.xticks([0,1],["Non-Drowsy","Drowsy"])

plt.show()


# ==============================
# 8 Pairplot (sampled)
# ==============================

sample_df = df.sample(5000)

sns.pairplot(
    sample_df,
    vars=features,
    hue="label"
)

plt.show()