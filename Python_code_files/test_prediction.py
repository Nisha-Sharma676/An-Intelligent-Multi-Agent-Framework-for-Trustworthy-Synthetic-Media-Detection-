import pickle
import glob
import torch
import numpy as np

from model.class_model import ClassificationModel


# Load classification model
model = ClassificationModel.load_from_checkpoint(
    "checkpoints/class_last.ckpt",
    map_location="cpu"
)

model.eval()

# Take first feature file
files = glob.glob(
    r"C:\Users\AMAYA\Downloads\FakeAVCeleb_v1.2\FakeAVCeleb\decoder_voxceleb_feat\*.pkl"
)

f = files[0]

# Load features
with open(f, "rb") as file:
    d = pickle.load(file)

av = torch.tensor(
    np.asarray(d["av_hidden_feat"]),
    dtype=torch.float32
).unsqueeze(0)

v = torch.tensor(
    np.asarray(d["v_feat"]),
    dtype=torch.float32
).unsqueeze(0)

a = torch.tensor(
    np.asarray(d["a_feat"]),
    dtype=torch.float32
).unsqueeze(0)

# Number of actual frames
n = av.shape[1]

# Model expects 500 frames
max_len = 500

# Padding
av = torch.nn.functional.pad(av, (0, 0, 0, max_len - n))
v = torch.nn.functional.pad(v, (0, 0, 0, max_len - n))
a = torch.nn.functional.pad(a, (0, 0, 0, max_len - n))

# Padding mask
mask = torch.ones((1, max_len), dtype=torch.bool)
mask[:, :n] = False

# Number of frames
frames = torch.tensor([n])

# Prediction
with torch.no_grad():
    logit = model(av, v, a, frames, mask)
    probability = torch.sigmoid(logit).item()

print("FILE:", f)
print("FRAMES:", n)
print("LOGIT:", logit.item())
print("PROBABILITY:", probability)