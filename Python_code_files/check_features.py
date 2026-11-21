import pickle
import numpy as np

path = r"D:\LAV-DF\LAV-DF\decoder_voxceleb_feat\train_000472.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

for key in ["v_feat", "a_feat", "av_hidden_feat"]:
    arr = np.asarray(data[key], dtype=np.float32)

    print("\nFeature:", key)
    print("Shape:", arr.shape)
    print("All finite:", np.isfinite(arr).all())
    print("NaN count:", np.isnan(arr).sum())
    print("Inf count:", np.isinf(arr).sum())
    print("Min:", np.nanmin(arr))
    print("Max:", np.nanmax(arr))