import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "checkpoints" / "temporal_evidence_results.json"
OUT_DIR = BASE_DIR / "project_graphs"
OUT_DIR.mkdir(exist_ok=True)

# Load test-video results
with open(JSON_PATH, "r", encoding="utf-8") as f:
    records = json.load(f)

if isinstance(records, dict):
    for key in ["results", "predictions", "videos", "data"]:
        if isinstance(records.get(key), list):
            records = records[key]
            break

# Normalize keys like A-V_weight, av_weight and avweight
def normalize(key):
    return "".join(ch.lower() for ch in str(key) if ch.isalnum())

aliases = {
    "V-V": {"vvweight", "vv", "visualvisualweight"},
    "A-A": {"aaweight", "aa", "audioaudioweight"},
    "A-V": {"avweight", "av", "audiovisualweight"},
}

def collect_weights(obj, found):
    if isinstance(obj, dict):
        for key, value in obj.items():
            k = normalize(key)
            for agent, names in aliases.items():
                if k in names and isinstance(value, (int, float)):
                    found[agent] = float(value)
            collect_weights(value, found)
    elif isinstance(obj, list):
        for item in obj:
            collect_weights(item, found)

rows = []
skipped = 0

for i, record in enumerate(records):
    found = {}
    collect_weights(record, found)

    if all(agent in found for agent in aliases):
        rows.append({
            "video_index": i,
            "V-V": found["V-V"],
            "A-A": found["A-A"],
            "A-V": found["A-V"],
        })
    else:
        skipped += 1

if not rows:
    print("No complete weight records found.")
    print("Please inspect the JSON structure with:")
    print('python -c "import json; d=json.load(open(\'checkpoints/temporal_evidence_results.json\',encoding=\'utf-8\')); print(json.dumps(d[0] if isinstance(d,list) else d,indent=2)[:6000])"')
    raise SystemExit(1)

df = pd.DataFrame(rows)

agents = ["V-V", "A-A", "A-V"]

# Check valid values
for agent in agents:
    if not df[agent].map(
        lambda x: pd.notna(x) and x >= 0
    ).all():
        raise ValueError(f"Invalid weights found for {agent}")

# Save extracted per-video weights
csv_path = OUT_DIR / "adaptive_agent_weights.csv"
df.to_csv(csv_path, index=False)

# Create box plot
fig, ax = plt.subplots(figsize=(9, 6))

box = ax.boxplot(
    [df[a].values for a in agents],
    tick_labels=agents,
    patch_artist=True,
    showmeans=True,
    showfliers=True,
    medianprops={"color": "#102A43", "linewidth": 2},
    meanprops={
        "marker": "D",
        "markerfacecolor": "#F4A261",
        "markeredgecolor": "#F4A261",
        "markersize": 5,
    },
)

for patch, color in zip(
    box["boxes"], ["#91CDE8", "#145DA0", "#2E86C1"]
):
    patch.set_facecolor(color)

ax.set_title(
    "Adaptive Agent Weight Distribution on Test Videos",
    fontsize=14,
    fontweight="bold",
    pad=14,
)
ax.set_xlabel("Agent")
ax.set_ylabel("Adaptive Weight")
ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)

plt.tight_layout()

png_path = OUT_DIR / "adaptive_agent_weights_boxplot.png"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nGraph created successfully!")
print(f"Videos plotted: {len(df)}")
print(f"Records skipped: {skipped}")
print(f"Graph: {png_path}")
print(f"CSV: {csv_path}")
print("\nWeight distribution summary:")
print(df[agents].describe().round(4).to_string())

# Check if weights sum to approximately 1 per video
weight_sum = df[agents].sum(axis=1)
print(f"\nMean sum of three weights: {weight_sum.mean():.4f}")
print(f"Videos where weights sum to ~1: "
      f"{((weight_sum - 1).abs() < 0.02).sum()} / {len(df)}")