import os
import pandas
import seaborn
from matplotlib import pyplot

from Encoder import VecEncoder
from Encoder import LabelEncoder
from Filter import CorrFilter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT_DIR, "Results", "EDA")
IN_PATH  = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
os.makedirs(OUT_PATH, exist_ok = True)
dataset = pandas.read_csv(IN_PATH)

# NOTE: Label distribution
labels = dataset["Churn"]
nChurn    = int(labels.sum())
nNoChurn  = len(labels) - nChurn
x = ["Churn", "No Churn"]
y = [nChurn, nNoChurn]
print("Label Ratio:", nNoChurn / nChurn)
colours = ["#d32f2f", "#1976d2"]
fig, ax = pyplot.subplots(figsize = (7, 4))
bars = ax.bar(x = x, height = y, color = colours, width = 0.5)
for bar, val in zip(bars, y):
    pct = val / len(labels)
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(y) * 0.02,
            f"{val}\n({pct:.1%})",
            ha = "center", va = "bottom", fontsize = 10, fontweight = "bold")
ax.set_title("Class Distribution", fontsize = 12, fontweight = "bold")
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, max(y) * 1.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
pyplot.tight_layout()
pyplot.savefig(f"{OUT_PATH}/LabelDistrib.png", dpi = 600, bbox_inches = "tight")
pyplot.close()

# NOTE: Encoding / Filtering
encoder = VecEncoder()
dataset = encoder(dataset, "State")
encoder = LabelEncoder({"Yes": 1, "No": 0})
dataset = encoder(dataset, "Voice mail plan")
dataset = encoder(dataset, "International plan")
filter = CorrFilter()
dataset = filter(dataset)

# NOTE: Correlation
corr = abs(dataset.corr())
pyplot.matshow(corr)
pyplot.savefig(f"{OUT_PATH}/Correl-Postfilter.png")

# NOTE: Plots
fig, ax = pyplot.subplots(1, 1, figsize=(18, 18))
dataset.hist(ax = ax, bins = 100)
pyplot.savefig(f"{OUT_PATH}/Distrib.png")
pyplot.close()
